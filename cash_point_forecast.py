import pandas as pd
import boto3
import os
from datetime import datetime, timedelta
import yaml
from time import gmtime, strftime
import json
import time
from typing import Dict, Optional, Tuple


class PointForecastGenerator:
    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sm_client = boto3.client('sagemaker', region_name=self.config['region'])
        self.s3_client = boto3.client('s3', region_name=self.config['region'])
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())

    def _prepare_inference_data(
            self,
            product_id: str,
            branch_id: str,
            currency: str,
            effective_date: str
    ) -> pd.DataFrame:
        """Prepare inference data matching the main pipeline format."""
        effective_dt = pd.to_datetime(effective_date).tz_localize(None)

        # Generate future dates matching the main pipeline
        future_dates = [
            effective_dt + pd.Timedelta(days=j)
            for j in range(1, self.config['forecast_horizon'] + 1)
        ]

        # Create inference data matching main pipeline structure
        inference_data = pd.DataFrame({
            'ProductId': [str(product_id)] * len(future_dates),
            'BranchId': [str(branch_id)] * len(future_dates),
            'Currency': [currency] * len(future_dates),
            'EffectiveDate': [effective_dt] * len(future_dates),
            'ForecastDate': future_dates,
            'Demand': [None] * len(future_dates)
        })

        # Ensure correct data types matching main pipeline
        inference_data['ProductId'] = inference_data['ProductId'].astype(str)
        inference_data['BranchId'] = inference_data['BranchId'].astype(str)
        inference_data['Currency'] = inference_data['Currency'].astype(str)

        return inference_data

    def generate_forecast(
            self,
            product_id: str,
            branch_id: str,
            currency: str,
            effective_date: str,
            country_code: str,
            model_name: str
    ) -> Tuple[str, pd.DataFrame]:
        """Generate a point forecast matching main pipeline behavior."""
        temp_file = f'temp_inference_{self.timestamp}.csv'

        try:
            # Prepare inference data
            inference_data = self._prepare_inference_data(
                product_id, branch_id, currency, effective_date
            )

            print("\nInference data preview:")
            print(inference_data)

            # Save and upload to S3
            inference_data.to_csv(temp_file, index=False)
            s3_key = f'{self.config["prefix"]}-{country_code}/{self.timestamp}/inference/{os.path.basename(temp_file)}'
            self.s3_client.upload_file(temp_file, self.config['bucket'], s3_key)

            # Configure transform job to match main pipeline
            transform_job_name = f'point-forecast-{self.timestamp}'
            transform_config = {
                'TransformJobName': transform_job_name,
                'ModelName': model_name,
                'BatchStrategy': 'MultiRecord',  # Match main pipeline
                'TransformInput': {
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f's3://{self.config["bucket"]}/{s3_key}'
                        }
                    },
                    'ContentType': 'text/csv',  # Match main pipeline
                    'SplitType': 'Line'
                },
                'TransformOutput': {
                    'S3OutputPath': f's3://{self.config["bucket"]}/{self.config["prefix"]}-{country_code}/{self.timestamp}/inference-output/',
                    'AssembleWith': 'Line'
                },
                'TransformResources': {
                    'InstanceType': self.config.get('instance_type', 'ml.m5.large'),
                    'InstanceCount': 1
                }
            }

            self.sm_client.create_transform_job(**transform_config)
            print(f"\nTransform job started: {transform_job_name}")

            # Monitor job to completion
            status = self._monitor_transform_job(transform_job_name)
            if status != 'Completed':
                raise RuntimeError(f"Transform job failed with status: {status}")

            # Process results
            results = self._process_forecast_results(
                country_code, inference_data
            )

            return transform_job_name, results

        except Exception as e:
            print(f"\nError in generate_forecast: {str(e)}")
            raise
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _monitor_transform_job(self, job_name: str) -> str:
        """Monitor transform job with exponential backoff matching main pipeline."""
        print(f"Monitoring transform job: {job_name}")
        sleep_time = 30

        while True:
            try:
                response = self.sm_client.describe_transform_job(
                    TransformJobName=job_name
                )
                status = response['TransformJobStatus']
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status}")

                if status in ['Completed', 'Failed', 'Stopped']:
                    if status != 'Completed':
                        failure_reason = response.get('FailureReason', 'Unknown')
                        print(f"\nFailure reason: {failure_reason}")
                    break

                time.sleep(sleep_time)
                sleep_time = min(sleep_time * 1.5, 600)  # Match main pipeline backoff

            except Exception as e:
                print(f"Error monitoring transform job: {e}")
                time.sleep(60)

        return status

    def _process_forecast_results(
            self,
            country_code: str,
            inference_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Process forecast results matching main pipeline format."""
        output_s3_prefix = f"{self.config['prefix']}-{country_code}/{self.timestamp}/inference-output/"

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config['bucket'],
                Prefix=output_s3_prefix
            )

            if 'Contents' not in response:
                raise FileNotFoundError(f"No results found in {output_s3_prefix}")

            # Process forecast files
            forecast_data = []
            for obj in response['Contents']:
                if obj['Key'].endswith('.out'):
                    result = self.s3_client.get_object(
                        Bucket=self.config['bucket'],
                        Key=obj['Key']
                    )
                    df = pd.read_csv(result['Body'], header=None)
                    forecast_data.append(df)

            if not forecast_data:
                raise FileNotFoundError("No forecast output files found")

            # Combine forecasts and match main pipeline format
            forecast_df = pd.concat(forecast_data, ignore_index=True)
            forecast_df.columns = self.config.get('quantiles', ['p10', 'p50', 'p90'])

            # Combine with inference data
            results = pd.concat([
                inference_data[['ProductId', 'BranchId', 'Currency',
                                'EffectiveDate', 'ForecastDate']].reset_index(drop=True),
                forecast_df.reset_index(drop=True)
            ], axis=1)

            # Calculate ForecastDay to match main pipeline
            results['ForecastDay'] = (
                    (results['ForecastDate'] - results['EffectiveDate']).dt.days + 1
            )

            # Pivot results to match main pipeline format
            results_pivot = results.pivot_table(
                index=['ProductId', 'BranchId', 'Currency', 'EffectiveDate'],
                columns='ForecastDay',
                values=self.config.get('quantiles', ['p10', 'p50', 'p90'])
            )

            # Flatten column names
            results_pivot.columns = [
                f"{quantile}_Day{int(day)}"
                for quantile, day in results_pivot.columns
            ]

            return results_pivot.reset_index()

        except Exception as e:
            print(f"Error processing forecast results: {e}")
            raise


# Usage example
if __name__ == "__main__":
    try:
        forecaster = PointForecastGenerator('config.yaml')
        job_name, results = forecaster.generate_forecast(
            product_id="18",
            branch_id="11",
            currency="USD",
            effective_date="2023-05-01",
            country_code="ZM",
            model_name="your_model_name"
        )

        print("\nForecast results:")
        print(results)

    except Exception as e:
        print(f"Error: {str(e)}")