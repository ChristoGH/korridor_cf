# reuse_cash_forecast_model.py
# python reuse_cash_forecast_model.py --config cs_scripts/config.yaml --countries ZM --model-name ZM-model-20241118-025905 --cleanup
# python cs_scripts/reuse_cash_forecast_model.py --config cs_scripts/config.yaml --countries ZM --model-name ZM-model-20241118-025905 --cleanup
# python cs_scripts/reuse_cash_forecast_model.py --config cs_scripts/config.yaml --countries ZM --model-name ZM-model-20241118-025905
import argparse
import boto3
import pandas as pd
import numpy as np
import datetime
import os
import logging
import sys
from time import gmtime, strftime, sleep
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import yaml
import tempfile
import glob
from pydantic import BaseModel, ValidationError
from sagemaker import Session

import io  # For handling string inputs/outputs



class ConfigModel(BaseModel):
    """Pydantic model for configuration validation"""
    region: str
    bucket: str
    prefix: str
    role_arn: str
    instance_type: str = "ml.c5.4xlarge"
    instance_count: int = 1
    forecast_horizon: int = 10
    forecast_frequency: str = "1D"
    batch_size: int = 50
    quantiles: List[str] = ['p10', 'p50', 'p90']



@dataclass
class Config:
    """Configuration parameters for the forecasting pipeline."""
    region: str
    bucket: str
    prefix: str
    role_arn: str
    instance_type: str = "ml.c5.4xlarge"
    instance_count: int = 1
    forecast_horizon: int = 10
    forecast_frequency: str = "1D"
    batch_size: int = 10
    quantiles: List[str] = ('p10', 'p50', 'p90')



class ModelReusePipeline:
    def __init__(self, config: Config):
        self.config = config
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=config.region)
        self.s3_client = boto3.client('s3')
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = config.role_arn
        self._setup_logging()
        self.output_dir = None


    def _setup_logging(self):
        """Configure logging for the pipeline."""
        self.logger = logging.getLogger('CashForecast')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(f'cash_forecast_reuse_{self.timestamp}.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _safe_s3_upload(self,
                        local_path: str,
                        s3_key: str,
                        overwrite: bool = False) -> None:
        """Safely upload file to S3 with existence check."""
        try:
            if not overwrite:
                existing = self.s3_client.list_objects_v2(
                    Bucket=self.config.bucket,
                    Prefix=s3_key
                )
                if 'Contents' in existing:
                    raise FileExistsError(f"S3 object already exists: {s3_key}")

            self.s3_client.upload_file(
                Filename=local_path,
                Bucket=self.config.bucket,
                Key=s3_key
            )
            self.logger.info(f"Uploaded {local_path} to s3://{self.config.bucket}/{s3_key}")
        except Exception as e:
            self.logger.error(f"Failed to upload {local_path} to S3: {e}")
            raise

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str]:
        """Prepare data for inference."""
        self.logger.info(f"Preparing data from input file: {input_file} for country: {country_code}")
        data = pd.read_csv(input_file)

        required_columns = ['BranchId', 'CountryCode', 'ProductId', 'Currency', 'EffectiveDate', 'Demand']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Input data is missing required column: {col}")

        # Add CountryName column
        country_names = {'ZM': 'Zambia'}  # Add other countries as needed
        data['CountryName'] = data['CountryCode'].map(country_names)

        data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate']).dt.tz_localize(None)
        # Add DayOfWeekName
        data['DayOfWeekName'] = data['EffectiveDate'].dt.day_name()

        data.sort_values('EffectiveDate', inplace=True)

        self.output_dir = f"./cs_data/cash/{country_code}_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        data_file = os.path.join(self.output_dir, f"{country_code}_data.csv")
        inference_template_file = os.path.join(self.output_dir, f"{country_code}_inference_template.csv")

        data.to_csv(data_file, index=False)

        inference_template = data.drop_duplicates(
            subset=['ProductId', 'BranchId', 'CountryCode', 'Currency']
        )[['ProductId', 'BranchId', 'CountryCode', 'CountryName', 'Currency']]

        inference_template.to_csv(inference_template_file, index=False)

        self.logger.info(f"Data prepared and saved to {data_file} and {inference_template_file}")
        return data_file, inference_template_file

    def generate_forecasts(self, country_code: str, model_name: str, data_file: str,
                           template_file: str) -> pd.DataFrame:
        """Generate forecasts using existing model."""
        self.logger.info(f"Generating forecasts for country: {country_code}")
        inference_template = pd.read_csv(template_file)
        data = pd.read_csv(data_file)
        data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate'])

        # Add CountryName if not present
        if 'CountryName' not in inference_template.columns:
            country_names = {'ZM': 'Zambia'}  # Add other countries as needed
            inference_template['CountryName'] = inference_template['CountryCode'].map(country_names)

        effective_dates = sorted(data['EffectiveDate'].unique())

        batch_number = 0
        all_forecasts = pd.DataFrame()

        for i in range(0, len(effective_dates), self.config.batch_size):
            batch_number += 1
            batch_dates = effective_dates[i:i + self.config.batch_size]
            inference_data_list = []

            for effective_date in batch_dates:
                future_dates = [effective_date + pd.Timedelta(days=j)
                                for j in range(1, self.config.forecast_horizon + 1)]

                num_combinations = len(inference_template)

                # Create base DataFrame with all required columns
                temp_df = pd.DataFrame({
                    'ProductId': np.tile(inference_template['ProductId'].values, self.config.forecast_horizon),
                    'BranchId': np.tile(inference_template['BranchId'].values, self.config.forecast_horizon),
                    'CountryCode': np.tile(inference_template['CountryCode'].values, self.config.forecast_horizon),
                    'CountryName': np.tile(inference_template['CountryName'].values, self.config.forecast_horizon),
                    'Currency': np.tile(inference_template['Currency'].values, self.config.forecast_horizon),
                    'EffectiveDate': effective_date,
                    'ForecastDate': np.repeat(future_dates, num_combinations),
                    'Demand': 0,  # placeholder for predictions
                    'DayOfWeekName': pd.Series(np.repeat(future_dates, num_combinations)).dt.day_name()
                })
                inference_data_list.append(temp_df)

            inference_data = pd.concat(inference_data_list, ignore_index=True)

            self.logger.info(f"Processing batch number {batch_number} with {len(inference_data)} records")
            # Process batch
            forecast_batch = self._process_forecast_batch(
                country_code, model_name, inference_data, batch_number)

            if batch_number == 1:
                all_forecasts = forecast_batch
            else:
                all_forecasts = pd.concat([all_forecasts, forecast_batch], ignore_index=True)

        self._save_results(country_code, all_forecasts, data)
        return all_forecasts

    def _process_forecast_batch(self, country_code: str, model_name: str,
                                inference_data: pd.DataFrame, batch_number: int) -> pd.DataFrame:
        """Process using batch transform instead of real-time endpoint"""
        required_columns = ['BranchId', 'CountryCode', 'CountryName', 'ProductId',
                            'Currency', 'EffectiveDate', 'Demand', 'DayOfWeekName']
        inference_data = inference_data[required_columns].copy()

        temp_csv = f"temp_inference_{batch_number}.csv"
        inference_data.to_csv(temp_csv, index=False, header=True)

        # Use consistent path structure
        input_prefix = f"{self.config.prefix}-{country_code}/inference"
        output_prefix = f"{self.config.prefix}-{country_code}/output"

        s3_input_key = f"{input_prefix}/batch_{batch_number}.csv"
        self._safe_s3_upload(temp_csv, s3_input_key, overwrite=True)

        transform_job_name = f"{model_name}-transform-{self.timestamp}-{batch_number}"

        try:
            self.sm_client.create_transform_job(
                TransformJobName=transform_job_name,
                ModelName=model_name,
                MaxPayloadInMB=50,
                BatchStrategy='MultiRecord',
                MaxConcurrentTransforms=2,
                TransformInput={
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f"s3://{self.config.bucket}/{input_prefix}/"
                        }
                    },
                    'ContentType': 'text/csv',
                    'SplitType': 'Line'
                },
                TransformOutput={
                    'S3OutputPath': f"s3://{self.config.bucket}/{output_prefix}/",
                    'Accept': 'text/csv',
                    'AssembleWith': 'Line'
                },
                TransformResources={
                    'InstanceType': self.config.instance_type,
                    'InstanceCount': 1
                }
            )

            # Monitor job status
            retry_delay = 30
            max_retries = 10
            retries = 0

            while retries < max_retries:
                response = self.sm_client.describe_transform_job(TransformJobName=transform_job_name)
                status = response['TransformJobStatus']
                self.logger.info(f"Transform job status: {status}")

                if status == 'Completed':
                    break
                elif status in ['Failed', 'Stopped']:
                    failure_reason = response.get('FailureReason', 'Unknown reason')
                    raise Exception(f"Transform job failed: {failure_reason}")

                retries += 1
                sleep(min(retry_delay * (2 ** retries), 300))

                if retries == max_retries:
                    raise Exception("Maximum retries reached waiting for transform job")

            # List objects in output prefix to find the results
            paginator = self.s3_client.get_paginator('list_objects_v2')
            output_files = []

            for page in paginator.paginate(
                    Bucket=self.config.bucket,
                    Prefix=f"{output_prefix}/batch_{batch_number}"
            ):
                if 'Contents' in page:
                    output_files.extend(page['Contents'])

            if not output_files:
                raise FileNotFoundError(f"No output files found in {output_prefix}")

            # Download the first matching output file
            output_key = output_files[0]['Key']
            local_output = f"output_{batch_number}.csv"
            self.s3_client.download_file(self.config.bucket, output_key, local_output)

            results = pd.read_csv(local_output)

            # Cleanup
            os.remove(local_output)
            os.remove(temp_csv)

            self.logger.info(f"Batch {batch_number} processed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Error in batch transform: {str(e)}")
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            raise


    def _save_results(self, country_code: str, forecasts: pd.DataFrame,
                      historical_data: pd.DataFrame) -> None:
        """Save final results."""
        output_dir = f"./results/{country_code}_reuse_{self.timestamp}/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "final_forecast.csv")

        # Merge forecasts with historical data if necessary
        # For this example, we'll save the forecasts directly
        forecasts.to_csv(output_file, index=False)
        self.logger.info(f"Final forecast saved to {output_file}")




def main():
    parser = argparse.ArgumentParser(description='Cash Forecast Model Reuse Pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--countries', nargs='+', default=['ZM'],
                        help='Country codes to process')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of existing model to use')
    args = parser.parse_args()  # Remove --cleanup argument as it's not needed

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    try:
        config_model = ConfigModel(**config_dict)
        config = Config(**config_model.dict())
    except ValidationError as e:
        logging.error(f"Configuration validation error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Configuration error: {str(e)}")
        sys.exit(1)

    pipeline = ModelReusePipeline(config)

    for country_code in args.countries:
        try:
            pipeline.logger.info(f"Processing country: {country_code}")
            input_file = f"./data/cash/{country_code}.csv"
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            data_file, template_file = pipeline.prepare_data(input_file, country_code)
            pipeline.generate_forecasts(country_code, args.model_name,
                                     data_file, template_file)

        except Exception as e:
            pipeline.logger.error(f"Failed to process {country_code}: {str(e)}")
            raise

if __name__ == "__main__":
    main()
