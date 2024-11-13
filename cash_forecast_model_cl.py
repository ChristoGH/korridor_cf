import argparse
import boto3
import pandas as pd  # Ensure this import is before using 'pd'
import datetime
import os
import logging
import sys
from time import gmtime, strftime, sleep
from sagemaker import get_execution_role, Session
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import yaml
import tempfile


@dataclass
class Config:
    region: str
    bucket: str
    prefix: str
    instance_type: str = "ml.m5.large"
    instance_count: int = 1
    forecast_horizon: int = 1  # Keep as 1 for one-step-ahead
    forecast_frequency: str = "1D"
    iterative_forecast_steps: int = 20  # Number of steps to forecast


class CashForecastingPipeline:
    def __init__(self, config: Config):
        """Initialize the forecasting pipeline with configuration"""
        self.config = config
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=config.region)
        self.s3_client = boto3.client('s3')
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = get_execution_role()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the pipeline"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'cash_forecast_{self.timestamp}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('CashForecast')

    def _ensure_column_types(self,
                             data: pd.DataFrame,
                             columns: List[str],
                             prefixes: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Ensure column types and add prefixes where specified"""
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' missing.")
            prefix = (prefixes or {}).get(col, '')
            data[col] = prefix + data[col].astype(str)
        return data

    def _safe_s3_upload(self,
                        local_path: str,
                        s3_key: str,
                        overwrite: bool = False) -> None:
        """Safely upload file to S3 with existence check"""
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

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str, str]:
        """Prepare data for iterative one-step-ahead forecasting"""
        data = pd.read_csv(input_file)
        data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate'])
        data.sort_values('EffectiveDate', inplace=True)

        # Calculate cutoff date for training (m - t)
        latest_date = data['EffectiveDate'].max()
        cutoff_date = latest_date - pd.Timedelta(days=self.config.iterative_forecast_steps)

        # Split data
        train_data = data[data['EffectiveDate'] <= cutoff_date].copy()
        future_data = data[data['EffectiveDate'] > cutoff_date].copy()

        # Save files
        output_dir = f"./cs_data/cash/{country_code}_{self.timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        paths = {
            'train': os.path.join(output_dir, f"{country_code}_train.csv"),
            'future': os.path.join(output_dir, f"{country_code}_future.csv"),
            'inference_template': os.path.join(output_dir, f"{country_code}_inference_template.csv")
        }

        train_data.to_csv(paths['train'], index=False)
        future_data.to_csv(paths['future'], index=False)

        # Create inference template (one row per product/branch combination)
        inference_template = data.drop_duplicates(
            subset=['ProductId', 'BranchId', 'CountryCode', 'Currency']
        )[['ProductId', 'BranchId', 'CountryCode', 'Currency']]
        inference_template.to_csv(paths['inference_template'], index=False)

        return paths['train'], paths['future'], paths['inference_template']

    def train_model(self,
                    country_code: str,
                    train_data_s3_uri: str) -> str:
        """Train the forecasting model"""
        self.logger.info(f"Starting model training for {country_code}")

        job_name = f"{country_code}-ts-{self.timestamp}"

        # Configure AutoML job
        automl_config = {
            'AutoMLJobName': job_name,
            'AutoMLJobInputDataConfig': [{
                'ChannelType': 'training',
                'ContentType': 'text/csv;header=present',
                'CompressionType': 'None',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': train_data_s3_uri
                    }
                }
            }],
            'OutputDataConfig': {
                'S3OutputPath': f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/output"
            },
            'AutoMLProblemTypeConfig': {
                'TimeSeriesForecastingJobConfig': {
                    'ForecastFrequency': self.config.forecast_frequency,
                    'ForecastHorizon': self.config.forecast_horizon,
                    'ForecastQuantiles': ['p10', 'p50', 'p90'],
                    'TimeSeriesConfig': {
                        'TargetAttributeName': 'Demand',
                        'TimestampAttributeName': 'EffectiveDate',
                        'ItemIdentifierAttributeName': 'ProductId',
                        'GroupingAttributeNames': ['BranchId', 'CountryCode', 'Currency']
                    },
                    'HolidayConfig': [{'CountryCode': country_code}]
                }
            },
            'RoleArn': self.role
        }

        # Start training
        self.sm_client.create_auto_ml_job_v2(**automl_config)
        return job_name

    def _monitor_job(self, job_name: str) -> str:
        """Monitor the AutoML job until completion"""
        self.logger.info(f"Monitoring job {job_name}")
        while True:
            response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
            status = response['AutoMLJobStatus']
            self.logger.info(f"Job {job_name} status: {status}")
            if status in ['Completed', 'Failed', 'Stopped']:
                break
            sleep(120)
        return status

    def _get_best_model(self, job_name: str, country_code: str) -> str:
        """Retrieve the best model from the AutoML job"""
        response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
        best_candidate = response['BestCandidate']
        model_name = f"{country_code}-model-{self.timestamp}"

        # Check if model already exists
        existing_models = self.sm_client.list_models(NameContains=model_name)
        if existing_models['Models']:
            raise ValueError(f"A model with name {model_name} already exists. Aborting to prevent overwriting.")

        # Create model
        self.sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=self.role,
            Containers=best_candidate['InferenceContainers']
        )
        self.logger.info(f"Created model {model_name}")
        return model_name

    def iterative_forecast(self, country_code: str, model_name: str, train_file: str, template_file: str) -> pd.DataFrame:
        """Implement proper one-step-ahead forecasting"""
        # Load initial data
        historical_data = pd.read_csv(train_file)
        historical_data['EffectiveDate'] = pd.to_datetime(historical_data['EffectiveDate'])
        template = pd.read_csv(template_file)
        
        # Initialize forecast collection
        all_forecasts = []
        current_data = historical_data.copy()
        
        # Generate forecast dates
        last_date = historical_data['EffectiveDate'].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.config.iterative_forecast_steps,
            freq=self.config.forecast_frequency
        )
        
        # Ensure unique identification of rows
        key_columns = ['ProductId', 'BranchId', 'CountryCode', 'Currency', 'EffectiveDate']
        
        for forecast_date in forecast_dates:
            # Prepare inference data for current step
            inference_data = template.copy()
            inference_data['EffectiveDate'] = forecast_date
            inference_data['Demand'] = None
            
            # Drop any potential duplicates in current_data before concatenation
            current_data = current_data.drop_duplicates(subset=key_columns)
            
            # Ensure inference_data has no duplicates
            inference_data = inference_data.drop_duplicates(subset=key_columns[:-1])  # Exclude EffectiveDate
            
            # Combine with current data, ensuring no duplicates
            combined_data = pd.concat([current_data, inference_data], ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=key_columns)
            
            # Run inference for current step
            forecast = self._run_single_forecast(
                country_code,
                model_name,
                combined_data,
                forecast_date
            )
            
            # Add forecast to collection
            forecast['EffectiveDate'] = forecast_date
            all_forecasts.append(forecast)
            
            # Merge forecasted demand back to inference data
            forecast_with_data = pd.merge(
                inference_data,
                forecast[['ProductId', 'BranchId', 'CountryCode', 'Currency', 'EffectiveDate', 'ForecastDemand']],
                on=['ProductId', 'BranchId', 'CountryCode', 'Currency', 'EffectiveDate'],
                how='left'
            )
            
            # Rename 'ForecastDemand' to 'Demand' for consistency
            forecast_with_data.rename(columns={'ForecastDemand': 'Demand'}, inplace=True)
            
            # Update current data with the new forecasted demand, ensuring no duplicates
            current_data = pd.concat([current_data, forecast_with_data], ignore_index=True)
            current_data = current_data.drop_duplicates(subset=key_columns, keep='last')
        
        # Compile all forecasts
        final_forecasts = pd.concat(all_forecasts, ignore_index=True)
        self._save_and_visualize_forecasts(country_code, final_forecasts)
        return final_forecasts
    def _run_single_forecast(self, country_code: str, model_name: str,
                             combined_data: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.DataFrame:
        """Run single step forecast"""
        # Save combined data to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            combined_data.to_csv(tmp_file.name, index=False)
            inference_file = tmp_file.name

        # Upload to S3
        s3_inference_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference/{forecast_date.strftime('%Y%m%d')}.csv"
        self._safe_s3_upload(inference_file, s3_inference_key, overwrite=True)
        s3_inference_data_uri = f"s3://{self.config.bucket}/{s3_inference_key}"

        # Run batch transform job
        self._run_batch_transform_job(country_code, model_name, s3_inference_data_uri, forecast_date)

        # Get forecast result
        forecast = self._get_forecast_result(country_code, forecast_date)

        # Clean up temporary file
        os.unlink(inference_file)

        return forecast

    def _run_batch_transform_job(self, country_code: str, model_name: str,
                                 s3_inference_data_uri: str, forecast_date: pd.Timestamp) -> None:
        """Run a batch transform job for the given inference data"""
        self.logger.info(f"Running batch transform job for date {forecast_date} in {country_code}")
        transform_job_name = f"{model_name}-transform-{self.timestamp}-{forecast_date.strftime('%Y%m%d')}"

        output_s3_uri = f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/{forecast_date.strftime('%Y%m%d')}/"

        # Start batch transform job
        self.sm_client.create_transform_job(
            TransformJobName=transform_job_name,
            ModelName=model_name,
            TransformInput={
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': s3_inference_data_uri
                    }
                },
                'ContentType': 'text/csv',
                'SplitType': 'Line'
            },
            TransformOutput={
                'S3OutputPath': output_s3_uri,
                'AssembleWith': 'Line'
            },
            TransformResources={
                'InstanceType': self.config.instance_type,
                'InstanceCount': self.config.instance_count
            }
        )

        # Monitor transform job
        self._monitor_transform_job(transform_job_name)
    def _monitor_transform_job(self, transform_job_name: str) -> None:
        """Monitor the batch transform job until completion"""
        self.logger.info(f"Monitoring transform job {transform_job_name}")
        while True:
            response = self.sm_client.describe_transform_job(TransformJobName=transform_job_name)
            status = response['TransformJobStatus']
            self.logger.info(f"Transform job {transform_job_name} status: {status}")
            if status in ['Completed', 'Failed', 'Stopped']:
                break
            sleep(60)
        if status != 'Completed':
            raise RuntimeError(f"Transform job {transform_job_name} failed with status: {status}")

    def _get_forecast_result(self, country_code: str, forecast_date: pd.Timestamp) -> pd.DataFrame:
        """Download forecast result for the given date"""
        output_s3_prefix = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/{forecast_date.strftime('%Y%m%d')}/"
        response = self.s3_client.list_objects_v2(Bucket=self.config.bucket, Prefix=output_s3_prefix)
        if 'Contents' not in response:
            raise FileNotFoundError(f"No inference results found in S3 for date {forecast_date}.")

        # Download the output file
        for obj in response['Contents']:
            s3_key = obj['Key']
            if s3_key.endswith('.out'):
                local_file_path = f"./results/{country_code}_{self.timestamp}/{forecast_date.strftime('%Y%m%d')}.csv"
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                self.s3_client.download_file(self.config.bucket, s3_key, local_file_path)
                self.logger.info(f"Downloaded {s3_key} to {local_file_path}")

                # Load the forecast result
                forecast_df = pd.read_csv(local_file_path)
                # Select the desired forecast column
                forecast_df['ForecastDemand'] = forecast_df['p50']  # Or 'mean', 'p10', 'p90'
                # Keep only necessary columns
                forecast_df = forecast_df[['ProductId', 'BranchId', 'CountryCode', 'Currency', 'EffectiveDate', 'ForecastDemand']]
                return forecast_df

        raise FileNotFoundError(f"No forecast output file found for date {forecast_date}.")

    def _save_and_visualize_forecasts(self, country_code: str, forecasts_df: pd.DataFrame) -> None:
        """Save and visualize the collected forecasts"""
        # Save forecasts to CSV
        forecast_output_file = f"./results/{country_code}_{self.timestamp}/iterative_forecasts.csv"
        os.makedirs(os.path.dirname(forecast_output_file), exist_ok=True)
        forecasts_df.to_csv(forecast_output_file, index=False)

        # Plot forecasts
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(forecasts_df['EffectiveDate'], forecasts_df['ForecastDemand'], label='Forecast Demand')

        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.title(f'Iterative One-Step-Ahead Forecasts for {country_code}')
        plt.legend()
        plt.savefig(f"./results/{country_code}_{self.timestamp}/forecast_plot.png")
        plt.close()

    def run_pipeline(self, country_code: str, input_file: str) -> None:
        """Run the complete forecasting pipeline with iterative forecasting"""
        try:
            # Prepare data
            train_file, future_file, template_file = self.prepare_data(input_file, country_code)

            # Upload training data to S3
            s3_train_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/train/{os.path.basename(train_file)}"
            self._safe_s3_upload(train_file, s3_train_key)
            train_data_s3_uri = f"s3://{self.config.bucket}/{os.path.dirname(s3_train_key)}/"

            # Train model
            job_name = self.train_model(
                country_code,
                train_data_s3_uri
            )

            # Monitor training
            status = self._monitor_job(job_name)
            if status != 'Completed':
                raise RuntimeError(f"Training failed with status: {status}")

            # Get best model
            model_name = self._get_best_model(job_name, country_code)

            # Iterative Forecasting
            self.iterative_forecast(country_code, model_name, train_file, template_file)

        except Exception as e:
            self.logger.error(f"Pipeline failed for {country_code}: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Cash Forecasting Pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--countries', nargs='+', default=['ZM'],
                        help='Country codes to process')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)

    # Initialize and run pipeline
    pipeline = CashForecastingPipeline(config)

    for country_code in args.countries:
        try:
            input_file = f"./data/cash/{country_code}.csv"
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            pipeline.run_pipeline(country_code, input_file)

        except Exception as e:
            logging.error(f"Failed to process {country_code}: {str(e)}")


if __name__ == "__main__":
    main()
