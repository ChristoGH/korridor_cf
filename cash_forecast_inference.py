import argparse
import boto3
import pandas as pd
import numpy as np
import os
import logging
import sys
from time import gmtime, strftime
from sagemaker import Session
from typing import List
from dataclasses import dataclass, field
import yaml
from time import sleep
from pydantic import BaseModel, ValidationError

class ConfigModel(BaseModel):
    """Pydantic model for configuration validation"""
    region: str
    bucket: str
    prefix: str
    role_arn: str
    instance_type: str = "ml.m5.large"
    instance_count: int = 1
    forecast_horizon: int = 10
    forecast_frequency: str = "1D"
    batch_size: int = 100
    quantiles: List[str] = ['p10', 'p50', 'p90']


@dataclass
class Config:
    """Configuration parameters for the forecasting pipeline."""
    region: str
    bucket: str
    prefix: str
    role_arn: str
    instance_type: str = "ml.m5.large"
    instance_count: int = 1
    forecast_horizon: int = 10
    forecast_frequency: str = "1D"
    batch_size: int = 100
    quantiles: List[str] = field(default_factory=lambda: ['p10', 'p50', 'p90'])

class CashForecastingInference:
    def __init__(self, config: Config):
        """Initialize the inference pipeline with configuration."""
        self.config = config
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=config.region)
        self.s3_client = boto3.client('s3')
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = config.role_arn
        self._setup_logging()
        self.output_dir = None  # Will be set in prepare_inference_data

    def _setup_logging(self):
        """Configure logging for the pipeline."""
        self.logger = logging.getLogger('CashForecastInference')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # File handler
        fh = logging.FileHandler(f'cash_forecast_inference_{self.timestamp}.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        # Stream handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _safe_s3_upload(self, local_path: str, s3_key: str, overwrite: bool = False) -> None:
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

    def prepare_inference_data(self, inference_template_file: str, country_code: str) -> str:
        """Prepare the inference data based on the template."""
        # Load inference template
        inference_template = pd.read_csv(inference_template_file)

        # Ensure correct data types
        inference_template['ProductId'] = inference_template['ProductId'].astype(str)
        inference_template['BranchId'] = inference_template['BranchId'].astype(str)
        inference_template['Currency'] = inference_template['Currency'].astype(str)

        # Generate future dates for the forecast horizon
        last_date = pd.Timestamp.today().normalize()  # Use today's date or specify a date
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, self.config.forecast_horizon + 1)]

        # Number of combinations
        num_combinations = len(inference_template)

        # Create inference data
        inference_data = pd.DataFrame({
            'ProductId': np.tile(inference_template['ProductId'].values, self.config.forecast_horizon),
            'BranchId': np.tile(inference_template['BranchId'].values, self.config.forecast_horizon),
            'Currency': np.tile(inference_template['Currency'].values, self.config.forecast_horizon),
            'EffectiveDate': np.repeat(future_dates, num_combinations),
            'Demand': np.nan  # Demand is unknown for future dates
        })

        # Create directories for output
        self.output_dir = f"./cs_data/cash/{country_code}_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Save inference data
        inference_file = os.path.join(self.output_dir, f"{country_code}_inference.csv")
        inference_data.to_csv(inference_file, index=False)
        return inference_file

    def run_inference(self, country_code: str, model_name: str, inference_file: str) -> None:
        """Run the inference process."""
        # Upload inference data to S3
        s3_inference_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference/{os.path.basename(inference_file)}"
        self._safe_s3_upload(inference_file, s3_inference_key, overwrite=True)
        s3_inference_data_uri = f"s3://{self.config.bucket}/{s3_inference_key}"

        # Run batch transform job
        self._run_batch_transform_job(country_code, model_name, s3_inference_data_uri)

        # Retrieve and process results
        forecast_df = self._get_forecast_result(country_code)

        # Save forecasts
        self._save_forecasts(country_code, forecast_df)

    def _run_batch_transform_job(self, country_code: str, model_name: str, s3_inference_data_uri: str) -> None:
        """Run a batch transform job for the given inference data."""
        transform_job_name = f"{model_name}-transform-{self.timestamp}"

        output_s3_uri = f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/"

        # Start batch transform job
        try:
            self.sm_client.create_transform_job(
                TransformJobName=transform_job_name,
                ModelName=model_name,
                BatchStrategy='MultiRecord',
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
            self._monitor_transform_job(transform_job_name)
        except Exception as e:
            self.logger.error(f"Failed to create transform job {transform_job_name}: {e}")
            raise

    def _monitor_transform_job(self, transform_job_name: str) -> None:
        """Monitor the batch transform job until completion."""
        self.logger.info(f"Monitoring transform job {transform_job_name}")
        sleep_time = 30  # Start with 30 seconds
        while True:
            try:
                response = self.sm_client.describe_transform_job(TransformJobName=transform_job_name)
                status = response['TransformJobStatus']
                self.logger.info(f"Transform job {transform_job_name} status: {status}")
                if status in ['Completed', 'Failed', 'Stopped']:
                    if status != 'Completed':
                        failure_reason = response.get('FailureReason', 'No failure reason provided.')
                        self.logger.error(f"Transform job {transform_job_name} failed: {failure_reason}")
                        raise RuntimeError(
                            f"Transform job {transform_job_name} failed with status: {status}. Reason: {failure_reason}")
                    break
                sleep(sleep_time)
                sleep_time = min(sleep_time * 1.5, 600)  # Increase sleep time up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring transform job: {e}")
                sleep(60)  # Wait before retrying

    def _get_forecast_result(self, country_code: str) -> pd.DataFrame:
        """Download and process forecast results."""
        output_s3_prefix = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/"
        # The rest of the method is similar to your original _get_forecast_result method
        # Make sure to adjust any references to batches or multiple files if not applicable
        pass  # Implement as per your original code

    def _save_forecasts(self, country_code: str, forecasts_df: pd.DataFrame) -> None:
        """Save the forecasts with appropriate format."""
        # Similar to your original _save_forecasts method
        pass  # Implement as per your original code

def main():
    parser = argparse.ArgumentParser(description='Cash Forecasting Inference')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--countries', nargs='+', default=['ZM'],
                        help='Country codes to process')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the existing SageMaker model to use for inference')
    parser.add_argument('--inference_template', type=str, required=True,
                        help='Path to the inference template CSV file')
    args = parser.parse_args()

    # Load and validate configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    try:
        # Validate with Pydantic and create Config dataclass
        config_model = ConfigModel(**config_dict)
        config = Config(**config_model.dict())
    except Exception as e:
        logging.error(f"Configuration error: {str(e)}")
        sys.exit(1)

    for country_code in args.countries:
        inference_pipeline = CashForecastingInference(config)
        try:
            inference_file = inference_pipeline.prepare_inference_data(
                args.inference_template, country_code)
            inference_pipeline.run_inference(country_code, args.model_name, inference_file)
        except Exception as e:
            logging.error(f"Failed to process {country_code}: {str(e)}")

if __name__ == "__main__":
    main()
