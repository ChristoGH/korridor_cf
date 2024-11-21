# python cs_scripts/cash_forecast_model.py --config cs_scripts/config.yaml --countries ZM

import argparse
import boto3
import pandas as pd
import numpy as np
import os
import logging
import sys
from time import gmtime, strftime, sleep
from sagemaker import Session
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import yaml
import glob
from pydantic import BaseModel, ValidationError
import shutil


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
    quantiles: List[str] = ('p10', 'p50', 'p90')


class CashForecastingPipeline:
    def __init__(self, config: Config):
        """Initialize the forecasting pipeline with configuration."""
        self.config = config
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=config.region)
        self.s3_client = boto3.client('s3')
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = config.role_arn
        self._setup_logging()
        self.train_file = None
        self.output_dir = None  # Will be set in prepare_data

    def _setup_logging(self):
        """Configure logging for the pipeline."""
        self.logger = logging.getLogger('CashForecast')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # File handler
        fh = logging.FileHandler(f'cash_forecast_{self.timestamp}.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        # Stream handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        # Add handlers to logger
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
        """Prepare data for training and create inference template."""
        # Load and preprocess data
        data = pd.read_csv(input_file)
        required_columns = ['ProductId', 'BranchId', 'Currency', 'EffectiveDate', 'Demand']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Input data is missing required column: {col}")

        # Use only the required columns
        data = data[required_columns]

        # Ensure correct data types
        data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate']).dt.tz_localize(None)
        data['ProductId'] = data['ProductId'].astype(str)
        data['BranchId'] = data['BranchId'].astype(str)
        data['Currency'] = data['Currency'].astype(str)
        data.sort_values('EffectiveDate', inplace=True)

        # Create directories for output
        self.output_dir = f"./cs_data/cash/{country_code}_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Define file paths
        train_file = os.path.join(self.output_dir, f"{country_code}_train.csv")
        inference_template_file = os.path.join(self.output_dir, f"{country_code}_inference_template.csv")

        # Save training data
        data.to_csv(train_file, index=False)
        self.train_file = train_file  # Save train file path for later use

        # Create inference template with unique combinations
        inference_template = data.drop_duplicates(
            subset=['ProductId', 'BranchId', 'Currency']
        )[['ProductId', 'BranchId', 'Currency']]

        # Save inference template
        inference_template.to_csv(inference_template_file, index=False)

        return train_file, inference_template_file

    def train_model(self, country_code: str, train_data_s3_uri: str) -> str:
        """Train the forecasting model."""
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
                    'ForecastQuantiles': self.config.quantiles,
                    'TimeSeriesConfig': {
                        'TargetAttributeName': 'Demand',
                        'TimestampAttributeName': 'EffectiveDate',
                        'ItemIdentifierAttributeName': 'ProductId',
                        'GroupingAttributeNames': ['BranchId', 'Currency']
                    }
                }
            },
            'RoleArn': self.role
        }

        # Start training
        try:
            self.sm_client.create_auto_ml_job_v2(**automl_config)
        except Exception as e:
            self.logger.error(f"Failed to start AutoML job: {e}")
            raise
        return job_name

    def _monitor_job(self, job_name: str) -> str:
        """Monitor the AutoML job until completion."""
        self.logger.info(f"Monitoring job {job_name}")
        sleep_time = 60  # Start with 1 minute
        while True:
            try:
                response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
                status = response['AutoMLJobStatus']
                self.logger.info(f"Job {job_name} status: {status}")
                if status in ['Completed', 'Failed', 'Stopped']:
                    if status != 'Completed':
                        failure_reason = response.get('FailureReason', 'No failure reason provided.')
                        self.logger.error(f"AutoML job {job_name} failed: {failure_reason}")
                    break
                sleep(sleep_time)
                sleep_time = min(sleep_time * 1.5, 600)  # Increase sleep time up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring AutoML job: {e}")
                sleep(60)  # Wait before retrying
        return status

    def _get_best_model(self, job_name: str, country_code: str) -> str:
        """Retrieve the best model from the AutoML job."""
        self.logger.info(f"Retrieving best model for job {job_name}")
        while True:
            try:
                response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
                if 'BestCandidate' in response:
                    best_candidate = response['BestCandidate']
                    break
                else:
                    self.logger.info(f"Waiting for best candidate model to be available for job {job_name}")
                    sleep(60)
            except Exception as e:
                self.logger.error(f"Error retrieving best model: {e}")
                sleep(60)

        model_name = f"{country_code}-model-{self.timestamp}"

        # Check if model already exists
        try:
            existing_models = self.sm_client.list_models(NameContains=model_name)
            if existing_models['Models']:
                raise ValueError(f"A model with name {model_name} already exists. Aborting to prevent overwriting.")
        except Exception as e:
            self.logger.error(f"Error checking for existing models: {e}")
            raise

        # Create model
        try:
            self.sm_client.create_model(
                ModelName=model_name,
                ExecutionRoleArn=self.role,
                Containers=best_candidate['InferenceContainers']
            )
            self.logger.info(f"Created model {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to create model {model_name}: {e}")
            raise
        return model_name

    def forecast(self, country_code: str, model_name: str, template_file: str,
                 backtesting: bool = False) -> pd.DataFrame:
        """Generate forecasts starting from specified dates."""
        # Load inference template
        inference_template = pd.read_csv(template_file)

        # Ensure correct data types
        inference_template['ProductId'] = inference_template['ProductId'].astype(str)
        inference_template['BranchId'] = inference_template['BranchId'].astype(str)
        inference_template['Currency'] = inference_template['Currency'].astype(str)

        # Load training data to get EffectiveDates
        training_data = pd.read_csv(self.train_file)
        training_data['EffectiveDate'] = pd.to_datetime(training_data['EffectiveDate']).dt.tz_localize(None)

        if backtesting:
            # Use past EffectiveDates for backtesting
            effective_dates = sorted(training_data['EffectiveDate'].unique())
        else:
            # Use only the most recent date
            effective_dates = [training_data['EffectiveDate'].max()]

        # Define batch size
        batch_size = self.config.batch_size

        # Prepare inference data in batches
        batch_number = 0
        for i in range(0, len(effective_dates), batch_size):
            batch_number += 1
            batch_dates = effective_dates[i:i + batch_size]
            inference_data_list = []

            for effective_date in batch_dates:
                # Generate future dates for the forecast horizon
                future_dates = [effective_date + pd.Timedelta(days=j) for j in
                                range(1, self.config.forecast_horizon + 1)]

                # Number of combinations
                num_combinations = len(inference_template)

                # Create inference data for each combination and future dates
                temp_df = pd.DataFrame({
                    'ProductId': np.tile(inference_template['ProductId'].values, self.config.forecast_horizon),
                    'BranchId': np.tile(inference_template['BranchId'].values, self.config.forecast_horizon),
                    'Currency': np.tile(inference_template['Currency'].values, self.config.forecast_horizon),
                    'EffectiveDate': effective_date,
                    'ForecastDate': np.repeat(future_dates, num_combinations),
                    'Demand': np.nan  # Demand is unknown for future dates
                })
                inference_data_list.append(temp_df)

            # Combine batch inference data
            inference_data = pd.concat(inference_data_list, ignore_index=True)

            # Save to CSV using self.output_dir
            inference_file = os.path.join(self.output_dir, f"{country_code}_inference_batch_{batch_number}.csv")
            inference_data.to_csv(inference_file, index=False)

            # Upload to S3
            s3_inference_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference/{os.path.basename(inference_file)}"
            self._safe_s3_upload(inference_file, s3_inference_key, overwrite=True)
            s3_inference_data_uri = f"s3://{self.config.bucket}/{s3_inference_key}"

            # Run batch transform
            self._run_batch_transform_job(country_code, model_name, s3_inference_data_uri, batch_number)

        # After processing all batches, retrieve and combine forecast results
        forecast_df = self._get_forecast_result(country_code)

        # Save forecasts
        self._save_forecasts(country_code, forecast_df)

        return forecast_df

    def _run_batch_transform_job(self, country_code: str, model_name: str, s3_inference_data_uri: str,
                                 batch_number: int) -> None:
        """Run a batch transform job for the given inference data."""
        transform_job_name = f"{model_name}-transform-{self.timestamp}-{batch_number}"

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
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.config.bucket, Prefix=output_s3_prefix)
        except Exception as e:
            self.logger.error(f"Failed to list objects in S3: {e}")
            raise

        if 'Contents' not in response:
            raise FileNotFoundError(f"No inference results found in S3 for {output_s3_prefix}")

        # Create a temporary directory
        temp_dir = f"./temp/{country_code}_{self.timestamp}/"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Download and process forecast files
            forecast_data = []
            for obj in response['Contents']:
                s3_key = obj['Key']
                if s3_key.endswith('.out'):
                    local_file = os.path.join(temp_dir, os.path.basename(s3_key))
                    try:
                        self.s3_client.download_file(self.config.bucket, s3_key, local_file)
                        df = pd.read_csv(local_file, header=None)
                        forecast_data.append(df)
                    except Exception as e:
                        self.logger.error(f"Failed to process {s3_key}: {e}")
                        continue
                    finally:
                        if os.path.exists(local_file):
                            os.remove(local_file)

            if not forecast_data:
                raise FileNotFoundError("No forecast output files found.")

            # Combine forecast data
            forecast_df = pd.concat(forecast_data, ignore_index=True)

            # Assign quantile columns
            forecast_df.columns = self.config.quantiles

            # Load and combine inference data
            inference_files = glob.glob(os.path.join(self.output_dir, f"{country_code}_inference_batch_*.csv"))
            if not inference_files:
                raise FileNotFoundError(f"No inference batch files found in {self.output_dir}")

            inference_data_list = []
            for file in inference_files:
                try:
                    df = pd.read_csv(file)
                    inference_data_list.append(df)
                except Exception as e:
                    self.logger.error(f"Failed to read inference file {file}: {e}")
                    continue
                finally:
                    if os.path.exists(file):
                        os.remove(file)

            if not inference_data_list:
                raise ValueError("No valid inference data found")

            inference_data = pd.concat(inference_data_list, ignore_index=True)

            # Convert dates
            for date_col in ['EffectiveDate', 'ForecastDate']:
                if date_col in inference_data.columns:
                    inference_data[date_col] = pd.to_datetime(inference_data[date_col]).dt.tz_localize(None)

            # Combine inference data with forecasts
            forecast_df = pd.concat([
                inference_data[['ProductId', 'BranchId', 'Currency',
                                'EffectiveDate', 'ForecastDate']].reset_index(drop=True),
                forecast_df.reset_index(drop=True)
            ], axis=1)

            return forecast_df

        except Exception as e:
            self.logger.error(f"Error in _get_forecast_result: {str(e)}")
            raise
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp directory {temp_dir}: {e}")

    def _save_forecasts(self, country_code: str, forecasts_df: pd.DataFrame) -> None:
        """Save the forecasts with appropriate format."""
        # Required columns
        required_columns = [
            'ProductId', 'BranchId', 'Currency',
            'EffectiveDate', 'ForecastDate'] + self.config.quantiles

        self.logger.info(f"Forecast df columns: {forecasts_df.columns.tolist()}")

        for col in required_columns:
            if col not in forecasts_df.columns:
                raise ValueError(f"Column '{col}' is missing from forecasts DataFrame.")

        # Convert date columns to datetime
        forecasts_df['EffectiveDate'] = pd.to_datetime(forecasts_df['EffectiveDate'])
        forecasts_df['ForecastDate'] = pd.to_datetime(forecasts_df['ForecastDate'])

        # Calculate 'ForecastDay'
        forecasts_df['ForecastDay'] = (forecasts_df['ForecastDate'] - forecasts_df['EffectiveDate']).dt.days + 1

        # Filter forecasts within horizon
        forecasts_df = forecasts_df[
            (forecasts_df['ForecastDay'] >= 1) & (forecasts_df['ForecastDay'] <= self.config.forecast_horizon)]

        # Pivot the data
        forecasts_pivot = forecasts_df.pivot_table(
            index=['ProductId', 'BranchId', 'Currency', 'EffectiveDate'],
            columns='ForecastDay',
            values=self.config.quantiles
        )

        # Flatten the MultiIndex columns
        forecasts_pivot.columns = [f"{quantile}_Day{int(day)}"
                                   for quantile, day in forecasts_pivot.columns]

        # Reset index
        forecasts_pivot.reset_index(inplace=True)

        # Save results
        output_file = f"./results/{country_code}_{self.timestamp}/final_forecast.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        forecasts_pivot.to_csv(output_file, index=False)
        self.logger.info(f"Final forecast saved to {output_file}")

    def run_pipeline(self, country_code: str, input_file: str, backtesting: bool = False) -> None:
        """Run the complete forecasting pipeline."""
        try:
            # Prepare data
            train_file, template_file = self.prepare_data(input_file, country_code)

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

            # Forecasting
            self.forecast(country_code, model_name, template_file, backtesting=backtesting)

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

    # Load and validate configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    try:
        # First validate with Pydantic
        config_model = ConfigModel(**config_dict)
        # Then create Config dataclass
        config = Config(**config_model.dict())
    except Exception as e:
        logging.error(f"Configuration error: {str(e)}")
        sys.exit(1)

    for country_code in args.countries:
        pipeline = CashForecastingPipeline(config)
        try:
            input_file = f"./data/cash/{country_code}.csv"
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            pipeline.run_pipeline(country_code, input_file, backtesting=False)

        except Exception as e:
            logging.error(f"Failed to process {country_code}: {str(e)}")


if __name__ == "__main__":
    main()
