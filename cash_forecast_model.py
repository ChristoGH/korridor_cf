# cash_forecast_model.py

import argparse
import boto3
import pandas as pd
import numpy as np
import os
import sys
import json
import shutil
from time import gmtime, strftime, sleep
from typing import Dict, Tuple, List
from dataclasses import dataclass, field

from sagemaker import Session
import yaml
import glob

from common import Config, load_and_validate_config, setup_logging, safe_s3_upload, load_scaling_parameters


class CashForecastingPipeline:
    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize the forecasting pipeline with configuration."""
        self.config = config
        self.logger = logger
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=config.region)
        self.s3_client = boto3.client('s3')
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = config.role_arn
        self.train_file = None
        self.output_dir = None  # Will be set in prepare_data

    def _safe_s3_upload(self, local_path: str, s3_key: str, overwrite: bool = False) -> None:
        """Safely upload file to S3 with existence check."""
        safe_s3_upload(
            s3_client=self.s3_client,
            logger=self.logger,
            local_path=local_path,
            s3_key=s3_key,
            bucket=self.config.bucket,
            overwrite=overwrite
        )

    def _load_scaling_parameters_from_s3(self, country_code: str) -> Dict[Tuple[str, str], Dict]:
        """Load scaling parameters from S3."""
        scaling_params_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/scaling/{country_code}_scaling_params.json"
        return load_scaling_parameters(
            s3_client=self.s3_client,
            logger=self.logger,
            bucket=self.config.bucket,
            scaling_params_key=scaling_params_key
        )

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str]:
        """Prepare data for training and create inference template."""
        # Load and preprocess data
        data = pd.read_csv(input_file)
        required_columns = ['ProductId', 'BranchId', 'Currency', 'EffectiveDate', 'Demand']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Input data is missing required columns: {missing_columns}")

        # Check for missing values in required columns
        if data[required_columns].isnull().any().any():
            missing = data[required_columns].isnull().sum()
            raise ValueError(f"Input data contains missing values: {missing[missing > 0]}")

        # Use only the required columns
        data = data[required_columns]

        # Ensure correct data types
        data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate'], errors='coerce').dt.tz_localize(None)
        if data['EffectiveDate'].isnull().any():
            raise ValueError("Some 'EffectiveDate' entries could not be parsed as datetime.")

        data['ProductId'] = data['ProductId'].astype(str)
        data['BranchId'] = data['BranchId'].astype(str)
        data['Currency'] = data['Currency'].astype(str)
        data.sort_values('EffectiveDate', inplace=True)

        # Calculate scaling parameters per Currency-BranchId combination
        scaling_params = {}
        for (currency, branch), group in data.groupby(['Currency', 'BranchId']):
            scaling_params[(currency, branch)] = {
                'mean': group['Demand'].mean(),
                'std': group['Demand'].std(),
                'min': group['Demand'].min(),
                'max': group['Demand'].max(),
                'last_value': group.iloc[-1]['Demand'],
                'n_observations': len(group)
            }

        # Apply scaling to Demand values
        scaled_data = data.copy()
        for (currency, branch), params in scaling_params.items():
            mask = (data['Currency'] == currency) & (data['BranchId'] == branch)
            # Standardize the data
            scaled_data.loc[mask, 'Demand'] = (
                (data.loc[mask, 'Demand'] - params['mean']) /
                (params['std'] if params['std'] != 0 else 1)
            )

        # Create directories for output
        self.output_dir = f"./cs_data/cash/{country_code}_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Save scaling parameters
        scaling_file = os.path.join(self.output_dir, f"{country_code}_scaling_params.json")
        with open(scaling_file, 'w') as f:
            # Convert scaling_params keys to strings for JSON serialization
            serializable_params = {
                f"{currency}_{branch}": params
                for (currency, branch), params in scaling_params.items()
            }
            json.dump(serializable_params, f, indent=2)

        # Save metadata about the scaling
        metadata = {
            'scaling_method': 'standardization',
            'scaling_level': 'currency_branch',
            'scaled_column': 'Demand',
            'timestamp': self.timestamp,
            'scaling_stats': {
                'global_mean': data['Demand'].mean(),
                'global_std': data['Demand'].std(),
                'global_min': data['Demand'].min(),
                'global_max': data['Demand'].max(),
                'n_groups': len(scaling_params),
                'n_total_observations': len(data)
            }
        }

        metadata_file = os.path.join(self.output_dir, f"{country_code}_scaling_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Define file paths
        train_file = os.path.join(self.output_dir, f"{country_code}_train.csv")
        inference_template_file = os.path.join(self.output_dir, f"{country_code}_inference_template.csv")

        # Save training data
        scaled_data.to_csv(train_file, index=False)
        self.train_file = train_file  # Save train file path for later use

        # Create inference template with unique combinations
        inference_template = data.drop_duplicates(
            subset=['ProductId', 'BranchId', 'Currency']
        )[['ProductId', 'BranchId', 'Currency']]

        # Save inference template
        inference_template.to_csv(inference_template_file, index=False)

        # Log scaling information
        self.logger.info(f"Applied standardization scaling at currency-branch level")
        self.logger.info(f"Scaling parameters saved to {scaling_file}")
        self.logger.info(f"Scaling metadata saved to {metadata_file}")
        self.logger.info(f"Global scale: mean={metadata['scaling_stats']['global_mean']:.2f}, "
                         f"std={metadata['scaling_stats']['global_std']:.2f}")

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
            self.logger.info(f"AutoML job {job_name} initiated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to start AutoML job: {e}")
            raise
        return job_name

    def _monitor_job(self, job_name: str, max_wait_time: int = 24 * 60 * 60) -> str:
        """Monitor the AutoML job until completion with a timeout."""
        self.logger.info(f"Monitoring job {job_name}")
        sleep_time = 60  # Start with 1 minute
        elapsed_time = 0
        while elapsed_time < max_wait_time:
            try:
                response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
                status = response['AutoMLJobStatus']
                self.logger.info(f"Job {job_name} status: {status}")
                if status in ['Completed', 'Failed', 'Stopped']:
                    if status != 'Completed':
                        failure_reason = response.get('FailureReason', 'No failure reason provided.')
                        self.logger.error(f"AutoML job {job_name} failed: {failure_reason}")
                    return status
                sleep(sleep_time)
                elapsed_time += sleep_time
                sleep_time = min(int(sleep_time * 1.5), 600)  # Increase sleep time up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring AutoML job: {e}")
                sleep_time = min(int(sleep_time * 1.5), 600)
                elapsed_time += sleep_time
                sleep(60)  # Wait before retrying
        raise TimeoutError(f"AutoML job {job_name} did not complete within {max_wait_time} seconds.")

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

    # The rest of the methods (forecast, _run_batch_transform_job, _monitor_transform_job,
    # _get_forecast_result, _save_forecasts) remain unchanged from your original script.
    # Due to space constraints, they are not shown here but should follow similar refactoring
    # principles by utilizing shared utilities where applicable.

def main():
    parser = argparse.ArgumentParser(description='Cash Forecasting Pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--countries', nargs='+', default=['ZM'],
                        help='Country codes to process')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to the input CSV file (overrides default)')
    args = parser.parse_args()

    # Load and validate configuration
    config = load_and_validate_config(args.config)

    # Setup logging
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
    logger = setup_logging('CashForecast', timestamp)

    for country_code in args.countries:
        pipeline = CashForecastingPipeline(config, logger)
        try:
            if args.input_file:
                input_file = args.input_file
            else:
                input_file = f"./data/cash/{country_code}.csv"
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            pipeline.run_pipeline(country_code, input_file, backtesting=False)

        except Exception as e:
            logger.error(f"Failed to process {country_code}: {str(e)}")


if __name__ == "__main__":
    main()
