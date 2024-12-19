# cash_forecast_model.py

"""
Cash Forecasting Model Building Script

This script builds and trains forecasting models for cash demand using the provided configuration.
It leverages shared utilities from common.py for configuration management, data processing,
logging, and AWS interactions.

Usage:
    python cs_scripts/cash_forecast_model.py --config cs_scripts/config.yaml --countries ZM
"""

from time import sleep
import argparse
import sys
from pathlib import Path
from time import gmtime, strftime
import logging
import boto3
import pandas as pd
import numpy as np
from sagemaker import Session
from typing import Tuple, Optional, Dict, Any

from common import (
    Config,
    S3Handler,
    DataProcessor,
    setup_logging,
    load_config,
    parse_arguments
)


class CashForecastingPipeline:
    """Pipeline for training and forecasting cash demand models."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize the forecasting pipeline with configuration."""
        self.config = config
        self.logger = logger
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=self.config.region)
        self.s3_handler = S3Handler(region=self.config.region, logger=self.logger)
        self.data_processor = DataProcessor(logger=self.logger)
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = self.config.role_arn
        self.train_file: Optional[str] = None
        self.output_dir: Optional[Path] = None  # Will be set in run_pipeline
        self.scaling_params: Optional[Dict[str, Dict[str, float]]] = None
        self.scaling_metadata: Optional[Dict[str, Any]] = None
        # After determining self.output_dir = Path(f"./output/{country_code}/{self.timestamp}")
        self.scaling_dir = self.output_dir / 'scaling'
        self.scaling_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_dir = self.output_dir / 'training_data'
        self.training_data_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self, country_code: str, input_file: str, backtesting: bool = False) -> None:
        """
        Run the model building pipeline.

        This method handles data preparation, uploads training data to S3,
        initiates model training via SageMaker AutoML, monitors the training job,
        retrieves the best model, and saves scaling parameters.

        Args:
            country_code (str): The country code for which to build the model.
            input_file (str): Path to the input CSV file.
            backtesting (bool): Flag to indicate if backtesting is to be performed.

        Raises:
            RuntimeError: If the training job fails to complete successfully.
        """
        try:
            self.logger.info(f"Running pipeline for country: {country_code}")

            # Define and create the output directory
            self.output_dir = Path(f"./output/{country_code}/{self.timestamp}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory set to: {self.output_dir}")

            # Load and prepare data
            train_file, inference_template_file = self.prepare_data(input_file, country_code)

            # Upload training data to S3
            train_data_s3_uri = self.upload_training_data(train_file, country_code)

            # Train model
            job_name = self.train_model(country_code, train_data_s3_uri)

            # Monitor training
            status = self._monitor_job(job_name)
            if status != 'Completed':
                raise RuntimeError(f"Training failed with status: {status}")

            # Get best model
            model_name = self._get_best_model(job_name, country_code)

            # Save scaling parameters and metadata
            self.save_scaling_parameters(country_code)

            self.logger.info(f"Model building pipeline completed successfully for {country_code}")

        except Exception as e:
            self.logger.error(f"Pipeline failed for {country_code}: {str(e)}")
            raise

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str]:
        """
        Prepare data for training with enhanced sorting and validation as per AWS blog best practices.

        This includes:
        - Loading data
        - Converting and validating timestamps
        - Sorting data
        - Handling duplicates and gaps
        - Handling missing values
        - Splitting data into training and testing sets
        - Scaling the data

        Args:
            input_file (str): Path to the input CSV file.
            country_code (str): The country code.

        Returns:
            Tuple[str, str]: Paths to the training and inference template CSV files.
        """
        self.logger.info(f"Preparing data for country: {country_code}")

        # Load data using DataProcessor
        data = self.data_processor.load_data(input_file)
        self.logger.info(f"Loaded data with shape {data.shape}")

        # 1. Convert timestamp to datetime with validation
        try:
            data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate'])
            self.logger.info("Timestamp conversion successful")
        except Exception as e:
            self.logger.error(f"Failed to convert timestamps: {e}")
            raise ValueError("Timestamp conversion failed. Please ensure timestamp format is consistent.")

        # 2. Validate timestamp consistency
        timestamp_freq = pd.infer_freq(data['EffectiveDate'].sort_values())
        if timestamp_freq is None:
            self.logger.warning("Could not infer consistent timestamp frequency. Check for irregular intervals.")
        else:
            self.logger.info(f"Inferred timestamp frequency: {timestamp_freq}")

        # 3. Multi-level sort implementation
        try:
            data = data.sort_values(
                by=['ProductId', 'BranchId', 'Currency', 'EffectiveDate'],
                ascending=[True, True, True, True],
                na_position='first'  # Handle any NAs consistently
            )
            self.logger.info("Multi-level sort completed successfully")
        except Exception as e:
            self.logger.error(f"Sorting failed: {e}")
            raise

        # 4. Verify sort integrity
        for group_key, group_data in data.groupby(['ProductId', 'BranchId', 'Currency']):
            # Check if timestamps are strictly increasing within group
            if not group_data['EffectiveDate'].is_monotonic_increasing:
                self.logger.error(f"Non-monotonic timestamps found in group {group_key}")
                raise ValueError(f"Time series integrity violated in group {group_key}")

            # Check for duplicates
            duplicates = group_data.duplicated(subset=['EffectiveDate'], keep=False)
            if duplicates.any():
                self.logger.warning(f"Duplicate timestamps found in group {group_key}")
                # Log the duplicates for investigation
                self.logger.warning(f"Duplicate records:\n{group_data[duplicates]}")

        # 5. Handle gaps in time series
        groups_with_gaps = []
        for group_key, group_data in data.groupby(['ProductId', 'BranchId', 'Currency']):
            expected_dates = pd.date_range(
                start=group_data['EffectiveDate'].min(),
                end=group_data['EffectiveDate'].max(),
                freq=timestamp_freq
            )
            if len(expected_dates) != len(group_data):
                groups_with_gaps.append(group_key)

        if groups_with_gaps:
            self.logger.warning(f"Found gaps in time series for groups: {groups_with_gaps}")
            # Optionally, implement gap filling or further handling here

        self.logger.info(f"Data preparation completed. Processed {len(data)} records across "
                         f"{len(data.groupby(['ProductId', 'BranchId', 'Currency']))} groups")

        # Handle missing values during data preparation
        data = data.ffill().bfill()  # Updated to use ffill and bfill directly to avoid FutureWarning
        self.logger.info("Handled missing values with forward fill and backfill strategies")

        # Proceed with splitting and scaling as per original methodology
        train_df, test_df = self._split_data(data)
        scaled_data, scaling_params = self.data_processor.prepare_data(train_df, country_code)

        # Assign scaling parameters to the pipeline for later use
        self.scaling_params = scaling_params

        # Generate scaling metadata
        self.scaling_metadata = self.data_processor.generate_scaling_metadata(train_df, scaling_params)

        # Save the data

        # Save train and test data in training_data directory
        train_file = self.training_data_dir / f"{country_code}_train.csv"
        test_file = self.training_data_dir / f"{country_code}_test.csv"

        scaled_data.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        self.logger.info(f"Training data saved to {train_file}")
        self.logger.info(f"Test data saved to {test_file}")

        scaled_data.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        self.logger.info(f"Training data saved to {train_file}")
        self.logger.info(f"Test data saved to {test_file}")

        # Save scaling parameters and metadata locally

        scaling_params_file = self.scaling_dir / f"{country_code}_scaling_params.json"
        scaling_metadata_file = self.scaling_dir / f"{country_code}_scaling_metadata.json"

        self.data_processor.save_scaling_params(scaling_params, scaling_params_file)
        self.data_processor.save_metadata(self.scaling_metadata, scaling_metadata_file)

        self.logger.info(
            f"Scaling parameters and metadata saved locally at {scaling_params_file} and {scaling_metadata_file}")

        self.data_processor.save_scaling_params(scaling_params, scaling_params_file)
        self.data_processor.save_metadata(self.scaling_metadata, scaling_metadata_file)

        # Generate inference template
        inference_template_file = self.generate_inference_template(data, country_code)
        self.logger.info(f"Inference template saved to {inference_template_file}")

        return str(train_file), str(inference_template_file)

    def generate_inference_template(self, data: pd.DataFrame, country_code: str) -> str:
        """
        Generate an inference template based on the training data.

        This template is used to ensure that inference data aligns with training data's structure.

        Args:
            data (pd.DataFrame): The prepared training data.
            country_code (str): The country code.

        Returns:
            str: Path to the inference template CSV file.
        """
        self.logger.info("Generating inference template")

        # Assuming the inference template should include unique combinations of ProductId, BranchId, Currency
        inference_template = data[['ProductId', 'BranchId', 'Currency']].drop_duplicates().copy()
        # Add 'ForecastDate' for forecasting horizon (e.g., next date)
        # This can be adjusted based on forecasting requirements
        last_dates = data.groupby(['ProductId', 'BranchId', 'Currency'])['EffectiveDate'].max().reset_index()
        last_dates['ForecastDate'] = last_dates['EffectiveDate'] + pd.to_timedelta(1, unit='D')  # Example: next day
        inference_template = inference_template.merge(last_dates[['ProductId', 'BranchId', 'Currency', 'ForecastDate']],
                                                      on=['ProductId', 'BranchId', 'Currency'],
                                                      how='left')

        # Save the inference template
        inference_template_file = self.output_dir / f"{country_code}_inference_template.csv"
        inference_template.to_csv(inference_template_file, index=False)
        self.logger.info(f"Inference template saved to {inference_template_file}")

        return str(inference_template_file)

    def upload_training_data(self, train_file: str, country_code: str) -> str:
        """Upload training data to S3 and return the S3 URI."""
        s3_train_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/train/{Path(train_file).name}"
        self.s3_handler.safe_upload(local_path=train_file, bucket=self.config.bucket, s3_key=s3_train_key)
        train_data_s3_uri = f"s3://{self.config.bucket}/{s3_train_key}"
        self.logger.info(f"Training data uploaded to {train_data_s3_uri}")
        return train_data_s3_uri

    def train_model(self, country_code: str, train_data_s3_uri: str) -> str:
        """Train the forecasting model using SageMaker AutoML."""
        self.logger.info(f"Starting model training for {country_code}")
        job_name = f"{country_code}-ts-{self.timestamp}"

        # AutoML configuration without unsupported parameters
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
                        # Removed 'FillConfig' as it's unsupported
                    }
                }
            },
            'RoleArn': self.role,
            # Optional: Add Tags if needed
            'Tags': [
                {'Key': 'Project', 'Value': 'CashForecasting'}
            ],
            # Optional: Define AutoMLJobObjective if needed
            'AutoMLJobObjective': {
                'MetricName': 'RMSE'  # Example metric
            },
            # Optional: Define ModelDeployConfig if immediate deployment is desired
            'ModelDeployConfig': {
                'EndpointName': f"{country_code}-endpoint-{self.timestamp}"
                # Removed 'InitialInstanceCount' and 'InstanceType' as they are unsupported
                # You can use 'AutoGenerateEndpointName': True if you prefer auto-generated names
                # 'AutoGenerateEndpointName': True
            },
            # Optional: Define DataSplitConfig if custom data splitting is needed
            # 'DataSplitConfig': {
            #     'AutoSplit': True
            # },
        }

        # Start training
        try:
            self.sm_client.create_auto_ml_job_v2(**automl_config)
            self.logger.info(f"AutoML job {job_name} created successfully.")
        except Exception as e:
            self.logger.error(f"Failed to start AutoML job: {e}")
            raise

        return job_name

    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the sorted data into training and test sets following the blog's methodology.

        Splitting Logic:
            - For each (ProductId, BranchId, Currency) group:
                - Training: All data except the last 8 records
                - Testing: Last 8 records

        Args:
            data (pd.DataFrame): The prepared and sorted data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.
        """
        train_dfs = []
        test_dfs = []

        for group_key, group_data in data.groupby(['ProductId', 'BranchId', 'Currency']):
            timestamps = group_data['EffectiveDate'].unique()

            if len(timestamps) <= 8:
                self.logger.warning(f"Group {group_key} has insufficient data points (<= 8), skipping")
                continue

            # Split as per blog's specification
            train_end = len(timestamps) - 8
            test_start = len(timestamps) - 8
            test_end = len(timestamps) - 4  # Adjust based on specific testing requirements

            train_mask = group_data['EffectiveDate'] < timestamps[train_end]
            test_mask = (group_data['EffectiveDate'] >= timestamps[test_start]) & \
                        (group_data['EffectiveDate'] < timestamps[test_end])

            train_dfs.append(group_data[train_mask])
            test_dfs.append(group_data[test_mask])

        # Combine all training and testing DataFrames
        if train_dfs:
            train_combined = pd.concat(train_dfs, ignore_index=True)
        else:
            train_combined = pd.DataFrame()  # Empty DataFrame if no training data

        if test_dfs:
            test_combined = pd.concat(test_dfs, ignore_index=True)
        else:
            test_combined = pd.DataFrame()  # Empty DataFrame if no testing data

        self.logger.info(f"Training set shape: {train_combined.shape}")
        self.logger.info(f"Testing set shape: {test_combined.shape}")

        return train_combined, test_combined

    def _monitor_job(self, job_name: str) -> str:
        """Monitor the AutoML job until completion."""
        self.logger.info(f"Monitoring AutoML job: {job_name}")
        sleep_time = 60  # Start with 60 seconds

        while True:
            try:
                response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
                status = response['AutoMLJobStatus']
                self.logger.info(f"AutoML job {job_name} status: {status}")

                if status in ['Completed', 'Failed', 'Stopped']:
                    if status != 'Completed':
                        failure_reason = response.get('FailureReason', 'No failure reason provided.')
                        self.logger.error(f"AutoML job {job_name} failed: {failure_reason}")
                        raise RuntimeError(f"AutoML job {job_name} failed with status: {status}. Reason: {failure_reason}")
                    break

                sleep(sleep_time)
                sleep_time = min(sleep_time * 1.5, 600)  # Exponential backoff up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring AutoML job {job_name}: {e}")
                sleep(60)  # Wait before retrying

        return status

    def _get_best_model(self, job_name: str, country_code: str) -> str:
        """Retrieve and create the best model from the AutoML job."""
        self.logger.info(f"Retrieving best model for job: {job_name}")

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
            self.logger.info(f"Created model {model_name} successfully.")
        except Exception as e:
            self.logger.error(f"Failed to create model {model_name}: {e}")
            raise

        return model_name

    def save_scaling_parameters(self, country_code: str) -> None:
        """
        Save scaling parameters and metadata to JSON files and upload to S3.

        Args:
            country_code (str): The country code.

        Raises:
            Exception: If saving or uploading fails.
        """
        self.logger.info(f"Saving scaling parameters for country: {country_code}")

        if self.scaling_params is None or self.scaling_metadata is None:
            self.logger.error("Scaling parameters or metadata are not available to save.")
            raise ValueError("Scaling parameters or metadata are missing.")

        # Define local paths for JSON files
        scaling_params_file = self.output_dir / f"{country_code}_scaling_params.json"
        scaling_metadata_file = self.output_dir / f"{country_code}_scaling_metadata.json"

        # Save scaling parameters and metadata locally
        self.data_processor.save_scaling_params(self.scaling_params, scaling_params_file)
        self.data_processor.save_metadata(self.scaling_metadata, scaling_metadata_file)

        # Upload scaling parameters and metadata to S3
        scaling_params_s3_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/{country_code}_scaling_params.json"
        scaling_metadata_s3_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/{country_code}_scaling_metadata.json"

        self.s3_handler.safe_upload(
            local_path=str(scaling_params_file),
            bucket=self.config.bucket,
            s3_key=scaling_params_s3_key,
            overwrite=True
        )
        self.logger.info(f"Scaling parameters uploaded to s3://{self.config.bucket}/{scaling_params_s3_key}")

        self.s3_handler.safe_upload(
            local_path=str(scaling_metadata_file),
            bucket=self.config.bucket,
            s3_key=scaling_metadata_s3_key,
            overwrite=True
        )
        self.logger.info(f"Scaling metadata uploaded to s3://{self.config.bucket}/{scaling_metadata_s3_key}")


def main():
    """Main entry point for the cash forecasting model training script."""
    # Parse command line arguments
    args = parse_arguments()  # inference=False by default

    # Setup logging
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
    logger = setup_logging(timestamp, name='CashForecastModel')
    logger.info("Starting Cash Forecasting Model Training Pipeline")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("Configuration loaded successfully.")

        # Initialize and run the pipeline
        pipeline = CashForecastingPipeline(config=config, logger=logger)
        pipeline.run_pipeline(
            country_code=args.countries[0],
            input_file=args.input_file or f"./data/cash/{args.countries[0]}.csv",
            backtesting=args.resume
        )

        logger.info("Model training pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
