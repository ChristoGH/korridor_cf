# cash_forecast_model.py

"""
# cash_forecast_model.py
# Usage:
# python cs_scripts/cash_forecast_model.py --config cs_scripts/config.yaml --countries ZM

Cash Forecasting Model Building Script

This script builds and trains forecasting models for cash demand using the provided configuration.
It leverages shared utilities from common.py for configuration management, data processing,
logging, and AWS interactions.

Approaches to Ensure Long-Term Scalability and Accuracy:

1. Remove or Filter Out Unseen Combinations:
   This script generates an inference template from the training dataset.
   Because of this, all (Currency, BranchId) pairs in the inference template
   are guaranteed to have scaling parameters. No new combinations should appear
   at inference time unless the inference script uses a different template. If
   unexpected combinations appear during inference, you would either remove
   those combinations or ensure the inference template only includes known combinations.

2. Retrain or Update the Model:
   If the business requires forecasting for new (Currency, BranchId) pairs not
   present in the training data, you must retrain the model with historical data
   that includes these new combinations. This ensures scaling parameters and
   model parameters cover these new entities.

3. Handle Missing Combinations Gracefully:
   While this training script does not directly handle inference for unseen combinations,
   you could implement logic during inference to skip or assign default scaling parameters
   for unseen combinations, at the cost of accuracy. The code below focuses on ensuring
   that the training pipeline produces all required artifacts for known combinations.

Created model ZM-model-20241206-031017 successfully.
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
from typing import Tuple

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
        self.train_file = None
        self.output_dir = None  # Will be set in run_pipeline

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str]:
        """
        Prepare data for training following AWS blog best practices.

        Implements time-aware splitting strategy:
        - Sorts data chronologically within each group
        - Creates training set excluding last 8 timestamps
        - Creates test set from timestamps n-8 to n-4
        - Reserves last 4 timestamps for validation
        """
        self.logger.info(f"Preparing data for country: {country_code}")

        # Load data using DataProcessor
        data = self.data_processor.load_data(input_file)

        # Convert timestamp to datetime
        data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate'])

        # Sort data properly for time series
        data = data.sort_values(['ProductId', 'BranchId', 'Currency', 'EffectiveDate'])

        # Initialize containers for split data
        train_dfs = []
        test_dfs = []

        # Split data by group as per blog recommendations
        for group_key, group_data in data.groupby(['ProductId', 'BranchId', 'Currency']):
            if len(group_data) <= 8:
                self.logger.warning(f"Group {group_key} has insufficient data points (<= 8), skipping")
                continue

            # Get chronologically sorted timestamps
            timestamps = group_data['EffectiveDate'].unique()

            # Identify split points
            train_end = len(timestamps) - 8
            test_start = len(timestamps) - 8
            test_end = len(timestamps) - 4

            # Split the data
            train_mask = group_data['EffectiveDate'] < timestamps[train_end]
            test_mask = (group_data['EffectiveDate'] >= timestamps[test_start]) & \
                        (group_data['EffectiveDate'] < timestamps[test_end])

            train_dfs.append(group_data[train_mask])
            test_dfs.append(group_data[test_mask])

        # Combine split datasets
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Scale data if needed
        scaled_data, scaling_params = self.data_processor.prepare_data(train_df, country_code)

        # Save the data
        train_file = self.output_dir / f"{country_code}_train.csv"
        test_file = self.output_dir / f"{country_code}_test.csv"

        scaled_data.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        return str(train_file), str(test_file)

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

        # Enhanced AutoML configuration based on AWS blog best practices
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
                        'GroupingAttributeNames': ['BranchId', 'Currency'],
                        'FillConfig': {  # Added filling configuration as per blog
                            'ImputationStrategy': 'FORWARD_FILL',  # Example strategy
                            'BackfillStrategy': 'ZERO'
                        }
                    }
                }
            },
            'RoleArn': self.role,
            'ModelMetrics': {  # Added model metrics configuration
                'ModelQuality': {
                    'Statistics': {
                        'S3Uri': f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/metrics/"
                    }
                }
            },
            'GenerateCandidateDefinitionsOnly': False,  # Ensures full model training
            'RetryStrategy': {  # Added retry strategy for robustness
                'MaxAttempts': 3
            }
        }
        # Start training
        try:
            self.sm_client.create_auto_ml_job_v2(**automl_config)
            self.logger.info(f"AutoML job {job_name} created successfully.")
        except Exception as e:
            self.logger.error(f"Failed to start AutoML job: {e}")
            raise

        return job_name

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str]:
        """
        Prepare data for training with enhanced sorting as per AWS blog best practices.

        The sorting process follows these steps:
        1. Convert timestamp to proper datetime format
        2. Validate timestamp consistency
        3. Multi-level sort by identifiers and timestamp
        4. Verify sort integrity
        5. Handle any gaps in time series
        """
        self.logger.info(f"Preparing data for country: {country_code}")

        # Load data using DataProcessor
        data = self.data_processor.load_data(input_file)

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
        for group in data.groupby(['ProductId', 'BranchId', 'Currency']):
            group_data = group[1]  # group[0] is the key, group[1] is the data

            # Check if timestamps are strictly increasing within group
            if not group_data['EffectiveDate'].is_monotonic_increasing:
                self.logger.error(f"Non-monotonic timestamps found in group {group[0]}")
                raise ValueError(f"Time series integrity violated in group {group[0]}")

            # Check for duplicates
            duplicates = group_data.duplicated(subset=['EffectiveDate'], keep=False)
            if duplicates.any():
                self.logger.warning(f"Duplicate timestamps found in group {group[0]}")
                # Log the duplicates for investigation
                self.logger.warning(f"Duplicate records:\n{group_data[duplicates]}")

        # 5. Handle gaps in time series
        groups_with_gaps = []
        for group in data.groupby(['ProductId', 'BranchId', 'Currency']):
            group_data = group[1]
            expected_dates = pd.date_range(
                start=group_data['EffectiveDate'].min(),
                end=group_data['EffectiveDate'].max(),
                freq=timestamp_freq
            )
            if len(expected_dates) != len(group_data):
                groups_with_gaps.append(group[0])

        if groups_with_gaps:
            self.logger.warning(f"Found gaps in time series for groups: {groups_with_gaps}")

        self.logger.info(f"Data preparation completed. Processed {len(data)} records across "
                         f"{len(data.groupby(['ProductId', 'BranchId', 'Currency']))} groups")

        # Proceed with splitting and scaling as before...
        train_df, test_df = self._split_data(data)
        scaled_data, scaling_params = self.data_processor.prepare_data(train_df, country_code)

        # Save the data
        train_file = self.output_dir / f"{country_code}_train.csv"
        test_file = self.output_dir / f"{country_code}_test.csv"

        scaled_data.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        return str(train_file), str(test_file)

    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the sorted data into training and test sets following the blog's methodology.
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
            test_end = len(timestamps) - 4

            train_mask = group_data['EffectiveDate'] < timestamps[train_end]
            test_mask = (group_data['EffectiveDate'] >= timestamps[test_start]) & \
                        (group_data['EffectiveDate'] < timestamps[test_end])

            train_dfs.append(group_data[train_mask])
            test_dfs.append(group_data[test_mask])

        return pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)

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

    def _run_batch_transform_job(self, country_code: str, model_name: str, s3_inference_data_uri: str,
                                 batch_number: int) -> None:
        """Run a batch transform job for the given inference data."""
        transform_job_name = f"{model_name}-transform-{self.timestamp}-{batch_number}"
        output_s3_uri = f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/"

        # Start batch transform job using SageMaker
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
            self.logger.info(f"Created transform job {transform_job_name} successfully.")
            self._monitor_transform_job(transform_job_name)
        except Exception as e:
            self.logger.error(f"Failed to create transform job {transform_job_name}: {e}")
            raise

    def _monitor_transform_job(self, transform_job_name: str) -> None:
        """Monitor the batch transform job until completion."""
        self.logger.info(f"Monitoring transform job: {transform_job_name}")
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
                            f"Transform job {transform_job_name} failed with status: {status}. Reason: {failure_reason}"
                        )
                    break

                sleep(sleep_time)
                sleep_time = min(sleep_time * 1.5, 600)  # Increase sleep time up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring transform job {transform_job_name}: {e}")
                sleep(60)  # Wait before retrying

    def run_pipeline(self, country_code: str, input_file: str, backtesting: bool = False) -> None:
        """
        Run the model building pipeline.

        This method handles data preparation, uploads training data to S3,
        initiates model training via SageMaker AutoML, monitors the training job,
        and retrieves the best model.

        Args:
            country_code (str): The country code for which to build the model.
            input_file (str): Path to the input CSV file.
            backtesting (bool): Flag to indicate if backtesting is to be performed.

        Raises:
            RuntimeError: If the training job fails to complete successfully.
        """

        try:
            self.logger.info(f"Running pipeline for country: {country_code}")

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

            self.logger.info(f"Model building pipeline completed successfully for {country_code}")

        except Exception as e:
            self.logger.error(f"Pipeline failed for {country_code}: {str(e)}")
            raise

def main():
    # Parse command line arguments without inference-specific arguments
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