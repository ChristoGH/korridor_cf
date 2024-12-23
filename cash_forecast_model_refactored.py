# cash_forecast_model_refactored.py

"""
Cash Forecasting Model Building Script for Univariate Time Series

This script builds and trains forecasting models for cash demand using the provided configuration.
It is tailored for datasets containing only 'EffectiveDate' and 'Demand' columns.
It leverages shared utilities from common.py for configuration management, data processing,
logging, and AWS interactions.

Usage:
    python cs_scripts/cash_forecast_model_refactored.py --config cs_scripts/config.yaml --countries ZM --input_file input/filtered_18_11_USD.csv
"""

import shutil
import sys
from pathlib import Path
from time import gmtime, strftime, sleep
import json
import logging

import pandas as pd
import numpy as np
import boto3
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
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=self.config.region)
        self.s3_handler = S3Handler(region=self.config.region, logger=self.logger)
        self.data_processor = DataProcessor(logger=self.logger)
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = self.config.role_arn

        # Initialize attributes to be set later
        self.output_dir: Optional[Path] = None
        self.scaling_params: Optional[Dict[str, Any]] = None
        self.scaling_metadata: Optional[Dict[str, Any]] = None

    def run_pipeline(self, country_code: str, input_file: str, backtesting: bool = False,
                     required_columns: Optional[Set[str]] = None) -> None:
        """
        Execute the complete model training pipeline.

        Args:
            country_code (str): The country code.
            input_file (str): Path to the input CSV file.
            backtesting (bool): Flag to indicate backtesting mode.
            required_columns (Optional[Set[str]]): Set of required columns for data validation.

        Raises:
            Exception: If any step in the pipeline fails.
        """
        try:
            self.logger.info(f"Running pipeline for country: {country_code}")

            # Define and create the output directory
            self.output_dir = Path(f"./output/{country_code}/{self.timestamp}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory set to: {self.output_dir}")

            # Create subdirectories after self.output_dir is set
            self.scaling_dir = self.output_dir / 'scaling'
            self.scaling_dir.mkdir(parents=True, exist_ok=True)

            self.training_data_dir = self.output_dir / 'training_data'
            self.training_data_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data
            train_file, inference_template_file = self.prepare_data(input_file, country_code,
                                                                    required_columns=required_columns)

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
            self.logger.error(f"Pipeline failed for {country_code}: {str(e)}", exc_info=True)
            raise

    # cash_forecast_model_refactored.py

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str]:
        """
        Prepare data for training with enhanced sorting and validation.

        This includes:
        - Loading data
        - Selecting required columns
        - Converting and validating timestamps
        - Sorting data
        - Handling duplicates and gaps
        - Handling missing values
        - Splitting data into training and testing sets
        - Scaling the data
        - Recording the training schema for later inference-time validation

        Args:
            input_file (str): Path to the input CSV file.
            country_code (str): The country code.

        Returns:
            Tuple[str, str]: Paths to the training and inference template CSV files.

        Raises:
            ValueError: If required columns are missing, data types are incorrect, scaling parameters are missing,
                        or data integrity conditions are not met.
        """
        self.logger.info(f"Preparing data for country: {country_code}")

        # Load data using DataProcessor
        data = self.data_processor.load_data(input_file)
        self.logger.info(f"Loaded data with shape {data.shape}")

        # Select only 'EffectiveDate' and 'Demand'
        required_columns = {'EffectiveDate', 'Demand'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            self.logger.error(f"Input data is missing required columns: {missing_columns}")
            raise ValueError(f"Input data is missing required columns: {missing_columns}")
        data = data[list(required_columns)]
        self.logger.info(f"Selected required columns: {required_columns}")

        # --------------------------------------
        # Step 1: Record the Training Schema
        # Capture the exact set of columns and their data types
        training_schema = {
            'columns': data.columns.tolist(),
            'column_types': data.dtypes.apply(lambda dt: dt.name).to_dict()
        }
        training_schema_file = self.output_dir / "training_schema.json"
        with open(training_schema_file, "w") as f:
            json.dump(training_schema, f, indent=2)
        self.logger.info(f"Training schema saved at {training_schema_file}")
        # --------------------------------------

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

        # 3. Sort data by EffectiveDate
        try:
            data = data.sort_values(by=['EffectiveDate'], ascending=True, na_position='first')
            self.logger.info("Data sorted by 'EffectiveDate' successfully")
        except Exception as e:
            self.logger.error(f"Sorting failed: {e}")
            raise

        # 4. Verify sort integrity
        if not data['EffectiveDate'].is_monotonic_increasing:
            self.logger.error("Timestamps are not in monotonic increasing order after sorting.")
            raise ValueError("Time series integrity violated: Timestamps are not sorted correctly.")

        # 5. Handle duplicates in EffectiveDate
        duplicates = data.duplicated(subset=['EffectiveDate'], keep=False)
        if duplicates.any():
            self.logger.warning("Duplicate timestamps found in the dataset")
            self.logger.warning(f"Duplicate records:\n{data[duplicates]}")
            # Decide to drop duplicates or handle them
            # Here, we'll drop duplicates keeping the first occurrence
            data = data.drop_duplicates(subset=['EffectiveDate'], keep='first')
            self.logger.info("Duplicates dropped, keeping the first occurrence.")

        # 6. Handle gaps in time series
        if timestamp_freq is not None:
            expected_dates = pd.date_range(
                start=data['EffectiveDate'].min(),
                end=data['EffectiveDate'].max(),
                freq=timestamp_freq
            )
            if len(expected_dates) != len(data):
                missing_dates = expected_dates.difference(data['EffectiveDate'])
                self.logger.warning(f"Found gaps in time series. Missing dates: {missing_dates}")
                # Fill missing dates with NaN and then handle missing values
                data = data.set_index('EffectiveDate').reindex(expected_dates).rename_axis(
                    'EffectiveDate').reset_index()
                self.logger.info("Missing dates filled with NaN.")
        else:
            self.logger.warning("Timestamp frequency is unknown; skipping gap detection and filling.")

        self.logger.info(
            f"Data preparation completed. Processed {len(data)} records."
        )

        # Handle missing values by forward and backward filling
        data = data.ffill().bfill()
        self.logger.info("Handled missing values with forward fill and backfill strategies")

        # Split the data into training and testing sets
        train_df, test_df = self._split_data(data)

        # Scale the training data
        scaled_train_df, scaling_params = self.data_processor.prepare_data(train_df, country_code,
                                                                           required_columns=required_columns)

        # Assign scaling parameters for later use
        self.scaling_params = scaling_params

        # Generate scaling metadata
        self.scaling_metadata = self.data_processor.generate_scaling_metadata(train_df, scaling_params)

        # Save the training and testing data
        train_file = self.training_data_dir / f"{country_code}_train.csv"
        test_file = self.training_data_dir / f"{country_code}_test.csv"

        scaled_train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        self.logger.info(f"Training data saved to {train_file}")
        self.logger.info(f"Test data saved to {test_file}")

        # Save scaling parameters and metadata locally
        scaling_params_file = self.scaling_dir / f"{country_code}_scaling_params.json"
        scaling_metadata_file = self.scaling_dir / f"{country_code}_scaling_metadata.json"

        self.data_processor.save_scaling_params(scaling_params, scaling_params_file)
        self.data_processor.save_metadata(self.scaling_metadata, scaling_metadata_file)

        self.logger.info(
            f"Scaling parameters and metadata saved locally at {scaling_params_file} and {scaling_metadata_file}"
        )

        # Generate inference template
        # The inference template will be strictly based on the final training data schema recorded above.
        inference_template_file = self.generate_inference_template(data, country_code)
        self.logger.info(f"Inference template saved to {inference_template_file}")

        return str(train_file), str(inference_template_file)

    def generate_inference_template(self, data: pd.DataFrame, country_code: str) -> str:
        """
        Generate an inference template based on the training data.

        This template:
        - Uses the final training data, which already includes all required columns.
        - Selects the last row based on EffectiveDate,
          ensuring we have a stable set of columns that match the training schema exactly.
        - No placeholders or hardcoded defaults are introduced. If columns are missing, we fail.

        Args:
            data (pd.DataFrame): The prepared training data containing all required columns.
            country_code (str): The country code.

        Returns:
            str: Path to the inference template CSV file.

        Raises:
            ValueError: If essential columns or 'EffectiveDate' are missing, or if required columns cannot be reproduced.
        """
        self.logger.info("Generating inference template")

        # Load the training schema
        training_schema_file = self.output_dir / "training_schema.json"
        with open(training_schema_file, "r") as f:
            schema = json.load(f)

        required_columns = schema['columns']
        column_types = schema.get('column_types', {})

        self.logger.info(f"Training data columns (required schema): {required_columns}")

        # Check essential columns
        essential_cols = ['EffectiveDate', 'Demand']
        missing_essential = [col for col in essential_cols if col not in data.columns]
        if missing_essential:
            self.logger.error(f"Training data missing essential columns: {missing_essential}")
            raise ValueError(f"Cannot create inference template without {missing_essential} columns.")

        # Sort data by EffectiveDate to get the last record
        data_sorted = data.sort_values(by=['EffectiveDate'], ascending=True)
        last_row = data_sorted.iloc[-1:].copy()

        # Ensure that all required columns are present
        missing_required = [col for col in required_columns if col not in last_row.columns]
        if missing_required:
            self.logger.error(f"Missing required columns in derived template: {missing_required}")
            raise ValueError(f"Cannot produce a complete inference template. Missing columns: {missing_required}")

        # Reorder columns to match the training data schema exactly
        inference_template = last_row[required_columns]

        # Save the inference template
        inference_template_file = self.output_dir / f"{country_code}_inference_template.csv"
        inference_template.to_csv(inference_template_file, index=False)
        self.logger.info(f"Inference template saved to {inference_template_file}")

        return str(inference_template_file)

    def upload_training_data(self, train_file: str, country_code: str) -> str:
        """
        Upload training data to S3 and return the S3 URI.

        Args:
            train_file (str): Path to the training CSV file.
            country_code (str): The country code.

        Returns:
            str: S3 URI of the uploaded training data.
        """
        s3_train_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/train/{Path(train_file).name}"
        self.s3_handler.safe_upload(local_path=train_file, bucket=self.config.bucket, s3_key=s3_train_key)
        train_data_s3_uri = f"s3://{self.config.bucket}/{s3_train_key}"
        self.logger.info(f"Training data uploaded to {train_data_s3_uri}")
        return train_data_s3_uri

    def train_model(self, country_code: str, train_data_s3_uri: str) -> str:
        """
        Train the forecasting model using SageMaker AutoML.

        Args:
            country_code (str): The country code.
            train_data_s3_uri (str): S3 URI of the training data.

        Returns:
            str: Name of the AutoML training job.

        Raises:
            Exception: If the AutoML job creation fails.
        """
        self.logger.info(f"Starting model training for {country_code}")
        job_name = f"{country_code}-ts-{self.timestamp}"

        # AutoML configuration for univariate time series
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
                        'TimestampAttributeName': 'EffectiveDate'
                        # Removed 'ItemIdentifierAttributeName' and 'GroupingAttributeNames'
                    }
                }
            },
            'RoleArn': self.role,
            'Tags': [
                {'Key': 'Project', 'Value': 'CashForecasting'}
            ],
            'AutoMLJobObjective': {
                'MetricName': 'RMSE'  # Example metric; adjust as needed
            },
            'ModelDeployConfig': {
                'EndpointName': f"{country_code}-endpoint-{self.timestamp}"
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

    def _monitor_job(self, job_name: str) -> str:
        """
        Monitor the AutoML job until completion.

        Args:
            job_name (str): The name of the AutoML job.

        Returns:
            str: Final status of the AutoML job.

        Raises:
            RuntimeError: If the job fails or is stopped.
        """
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
                        raise RuntimeError(
                            f"AutoML job {job_name} failed with status: {status}. Reason: {failure_reason}"
                        )
                    self.logger.info(f"AutoML job {job_name} completed successfully.")
                    break

                sleep(sleep_time)
                sleep_time = min(int(sleep_time * 1.5), 600)  # Exponential backoff up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring AutoML job {job_name}: {e}")
                sleep(60)  # Wait before retrying

        return status

    def _get_best_model(self, job_name: str, country_code: str) -> str:
        """
        Retrieve and create the best model from the AutoML job.

        Args:
            job_name (str): The name of the AutoML job.
            country_code (str): The country code.

        Returns:
            str: Name of the created SageMaker model.

        Raises:
            Exception: If retrieving or creating the model fails.
        """
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

        # Check if model already exists to prevent overwriting
        try:
            existing_models = self.sm_client.list_models(NameContains=model_name)
            if existing_models['Models']:
                raise ValueError(f"A model with name {model_name} already exists. Aborting to prevent overwriting.")
        except Exception as e:
            self.logger.error(f"Error checking for existing models: {e}")
            raise

        # Create SageMaker model
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
        scaling_params_file = self.scaling_dir / f"{country_code}_scaling_params.json"
        scaling_metadata_file = self.scaling_dir / f"{country_code}_scaling_metadata.json"

        # Save scaling parameters and metadata locally
        self.data_processor.save_scaling_params(self.scaling_params, scaling_params_file)
        self.data_processor.save_metadata(self.scaling_metadata, scaling_metadata_file)

        self.logger.info(
            f"Scaling parameters and metadata saved locally at {scaling_params_file} and {scaling_metadata_file}"
        )

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

    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the sorted data into training and test sets.

        Splitting Logic:
            - Training: All data except the last 8 records
            - Testing: Last 8 records

        Args:
            data (pd.DataFrame): The prepared and sorted data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.
        """
        self.logger.info("Splitting data into training and testing sets")

        total_records = len(data)
        if total_records <= 8:
            self.logger.warning("Dataset has 8 or fewer records; skipping splitting.")
            return pd.DataFrame(), data.copy()

        train_df = data.iloc[:-8].copy()
        test_df = data.iloc[-8:].copy()

        self.logger.info(f"Training set shape: {train_df.shape}")
        self.logger.info(f"Testing set shape: {test_df.shape}")

        return train_df, test_df


def main():
    """Main entry point for the cash forecasting model training script."""
    # Parse command line arguments
    args = parse_arguments()  # inference=False by default

    # Setup logging
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
    logger = setup_logging(timestamp, name='CashForecastModel')
    logger.info("Starting Cash Forecasting Model Training Pipeline for Univariate Time Series")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("Configuration loaded successfully.")

        # Initialize and run the pipeline
        pipeline = CashForecastingPipeline(config=config, logger=logger)

        # Process each country
        for country_code in args.countries:
            try:
                logger.info(f"\nProcessing country: {country_code}")

                # Determine input file path
                input_file = args.input_file or f"./data/cash/{country_code}.csv"

                # Retrieve required columns for the current project
                # Assume 'project_type' is passed as an argument or inferred
                project_type = 'univariate'  # Modify as needed
                required_columns = set(config['required_columns'][project_type])

                # Run the pipeline
                pipeline.run_pipeline(
                    country_code=country_code,
                    input_file=input_file,
                    backtesting=args.backtesting,  # Assuming 'backtesting' flag is present
                    required_columns=required_columns
                )

            except Exception as e:
                logger.error(f"Failed to process country {country_code}: {e}", exc_info=True)
                logger.error("Continuing with next country.\n")
                continue  # Proceed with the next country even if one fails

        logger.info("Model training pipeline completed successfully for all specified countries.")

    except Exception as e:
        logger.error(f"Critical error in model training pipeline: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Clean up any temporary files or resources if necessary
        logger.info("Cleaning up resources...")
        try:
            # Add any cleanup code here if necessary
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()
