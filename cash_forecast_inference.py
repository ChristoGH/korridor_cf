"""
Cash Forecasting Inference Script

This script performs inference using a trained forecasting model on new data.
It leverages shared utilities from common.py for configuration management,
data processing, logging, and AWS interactions.

Usage:


ZM-model-20241219-064209
20241219-064209
python cs_scripts/cash_forecast_inference.py \
    --config cs_scripts/config.yaml \
    --countries ZM \
    --model_timestamp 20241219-064209 \
    --effective_date 2024-07-01 \
    --model_name ZM-model-20241219-064209 \
    --inference_template ./output/ZM/20241219-064209/ZM_inference_template.csv
"""

import shutil
import sys
from pathlib import Path
from time import gmtime, strftime, sleep
from datetime import datetime
import logging
from tempfile import TemporaryDirectory
import pandas as pd
import numpy as np
import boto3
from sagemaker import Session
import time
from botocore.exceptions import ClientError
import uuid

from common import (
    ConfigModel,
    Config,
    S3Handler,
    DataProcessor,
    setup_logging,
    load_config,
    parse_arguments
)
from typing import Dict, List, Tuple, Any, Optional
import json


class CashForecastingPipeline:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=self.config.region)
        self.s3_handler = S3Handler(region=self.config.region, logger=self.logger)
        self.data_processor = DataProcessor(logger=self.logger)
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())  # Retained for potential other uses
        self.role = self.config.role_arn

        self.output_dir = None
        self.model_timestamp = None  # Initialize as None

    def setup_directories(self, country_code: str, model_timestamp: str) -> None:
        self.output_dir = Path(f"./output/{country_code}/{model_timestamp}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scaling_dir = self.output_dir / 'scaling'
        self.scaling_dir.mkdir(parents=True, exist_ok=True)

        self.inference_dir = self.output_dir / 'inference'
        self.inference_dir.mkdir(parents=True, exist_ok=True)

        self.transform_logs_dir = self.inference_dir / 'transform_job_logs'
        self.transform_logs_dir.mkdir(parents=True, exist_ok=True)

        self.transform_outputs_dir = self.inference_dir / 'transform_outputs'
        self.transform_outputs_dir.mkdir(parents=True, exist_ok=True)

    def load_scaling_parameters(self, country_code: str, model_timestamp: str) -> None:
        """
        Load and validate scaling parameters and metadata from S3.

        Args:
            country_code (str): The country code.
            model_timestamp (str): The timestamp of the training run.

        Raises:
            FileNotFoundError: If scaling parameters or metadata are not found.
            ValueError: If scaling parameters or metadata are invalid.
        """
        self.logger.info(f"Loading scaling parameters for country: {country_code}")
        scaling_params_key = f"{self.config.prefix}-{country_code}/{model_timestamp}/{country_code}_scaling_params.json"
        scaling_metadata_key = f"{self.config.prefix}-{country_code}/{model_timestamp}/{country_code}_scaling_metadata.json"

        # Download scaling parameters
        scaling_params_path = Path(f"./temp/{country_code}_scaling_params.json")
        self.s3_handler.download_file(bucket=self.config.bucket, s3_key=scaling_params_key,
                                      local_path=scaling_params_path)
        scaling_params = self.data_processor.load_scaling_params(scaling_file=scaling_params_path)

        # Download scaling metadata
        scaling_metadata_path = Path(f"./temp/{country_code}_scaling_metadata.json")
        self.s3_handler.download_file(bucket=self.config.bucket, s3_key=scaling_metadata_key,
                                      local_path=scaling_metadata_path)
        scaling_metadata = self.data_processor.load_scaling_metadata(scaling_metadata_file=scaling_metadata_path)

        # Save scaling parameters and metadata locally
        local_scaling_params_file = self.scaling_dir / f"{country_code}_scaling_params.json"
        local_scaling_metadata_file = self.scaling_dir / f"{country_code}_scaling_metadata.json"

        shutil.copy(scaling_params_path, local_scaling_params_file)
        shutil.copy(scaling_metadata_path, local_scaling_metadata_file)

        # Remove temporary files
        scaling_params_path.unlink(missing_ok=True)
        scaling_metadata_path.unlink(missing_ok=True)

        self.logger.info(f"Scaling parameters and metadata also saved locally at {self.scaling_dir}")

        # Validate scaling parameters
        self.data_processor.validate_scaling_parameters(scaling_params)

        # Assign to the pipeline
        self.scaling_params = scaling_params
        self.scaling_metadata = scaling_metadata

        self.logger.info(f"Scaling parameters and metadata loaded and validated for {country_code}")

    def prepare_inference_data(self, country_code: str, effective_date: str, inference_template_path: str) -> str:
        """
        Prepare inference data by ensuring it aligns with the training schema and applying scaling.

        Args:
            country_code (str): The country code.
            effective_date (str): The effective date for forecasting.
            inference_template_path (str): Path to the inference template CSV file.

        Returns:
            str: Path to the scaled inference CSV file.

        Raises:
            ValueError: If required columns are missing, data types are incorrect, scaling parameters are missing,
                        or data integrity conditions are not met.
        """
        # Load the training schema
        training_schema_file = self.output_dir / "training_schema.json"
        if not training_schema_file.exists():
            self.logger.error(f"Training schema file not found at {training_schema_file}")
            raise FileNotFoundError(
                "training_schema.json is missing. Cannot validate inference template without schema.")

        with open(training_schema_file, "r") as f:
            schema = json.load(f)

        required_columns = schema['columns']  # Assuming schema has a 'columns' key with column names
        column_types = schema.get('column_types', {})  # Optional: data types per column

        # Load inference data
        inference_df = pd.read_csv(inference_template_path)
        self.logger.info(f"Loaded inference template with shape {inference_df.shape}")

        # Check for missing columns
        missing_cols = set(required_columns) - set(inference_df.columns)
        if missing_cols:
            self.logger.error(f"Missing required columns in inference data: {missing_cols}")
            raise ValueError(f"Inference data is missing columns required by the model: {missing_cols}")

        # Check for unexpected extra columns
        extra_cols = set(inference_df.columns) - set(required_columns)
        if extra_cols:
            self.logger.warning(f"Extra columns present in inference data: {extra_cols}. They will be dropped.")
            inference_df = inference_df[list(required_columns)]

        # Enforce data types if specified
        for col, dtype in column_types.items():
            if col in inference_df.columns:
                try:
                    inference_df[col] = inference_df[col].astype(dtype)
                    self.logger.debug(f"Converted column '{col}' to type '{dtype}'.")
                except Exception as e:
                    self.logger.error(f"Failed to convert column '{col}' to type '{dtype}': {e}")
                    raise ValueError(f"Column '{col}' has incorrect data type.")

        # Check data integrity
        critical_cols_no_nan = ['ProductId', 'BranchId', 'Currency', 'EffectiveDate']
        for col in critical_cols_no_nan:
            if inference_df[col].isnull().any():
                self.logger.error(f"Column {col} contains NaN values, which is not allowed.")
                raise ValueError(f"Integrity check failed: {col} has NaN values at inference.")

        # Create scaling key
        if 'Currency' not in inference_df.columns or 'BranchId' not in inference_df.columns:
            self.logger.error("Inference data must contain 'Currency' and 'BranchId' to create ScalingKey.")
            raise ValueError("Missing 'Currency' or 'BranchId' for scaling operations.")

        inference_df['ScalingKey'] = inference_df['Currency'] + '_' + inference_df['BranchId'].astype(str)

        # Retrieve scaling parameters
        scaling_means = inference_df['ScalingKey'].map(lambda x: self.scaling_params.get(x, {}).get('mean'))
        scaling_stds = inference_df['ScalingKey'].map(lambda x: self.scaling_params.get(x, {}).get('std'))

        # Identify missing scaling parameters
        missing_scaling = inference_df[scaling_means.isnull() | scaling_stds.isnull()]
        if not missing_scaling.empty:
            self.logger.error(
                f"Missing scaling parameters for combinations:\n{missing_scaling[['ProductId', 'BranchId', 'Currency']]}"
            )
            raise ValueError("Missing scaling parameters for some combinations in the inference template.")

        # Apply scaling to 'Demand' if required
        if 'Demand' in required_columns:
            if 'Demand' not in inference_df.columns:
                self.logger.error("Demand column is required by the schema but is missing at inference.")
                raise ValueError("Demand column missing at inference time.")

            # Apply scaling
            valid_stds = scaling_stds.replace({0: np.nan})
            if valid_stds.isnull().any():
                self.logger.error("Scaling std is zero for some combinations, cannot scale 'Demand'.")
                raise ValueError("Scaling standard deviation is zero for some combinations, cannot scale Demand.")

            inference_df['Demand_scaled'] = (inference_df['Demand'] - scaling_means) / valid_stds
            if inference_df['Demand_scaled'].isnull().any():
                self.logger.error(
                    "Some 'Demand_scaled' values are NaN after scaling. Check scaling parameters or input data.")
                raise ValueError("NaN values found in 'Demand_scaled' column after scaling.")
            self.logger.info("'Demand' column found and scaled successfully.")
        else:
            self.logger.info("'Demand' column not required by schema, skipping scaling.")

        # Save the scaled inference data
        scaled_inference_file = self.inference_dir / f"scaled_inference_{country_code}.csv"
        inference_df.to_csv(scaled_inference_file, index=False)
        self.logger.info(f"Scaled inference data saved to {scaled_inference_file}")

        return str(scaled_inference_file)

    def upload_inference_data(self, inference_file: str, country_code: str) -> str:
        """
        Upload the prepared inference data to S3.

        Args:
            inference_file (str): Path to the inference CSV file.
            country_code (str): The country code.

        Returns:
            str: S3 URI of the uploaded inference data.

        Raises:
            Exception: If upload fails.
        """
        self.logger.info(f"Uploading inference data for country: {country_code}")
        try:
            # Use self.model_timestamp instead of self.timestamp for consistent path referencing
            s3_inference_key = f"{self.config.prefix}-{country_code}/{self.model_timestamp}/scaling/inference/{Path(inference_file).name}"
            self.s3_handler.safe_upload(
                local_path=inference_file,
                bucket=self.config.bucket,
                s3_key=s3_inference_key,
                overwrite=True
            )
            s3_inference_uri = f"s3://{self.config.bucket}/{s3_inference_key}"
            self.logger.info(f"Inference data uploaded to {s3_inference_uri}")
            return s3_inference_uri
        except Exception as e:
            self.logger.error(f"Failed to upload inference data to S3: {e}")
            raise

    def run_batch_transform(self, country_code: str, model_name: str, s3_inference_uri: str) -> None:
        """
        Run a SageMaker batch transform job.

        Args:
            country_code (str): The country code.
            model_name (str): The SageMaker model name.
            s3_inference_uri (str): S3 URI of the inference data.

        Raises:
            Exception: If the transform job fails.
        """
        self.logger.info(f"Starting batch transform for country: {country_code}")
        # Use self.model_timestamp for transform job naming
        transform_job_name = f"{country_code}-transform-{self.model_timestamp}"

        # Define output S3 URI using self.model_timestamp
        output_s3_uri = f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.model_timestamp}/inference-output/"

        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                self.sm_client.create_transform_job(
                    TransformJobName=transform_job_name,
                    ModelName=model_name,
                    BatchStrategy='MultiRecord',
                    TransformInput={
                        'DataSource': {
                            'S3DataSource': {
                                'S3DataType': 'S3Prefix',
                                'S3Uri': s3_inference_uri
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
                self.logger.info(f"Transform job {transform_job_name} created successfully.")
                self.monitor_transform_job(transform_job_name)
                break  # Exit loop if successful
            except self.sm_client.exceptions.ResourceInUse as e:
                self.logger.error(f"Attempt {attempt}: Transform job name {transform_job_name} is already in use: {e}")
                if attempt < max_retries:
                    self.logger.info(f"Retrying after {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Generate a new unique suffix for the next attempt
                    unique_suffix = uuid.uuid4().hex[:8]
                    transform_job_name = f"{country_code}-transform-{self.model_timestamp}-{unique_suffix}"
                else:
                    self.logger.error(f"All {max_retries} attempts failed. Unable to create a unique transform job.")
                    raise
            except ClientError as e:
                self.logger.error(f"ClientError on attempt {attempt}: {e}")
                if attempt < max_retries:
                    self.logger.info(f"Retrying after {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"All {max_retries} attempts failed due to ClientError.")
                    raise
            except Exception as e:
                self.logger.error(f"Failed to create transform job {transform_job_name}: {e}")
                raise


    def monitor_transform_job(self, transform_job_name: str) -> None:
        """
        Monitor the SageMaker transform job until completion.

        Args:
            transform_job_name (str): The name of the transform job.

        Raises:
            RuntimeError: If the transform job fails.
        """
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
                        # Download logs before raising exception
                        self.download_transform_job_logs(transform_job_name, transform_job_name.split('-')[0])
                        raise RuntimeError(
                            f"Transform job {transform_job_name} failed with status: {status}. Reason: {failure_reason}"
                        )
                    self.logger.info(f"Transform job {transform_job_name} completed successfully.")
                    break

                sleep(sleep_time)
                sleep_time = min(sleep_time * 1.5, 600)  # Exponential backoff up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring transform job {transform_job_name}: {e}")
                sleep(60)  # Wait before retrying

    def download_transform_output(self, transform_job_name: str, country_code: str) -> Path:
        """
        Download the Batch Transform job output from S3 to a local directory.

        Args:
            transform_job_name (str): The name of the transform job.
            country_code (str): The country code.

        Returns:
            Path: Local path to the downloaded transform output.
        """
        self.logger.info(f"Downloading transform job output for {transform_job_name}")

        # Describe the transform job to get output S3 path
        try:
            response = self.sm_client.describe_transform_job(TransformJobName=transform_job_name)
            output_s3_uri = response['TransformOutput']['S3OutputPath']
            self.logger.info(f"Transform job output S3 URI: {output_s3_uri}")
        except Exception as e:
            self.logger.error(f"Failed to retrieve transform job details: {e}")
            raise

        # List objects under the output S3 URI
        try:
            output_prefix = '/'.join(output_s3_uri.replace("s3://", "").split('/')[1:])
            objects = self.s3_handler.list_s3_objects(bucket=self.config.bucket, prefix=output_prefix)
            output_keys = [obj['Key'] for obj in objects if obj['Key'].endswith('.csv') or obj['Key'].endswith('.out')]
            if not output_keys:
                self.logger.error(f"No output files found in {output_s3_uri}")
                raise FileNotFoundError(f"No output files found in {output_s3_uri}")
        except Exception as e:
            self.logger.error(f"Failed to list transform output objects: {e}")
            raise

        # Create local directory for transform outputs
        local_output_dir = self.transform_outputs_dir / transform_job_name
        local_output_dir.mkdir(parents=True, exist_ok=True)

        # Download the files into local_output_dir
        for key in output_keys:
            local_file = local_output_dir / Path(key).name
            self.s3_handler.download_file(bucket=self.config.bucket, s3_key=key, local_path=local_file)
            self.logger.info(f"Downloaded {key} to {local_file}")

        self.logger.info(f"All transform outputs downloaded to {local_output_dir}")
        return local_output_dir

    def download_transform_job_logs(self, transform_job_name: str, country_code: str) -> Path:
        """
        Download CloudWatch logs for the Batch Transform job.

        Args:
            transform_job_name (str): The name of the transform job.
            country_code (str): The country code.

        Returns:
            Path: Local path to the saved log file.
        """
        self.logger.info(f"Downloading CloudWatch logs for transform job: {transform_job_name}")

        # Initialize CloudWatch Logs client
        logs_client = boto3.client('logs', region_name=self.config.region)

        # Define the log group and log stream names
        # SageMaker Batch Transform log group format: /aws/sagemaker/TransformJobs
        log_group_name = "/aws/sagemaker/TransformJobs"
        log_stream_prefix = transform_job_name

        try:
            # Describe log streams with the given prefix
            response = logs_client.describe_log_streams(
                logGroupName=log_group_name,
                logStreamNamePrefix=log_stream_prefix,
                orderBy='LogStreamName',
                descending=True
            )

            log_streams = response.get('logStreams', [])
            if not log_streams:
                self.logger.error(f"No log streams found for transform job: {transform_job_name}")
                raise FileNotFoundError(f"No log streams found for transform job: {transform_job_name}")

            # Assume the first log stream is the desired one
            log_stream_name = log_streams[0]['logStreamName']
            self.logger.info(f"Found log stream: {log_stream_name}")

            # Get log events
            events_response = logs_client.get_log_events(
                logGroupName=log_group_name,
                logStreamName=log_stream_name,
                startFromHead=True
            )

            log_events = events_response.get('events', [])
            if not log_events:
                self.logger.warning(f"No log events found in log stream: {log_stream_name}")

            # Save logs to a local file
            local_log_dir = self.output_dir / f"{transform_job_name}_logs"
            local_log_dir.mkdir(parents=True, exist_ok=True)
            local_log_file = self.transform_logs_dir / f"{transform_job_name}_CloudWatch.log"
            with open(local_log_file, 'w') as f:
                for event in log_events:
                    f.write(event['message'] + '\n')
            self.logger.info(f"CloudWatch logs saved to {local_log_file}")
            return local_log_file

        except Exception as e:
            self.logger.error(f"Failed to download CloudWatch logs for {transform_job_name}: {e}")
            raise

    def download_forecast_results(self, country_code: str, inference_template_path: str) -> pd.DataFrame:
        """
        Download and combine forecast results from S3, associating them with their identifiers.

        Args:
            country_code (str): The country code.
            inference_template_path (str): Path to the inference template CSV file.

        Returns:
            pd.DataFrame: Combined forecast DataFrame containing ProductId, BranchId, Currency, and quantiles.

        Raises:
            FileNotFoundError: If no forecast results are found.
            Exception: If download or processing fails.
        """
        with TemporaryDirectory(dir="./") as temp_dir_path:
            temp_dir = Path(temp_dir_path)
            try:
                self.logger.info(f"Downloading forecast results for country: {country_code}")
                # Use self.model_timestamp for accurate path references
                forecast_s3_prefix = f"{self.config.prefix}-{country_code}/{self.model_timestamp}/inference-output/"
                forecast_files = self.s3_handler.list_s3_objects(bucket=self.config.bucket, prefix=forecast_s3_prefix)

                # Filter for forecast output files (assuming they have .out extension)
                forecast_keys = [obj['Key'] for obj in forecast_files if obj['Key'].endswith('.out')]

                if not forecast_keys:
                    self.logger.error(
                        f"No forecast output files found in s3://{self.config.bucket}/{forecast_s3_prefix}")
                    raise FileNotFoundError(
                        f"No forecast output files found in s3://{self.config.bucket}/{forecast_s3_prefix}"
                    )

                forecast_dfs = []
                for key in forecast_keys:
                    local_file = temp_dir / Path(key).name
                    self.s3_handler.download_file(bucket=self.config.bucket, s3_key=key, local_path=local_file)
                    self.logger.info(f"Downloaded forecast file {key} to {local_file}")

                    # Read forecast data, assuming no headers and only quantile columns
                    df = pd.read_csv(local_file, header=None, sep=',')  # Adjust 'sep' if necessary

                    # Assign quantile column names
                    quantile_columns = ['p10', 'p50', 'p90']
                    if len(df.columns) >= len(quantile_columns):
                        df_quantiles = df.iloc[:, :len(quantile_columns)].copy()
                        df_quantiles.columns = quantile_columns
                        self.logger.debug(f"Assigned quantile columns: {df_quantiles.columns.tolist()}")
                    else:
                        self.logger.error(
                            f"Unexpected number of columns ({len(df.columns)}) in forecast file {key}. Expected at least {len(quantile_columns)} quantile columns."
                        )
                        raise ValueError(
                            f"Unexpected number of columns ({len(df.columns)}) in forecast file {key}. Expected at least {len(quantile_columns)} quantile columns."
                        )

                    # Replace 'ERROR' strings with NaN and convert to numeric
                    df_quantiles.replace('ERROR', np.nan, inplace=True)
                    for q in quantile_columns:
                        df_quantiles[q] = pd.to_numeric(df_quantiles[q], errors='coerce')

                    forecast_dfs.append(df_quantiles)
                    self.logger.info(f"Processed forecast file {key} with shape {df_quantiles.shape}")

                # Combine all forecast DataFrames
                combined_forecast_quantiles = pd.concat(forecast_dfs, ignore_index=True)
                self.logger.info(f"Combined forecast quantiles shape: {combined_forecast_quantiles.shape}")

                # Load the inference template to get identifiers using the provided inference_template_path
                inference_template_path = Path(inference_template_path)
                if not inference_template_path.exists():
                    self.logger.error(f"Inference template not found at {inference_template_path}")
                    raise FileNotFoundError(f"Inference template not found at {inference_template_path}")

                inference_template_df = pd.read_csv(inference_template_path)
                self.logger.info(f"Loaded inference template with shape {inference_template_df.shape}")

                # Determine forecast horizon based on config
                forecast_horizon = self.config.forecast_horizon  # Ensure this is defined and an integer
                quantiles = self.config.quantiles  # ['p10', 'p50', 'p90']

                # Calculate expected number of forecast records
                expected_forecast_records = len(inference_template_df) * forecast_horizon
                actual_forecast_records = len(combined_forecast_quantiles)
                self.logger.info(
                    f"Expected forecast records: {expected_forecast_records}, Actual: {actual_forecast_records}"
                )

                if actual_forecast_records != expected_forecast_records:
                    self.logger.warning(
                        f"Number of forecast records ({actual_forecast_records}) does not match expected ({expected_forecast_records}). Proceeding with minimum records."
                    )
                    min_records = min(actual_forecast_records, expected_forecast_records)
                    combined_forecast_quantiles = combined_forecast_quantiles.iloc[:min_records].copy()
                    inference_template_df = inference_template_df.iloc[:min_records // forecast_horizon].copy()

                # Repeat the inference identifiers based on forecast horizon
                inference_repeated = inference_template_df.loc[
                    inference_template_df.index.repeat(forecast_horizon)
                ].reset_index(drop=True)
                self.logger.info(
                    f"Repeated inference template to match forecast records with shape {inference_repeated.shape}"
                )

                # Ensure the lengths match
                if len(inference_repeated) != len(combined_forecast_quantiles):
                    self.logger.error(
                        f"After adjustments, forecast records ({len(combined_forecast_quantiles)}) do not match repeated inference records ({len(inference_repeated)})."
                    )
                    raise ValueError(
                        f"Forecast records ({len(combined_forecast_quantiles)}) do not match repeated inference records ({len(inference_repeated)})."
                    )

                # Combine identifiers with forecast quantiles
                combined_forecast = pd.concat(
                    [
                        inference_repeated.reset_index(drop=True),
                        combined_forecast_quantiles.reset_index(drop=True)
                    ],
                    axis=1
                )
                self.logger.info(f"Combined forecast data shape: {combined_forecast.shape}")

                # Optionally, add ForecastDay if not present
                # Assuming ForecastDay ranges from 1 to forecast_horizon
                forecast_days = list(range(1, forecast_horizon + 1)) * len(inference_template_df)
                combined_forecast['ForecastDay'] = forecast_days[:len(combined_forecast)]

                # Final check for required columns
                required_columns = ['ProductId', 'BranchId', 'Currency'] + quantile_columns
                missing_cols = set(required_columns) - set(combined_forecast.columns)
                if missing_cols:
                    self.logger.error(f"Missing required columns after merging: {missing_cols}")
                    raise KeyError(f"Missing required columns: {missing_cols}")

                return combined_forecast

            except Exception as e:
                self.logger.error(f"Failed to download or process forecast results: {e}")
                raise

    def restore_forecasts_scale(self, forecast_df: pd.DataFrame, country_code: str) -> pd.DataFrame:
        """
        Restore the original scale of the forecasted Demand values.
        """
        self.logger.info("Restoring original scale to forecasted Demand values.")
        try:
            # Corrected file path to load the scaled inference data
            inference_csv = self.inference_dir / f"scaled_inference_{country_code}.csv"

            if not inference_csv.exists():
                self.logger.error(f"Inference CSV file does not exist: {inference_csv}")
                raise FileNotFoundError(f"Inference CSV file does not exist: {inference_csv}")

            inference_data = self.data_processor.load_data(str(inference_csv))
            self.logger.info(f"inference_csv: {inference_csv}")
            self.logger.info(f"Columns in inference_data: {list(inference_data.columns)}")

            # Check for required columns
            required_columns = ['ProductId', 'BranchId', 'Currency']
            missing_cols = set(required_columns) - set(inference_data.columns)
            if missing_cols:
                self.logger.error(f"Inference CSV is missing required columns: {missing_cols}")
                raise ValueError(f"Inference CSV is missing required columns: {missing_cols}")

            # Check for 'Demand_scaled' column
            if 'Demand_scaled' not in inference_data.columns:
                self.logger.error("'Demand_scaled' column is missing from the inference data.")
                raise KeyError("'Demand_scaled' column is missing from the inference data.")

            # Determine if forecast_df already contains identifier columns
            identifier_columns = ['ProductId', 'BranchId', 'Currency']
            if not all(col in forecast_df.columns for col in identifier_columns):
                # If not, attach them
                forecast_df = inference_data[identifier_columns].reset_index(drop=True).join(forecast_df)
            else:
                self.logger.info("forecast_df already contains identifier columns. Skipping join to avoid duplication.")

            # Create scaling_key column
            forecast_df['scaling_key'] = forecast_df.apply(lambda row: f"{row['Currency']}_{row['BranchId']}", axis=1)

            # Map scaling parameters to each row
            forecast_df['mean'] = forecast_df['scaling_key'].map(
                lambda x: self.scaling_params[x]['mean'] if x in self.scaling_params else np.nan)
            forecast_df['std'] = forecast_df['scaling_key'].map(
                lambda x: self.scaling_params[x]['std'] if x in self.scaling_params else 1)

            # Apply inverse scaling
            for quantile in self.config.quantiles:
                if quantile in forecast_df.columns:
                    forecast_df[quantile] = forecast_df[quantile] * forecast_df['std'] + forecast_df['mean']
                    self.logger.debug(f"Inverse scaled quantile '{quantile}'")
                else:
                    self.logger.warning(f"Quantile '{quantile}' not found in forecast DataFrame.")

            # Drop auxiliary columns
            forecast_df.drop(['scaling_key', 'mean', 'std'], axis=1, inplace=True)

            # Validate that 'p50' exists before renaming
            if 'p50' not in forecast_df.columns:
                self.logger.error("'p50' column is missing after inverse scaling.")
                raise KeyError("'p50' column is missing after inverse scaling.")

            # Rename 'p50' to 'Demand'
            forecast_df = forecast_df.rename(columns={'p50': 'Demand'})
            self.logger.info("Renamed 'p50' to 'Demand' successfully.")

            self.logger.info("Scale restoration completed successfully.")
            return forecast_df

        except Exception as e:
            self.logger.error(f"Failed to restore scale: {e}")
            raise

    def validate_forecast_data(self, forecast_df: pd.DataFrame) -> None:
        """
        Validate the forecast DataFrame for required columns and data types.

        Args:
            forecast_df (pd.DataFrame): DataFrame containing forecasted quantiles.

        Raises:
            ValueError: If validation fails.
        """
        required_columns = ['ProductId', 'BranchId', 'Currency', 'p10', 'Demand', 'p90']
        missing_columns = set(required_columns) - set(forecast_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in forecast data: {missing_columns}")

        # Additional type checks can be added here
        if not pd.api.types.is_numeric_dtype(forecast_df['p10']):
            raise TypeError("Quantile 'p10' must be numeric.")
        if not pd.api.types.is_numeric_dtype(forecast_df['Demand']):
            raise TypeError("Quantile 'Demand' must be numeric.")
        if not pd.api.types.is_numeric_dtype(forecast_df['p90']):
            raise TypeError("Quantile 'p90' must be numeric.")

    def save_final_forecasts(self, forecast_df: pd.DataFrame, country_code: str) -> None:
        """
        Save the final restored forecast DataFrame to CSV.

        Args:
            forecast_df (pd.DataFrame): DataFrame with restored forecasts.
            country_code (str): The country code.

        Raises:
            Exception: If saving fails.
        """
        try:
            final_forecast_path = self.inference_dir / f"{country_code}_final_forecast_{self.model_timestamp}.csv"
            forecast_df.to_csv(final_forecast_path, index=False)
            self.logger.info(f"Final forecasts saved to {final_forecast_path}")
        except Exception as e:
            self.logger.error(f"Failed to save final forecasts: {e}")
            raise

    def generate_statistical_report(self, forecast_df: pd.DataFrame, country_code: str) -> None:
        """
        Generate a statistical report for the forecasts.

        Args:
            forecast_df (pd.DataFrame): DataFrame containing the final forecasts.
            country_code (str): The country code.

        Raises:
            Exception: If report generation fails.
        """
        try:
            report = {
                'country_code': country_code,
                'timestamp': self.model_timestamp,
                'forecast_horizon': self.config.forecast_horizon,
                'quantiles': self.config.quantiles,
                'total_forecasts': len(forecast_df),
                'statistics': {}
            }

            for quantile in self.config.quantiles:
                if quantile in forecast_df.columns:
                    stats = forecast_df[quantile].describe().to_dict()
                    report['statistics'][quantile] = stats
                    self.logger.info(f"Statistics for {quantile}: {stats}")
                else:
                    self.logger.warning(f"Quantile '{quantile}' not found in forecast data.")

            # Save report to JSON
            report_path = self.inference_dir / f"{country_code}_forecast_report_{self.model_timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Statistical report saved to {report_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate statistical report: {e}")
            raise

    def map_forecasts_to_dates(self, restored_forecast_df: pd.DataFrame, country_code: str) -> pd.DataFrame:
        """
        Map forecasted quantiles to future dates based on the last EffectiveDate.

        Args:
            restored_forecast_df (pd.DataFrame): DataFrame with restored forecasts.
            country_code (str): The country code.

        Returns:
            pd.DataFrame: Forecast DataFrame with mapped dates.
        """
        self.logger.info("Mapping forecasts to future dates.")

        # Load the scaled inference data to get the last EffectiveDate per item
        inference_csv = self.inference_dir / f"scaled_inference_{country_code}.csv"
        inference_data = self.data_processor.load_data(str(inference_csv))

        # Ensure 'EffectiveDate' is in datetime format
        if 'EffectiveDate' not in inference_data.columns:
            self.logger.error("'EffectiveDate' column is missing from the inference data.")
            raise KeyError("'EffectiveDate' column is missing from the inference data.")

        inference_data['EffectiveDate'] = pd.to_datetime(inference_data['EffectiveDate'])

        # Get the last EffectiveDate for each (ProductId, BranchId, Currency) group
        last_dates = inference_data.groupby(['ProductId', 'BranchId', 'Currency'], as_index=False)[
            'EffectiveDate'].max()

        self.logger.info(f"Last EffectiveDate per item calculated.")

        # Merge the last EffectiveDate with the forecast DataFrame
        # Assuming that restored_forecast_df has forecasts ordered per item and per forecast horizon
        # Determine the number of forecasts per item based on forecast_horizon
        forecast_horizon = self.config.forecast_horizon  # Ensure this is defined and an integer
        quantiles = self.config.quantiles  # ['p10', 'p50', 'p90']

        num_items = len(last_dates)
        expected_forecasts = num_items * forecast_horizon

        if len(restored_forecast_df) != expected_forecasts:
            self.logger.error(
                f"Forecast DataFrame size mismatch: expected {expected_forecasts} rows "
                f"({num_items} items * {forecast_horizon} forecast horizon), got {len(restored_forecast_df)} rows."
            )
            raise ValueError(
                f"Forecast DataFrame size mismatch: expected {expected_forecasts} rows "
                f"({num_items} items * {forecast_horizon} forecast horizon), got {len(restored_forecast_df)} rows."
            )

        # Create a list of future dates per item
        future_dates = []
        for _, row in last_dates.iterrows():
            item_future_dates = [row['EffectiveDate'] + pd.Timedelta(days=i) for i in range(1, forecast_horizon + 1)]
            future_dates.extend(item_future_dates)

        # Assign the future dates to the forecast DataFrame
        restored_forecast_df['ForecastDate'] = future_dates[:len(restored_forecast_df)]

        self.logger.info("Mapped forecasts to future dates successfully.")
        return restored_forecast_df

    def run_inference(
            self,
            country_code: str,
            model_name: str,
            effective_date: str,
            inference_template_path: str,
            model_timestamp: str
    ) -> None:
        """
        Execute the complete inference pipeline.

        Args:
            country_code (str): The country code.
            model_name (str): The SageMaker model name.
            effective_date (str): The effective date for forecasting.
            inference_template_path (str): Path to the inference template CSV file.
            model_timestamp (str): The timestamp of the trained model.

        Raises:
            Exception: If any step in the pipeline fails.
        """
        # Store model_timestamp as an instance variable for consistent access
        self.model_timestamp = model_timestamp

        # Setup directories based on country_code and model_timestamp
        self.setup_directories(country_code, model_timestamp)

        # Load scaling parameters and metadata
        self.load_scaling_parameters(country_code, model_timestamp)

        self.logger.info(f"Starting inference pipeline for country: {country_code}")

        try:
            # Prepare inference data
            inference_file = self.prepare_inference_data(
                country_code, effective_date, inference_template_path
            )

            # Upload inference data to S3
            s3_inference_uri = self.upload_inference_data(inference_file, country_code)

            # Run batch transform job
            self.run_batch_transform(country_code, model_name, s3_inference_uri)

            # Define transform job name based on country_code and model_timestamp
            transform_job_name = f"{country_code}-transform-{model_timestamp}"

            # Download transform job output from S3 to local directory
            transform_output_dir = self.download_transform_output(transform_job_name, country_code)

            # Download CloudWatch logs for the transform job
            transform_logs_file = self.download_transform_job_logs(transform_job_name, country_code)

            # Ensure the scaled inference file exists locally
            scaled_inference_file = self.inference_dir / f"scaled_inference_{country_code}.csv"
            if not scaled_inference_file.exists():
                shutil.copy(inference_file, scaled_inference_file)
                self.logger.info(f"Copied scaled inference data to {scaled_inference_file}")

            # Download and process forecast results
            forecast_df = self.download_forecast_results(country_code, inference_template_path)

            # Restore original scale of forecasts
            restored_forecast_df = self.restore_forecasts_scale(forecast_df, country_code)

            # Map forecasts to future dates
            restored_forecast_df = self.map_forecasts_to_dates(restored_forecast_df, country_code)

            # Validate the final forecast data
            self.validate_forecast_data(restored_forecast_df)

            # Save the final forecasts to CSV
            self.save_final_forecasts(restored_forecast_df, country_code)

            # Generate a statistical report for the forecasts
            self.generate_statistical_report(restored_forecast_df, country_code)

            self.logger.info(f"Inference pipeline completed successfully for {country_code}")

        except Exception as e:
            self.logger.error(f"Inference pipeline failed for {country_code}: {e}", exc_info=True)
            raise


def main():
    """Main entry point for the cash forecasting inference script."""
    # Parse command line arguments using common.py's parse_arguments() with inference=True
    args = parse_arguments(inference=True)

    # Setup logging using common.py's setup_logging
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
    logger = setup_logging(timestamp, name='CashForecastInference')
    logger.info("Starting Cash Forecasting Inference Pipeline")

    try:
        # Load and validate configuration using common.py's load_config
        config = load_config(args.config)
        logger.info("Configuration loaded successfully.")

        # Initialize the inference pipeline
        inference_pipeline = CashForecastingPipeline(config=config, logger=logger)

        # Process each country
        for country_code in args.countries:
            try:
                logger.info(f"\nProcessing country: {country_code}")

                # Run inference
                inference_pipeline.run_inference(
                    country_code=country_code,
                    model_name=args.model_name,
                    effective_date=args.effective_date,
                    inference_template_path=args.inference_template,
                    model_timestamp=args.model_timestamp
                )

            except Exception as e:
                logger.error(f"Failed to process country {country_code}: {e}")
                logger.error("Continuing with next country.\n", exc_info=True)
                continue  # Proceed with the next country even if one fails

        logger.info("Inference pipeline completed for all specified countries.")

    except Exception as e:
        logger.error(f"Critical error in inference pipeline: {e}")
        logger.error("Traceback:", exc_info=True)
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
