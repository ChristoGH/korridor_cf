# cash_forecast_inference.py

import shutil
import sys
from pathlib import Path
from time import gmtime, strftime, sleep
from datetime import datetime
import logging

import pandas as pd
import numpy as np
import boto3
from sagemaker import Session

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
    """Pipeline for performing inference with trained cash demand models."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize the inference pipeline with configuration."""
        self.config = config
        self.logger = logger
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=self.config.region)
        self.s3_handler = S3Handler(region=self.config.region, logger=self.logger)
        self.data_processor = DataProcessor(logger=self.logger)
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = self.config.role_arn
        self.output_dir: Optional[Path] = None
        self.scaling_params: Optional[Dict[str, Dict[str, float]]] = None
        self.scaling_metadata: Optional[Dict[str, Any]] = None
        # After self.output_dir = Path(f"./output/{country_code}/{self.timestamp}")
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
        # Previously downloaded to ./temp
        local_scaling_params_file = self.scaling_dir / f"{country_code}_scaling_params.json"
        local_scaling_metadata_file = self.scaling_dir / f"{country_code}_scaling_metadata.json"

        shutil.copy(scaling_params_path, local_scaling_params_file)
        shutil.copy(scaling_metadata_path, local_scaling_metadata_file)

        # Now remove the temp files
        scaling_params_path.unlink(missing_ok=True)
        scaling_metadata_path.unlink(missing_ok=True)

        self.logger.info(f"Scaling parameters and metadata also saved locally at {self.scaling_dir}")

        # Validate scaling parameters
        self.data_processor.validate_scaling_parameters(scaling_params)

        # Assign to the pipeline
        self.scaling_params = scaling_params
        self.scaling_metadata = scaling_metadata

        # Clean up temporary files
        scaling_params_path.unlink(missing_ok=True)
        scaling_metadata_path.unlink(missing_ok=True)

        self.logger.info(f"Scaling parameters and metadata loaded and validated for {country_code}")

    def prepare_inference_data(self, country_code: str, effective_date: str, inference_template_path: str) -> str:
        """
        Prepare inference data by scaling the 'Demand' column if present,
        or setting 'Demand_scaled' based on the last known value from scaling parameters.

        Args:
            country_code (str): The country code.
            effective_date (str): The effective date for forecasting.
            inference_template_path (str): Path to the inference template CSV file.

        Returns:
            str: Path to the scaled inference CSV file.

        Raises:
            ValueError: If scaling parameters are missing for any combination.
        """
        # Load inference data
        inference_df = pd.read_csv(inference_template_path)
        self.logger.info(f"Loaded data with shape {inference_df.shape}")

        # **No need to filter 'IsValid' as the inference template excludes invalid entries**

        # **Create scaling keys based only on Currency and BranchId**
        inference_df['ScalingKey'] = inference_df['Currency'] + '_' + inference_df['BranchId'].astype(str)

        # Map scaling parameters
        scaling_means = inference_df['ScalingKey'].map(lambda x: self.scaling_params.get(x, {}).get('mean'))
        scaling_stds = inference_df['ScalingKey'].map(lambda x: self.scaling_params.get(x, {}).get('std'))

        # Identify missing scaling parameters
        missing = inference_df[scaling_means.isnull() | scaling_stds.isnull()]
        if not missing.empty:
            self.logger.error(
                f"Missing scaling parameters for combinations:\n{missing[['ProductId', 'BranchId', 'Currency']]}")
            raise ValueError("Missing scaling parameters for some combinations in the inference template.")

        # Handle 'Demand' column absence
        if 'Demand' in inference_df.columns:
            # Apply scaling if 'Demand' exists
            inference_df['Demand_scaled'] = (inference_df['Demand'] - scaling_means) / scaling_stds
            self.logger.info("'Demand' column found. Applied scaling to 'Demand'.")
        else:
            # Set 'Demand_scaled' based on 'last_value' from scaling parameters
            self.logger.warning(
                "'Demand' column not found in inference data. Setting 'Demand_scaled' based on last known value.")
            inference_df['Demand_scaled'] = inference_df['ScalingKey'].map(
                lambda x: (self.scaling_params.get(x, {}).get('last_value', 0) - self.scaling_params.get(x, {}).get(
                    'mean', 0)) /
                          (self.scaling_params.get(x, {}).get('std', 1) or 1)
            )
            self.logger.info("Set 'Demand_scaled' based on 'last_value' from scaling parameters.")

        # Ensure 'Demand_scaled' is not NaN
        if inference_df['Demand_scaled'].isnull().any():
            self.logger.error("Some 'Demand_scaled' values are NaN. Please check scaling parameters.")
            raise ValueError("NaN values found in 'Demand_scaled' column.")

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
            s3_inference_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference/{Path(inference_file).name}"
            self.s3_handler.safe_upload(local_path=inference_file, bucket=self.config.bucket, s3_key=s3_inference_key,
                                        overwrite=True)
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
        transform_job_name = f"{country_code}-transform-{self.timestamp}"

        output_s3_uri = f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/"

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
                        self.download_transform_job_logs(transform_job_name,
                                                         country_code=transform_job_name.split('-')[0])
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
            local_log_file = local_log_dir / f"{transform_job_name}_CloudWatch.log"
            local_log_file = self.transform_logs_dir / f"{transform_job_name}_CloudWatch.log"
            with open(local_log_file, 'w') as f:
                for event in log_events:
                    f.write(event['message'] + '\n')
            self.logger.info(f"CloudWatch logs saved to {local_log_file}")
            return local_log_file

        except Exception as e:
            self.logger.error(f"Failed to download CloudWatch logs for {transform_job_name}: {e}")
            raise

    def download_forecast_results(self, country_code: str) -> pd.DataFrame:
        """
        Download and combine forecast results from S3.

        Args:
            country_code (str): The country code.

        Returns:
            pd.DataFrame: Combined forecast DataFrame containing p10, p50, p90.

        Raises:
            FileNotFoundError: If no forecast results are found.
            Exception: If download or processing fails.
        """
        self.logger.info(f"Downloading forecast results for country: {country_code}")
        forecast_s3_prefix = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/"
        forecast_files = self.s3_handler.list_s3_objects(bucket=self.config.bucket, prefix=forecast_s3_prefix)

        # Filter for forecast output files (assuming they have .out extension)
        forecast_keys = [obj['Key'] for obj in forecast_files if obj['Key'].endswith('.out')]

        if not forecast_keys:
            self.logger.error(f"No forecast output files found in s3://{self.config.bucket}/{forecast_s3_prefix}")
            raise FileNotFoundError(f"No forecast output files found in s3://{self.config.bucket}/{forecast_s3_prefix}")

        # Create temporary directory for downloads
        temp_dir = Path("./temp_forecast_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)

        forecast_dfs = []
        try:
            for key in forecast_keys:
                local_file = temp_dir / Path(key).name
                self.s3_handler.download_file(bucket=self.config.bucket, s3_key=key, local_path=local_file)
                self.logger.info(f"Downloaded forecast file {key} to {local_file}")

                # Read forecast data, skipping the first two rows (headers and metadata)
                # Adjust 'sep' if your CSV uses a different delimiter
                df = pd.read_csv(local_file, header=None, skiprows=2, sep=',')  # Change sep if needed

                # Debugging: Log the DataFrame shape and first few rows
                self.logger.debug(f"Forecast DataFrame shape after reading CSV: {df.shape}")
                self.logger.debug(f"Forecast DataFrame columns before assignment: {df.columns.tolist()}")
                self.logger.debug(f"Forecast DataFrame head:\n{df.head()}")

                # Assign meaningful column names based on known structure
                # Ensure the number of column names matches the actual number of columns in df
                expected_columns = {
                    8: [
                        'Currency', 'BranchId', 'ProductId', 'EffectiveDate', 'p10', 'p50', 'p90', 'mean'
                    ],
                    10: [
                        'Currency', 'BranchId', 'ProductId', 'BranchId_dup',
                        'Currency_dup', 'EffectiveDate', 'p10', 'p50', 'p90', 'mean'
                    ]
                }

                num_columns = len(df.columns)
                if num_columns in expected_columns:
                    df.columns = expected_columns[num_columns]
                    self.logger.debug(f"Assigned column names: {df.columns.tolist()}")
                else:
                    self.logger.error(
                        f"Unexpected number of columns ({num_columns}) in forecast file {key}. Expected {list(expected_columns.keys())}."
                    )
                    raise ValueError(
                        f"Unexpected number of columns ({num_columns}) in forecast file {key}. Expected {list(expected_columns.keys())}."
                    )

                # Extract only the quantile columns
                quantile_columns = ['p10', 'p50', 'p90']
                missing_quantiles = [q for q in quantile_columns if q not in df.columns]
                if missing_quantiles:
                    self.logger.warning(f"Missing quantiles {missing_quantiles} in forecast file {key}")
                    # Decide whether to skip, fill with NaN, or handle differently

                df_quantiles = df[quantile_columns].copy()

                # Replace 'ERROR' strings with NaN
                df_quantiles.replace('ERROR', np.nan, inplace=True)

                # Convert quantile columns to numeric, coercing errors to NaN
                for q in quantile_columns:
                    df_quantiles[q] = pd.to_numeric(df_quantiles[q], errors='coerce')

                forecast_dfs.append(df_quantiles)
                self.logger.info(f"Processed forecast file {key} with shape {df_quantiles.shape}")

            # Combine all forecast DataFrames
            combined_forecast = pd.concat(forecast_dfs, ignore_index=True)
            self.logger.info(f"Combined forecast data shape: {combined_forecast.shape}")

            return combined_forecast

        except Exception as e:
            self.logger.error(f"Failed to download or process forecast results: {e}")
            raise

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.info(f"Cleaned up temporary forecast downloads at {temp_dir}")

    def restore_forecasts_scale(self, forecast_df: pd.DataFrame, country_code: str) -> pd.DataFrame:
        """
        Restore the original scale of the forecasted Demand values.
        """
        self.logger.info("Restoring original scale to forecasted Demand values.")
        try:
            # Corrected file path to load the scaled inference data
            inference_csv = self.output_dir / f"scaled_inference_{country_code}.csv"

            if not inference_csv.exists():
                self.logger.error(f"Inference CSV file does not exist: {inference_csv}")
                raise FileNotFoundError(f"Inference CSV file does not exist: {inference_csv}")

            inference_data = self.data_processor.load_data(str(inference_csv))

            # Check for required columns
            required_columns = ['ProductId', 'BranchId', 'Currency', 'ForecastDate']
            missing_cols = set(required_columns) - set(inference_data.columns)
            if missing_cols:
                self.logger.error(f"Inference CSV is missing required columns: {missing_cols}")
                raise ValueError(f"Inference CSV is missing required columns: {missing_cols}")

            # Check for 'Demand_scaled' column
            if 'Demand_scaled' not in inference_data.columns:
                self.logger.error("'Demand_scaled' column is missing from the inference data.")
                raise KeyError("'Demand_scaled' column is missing from the inference data.")

            # Re-attach necessary columns: ProductId and ForecastDate
            forecast_df = inference_data[['ProductId', 'BranchId', 'Currency', 'ForecastDate']].reset_index(
                drop=True).join(forecast_df)

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
        required_columns = ['ProductId', 'BranchId', 'Currency', 'ForecastDate', 'p10', 'Demand', 'p90']
        missing_columns = set(required_columns) - set(forecast_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in forecast data: {missing_columns}")

        # Additional type checks can be added here
        if not pd.api.types.is_numeric_dtype(forecast_df['p10']):
            raise TypeError("Quantile 'p10' must be numeric.")
        if not pd.api.types.is_numeric_dtype(forecast_df['Demand']):
            raise TypeError("Quantile 'p50' (renamed to 'Demand') must be numeric.")
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
            # Optionally, rename 'p50' to 'Demand' if required
            # Already done in restore_forecasts_scale
            # forecast_df = forecast_df.rename(columns={'p50': 'Demand'})
            final_forecast_path = self.inference_dir / f"{country_code}_final_forecast_{self.timestamp}.csv"
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
                'timestamp': self.timestamp,
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
            report_path = self.inference_dir / f"{country_code}_forecast_report_{self.timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Statistical report saved to {report_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate statistical report: {e}")
            raise

    def run_inference(self, country_code: str, model_name: str, effective_date: str,
                      inference_template_path: str) -> None:
        """
        Execute the complete inference pipeline.

        Args:
            country_code (str): The country code.
            model_name (str): The SageMaker model name.
            effective_date (str): The effective date for forecasting.
            inference_template_path (str): Path to the inference template CSV file.

        Raises:
            Exception: If any step in the pipeline fails.
        """
        try:
            self.logger.info(f"Starting inference pipeline for country: {country_code}")

            # Prepare inference data
            inference_file = self.prepare_inference_data(country_code, effective_date, inference_template_path)

            # Upload inference data to S3
            s3_inference_uri = self.upload_inference_data(inference_file, country_code)

            # Run batch transform job
            self.run_batch_transform(country_code, model_name, s3_inference_uri)

            # Get transform job name
            transform_job_name = f"{country_code}-transform-{self.timestamp}"

            # Download forecast results from S3 to local directory
            transform_output_dir = self.download_transform_output(transform_job_name, country_code)

            # Download CloudWatch logs
            transform_logs_file = self.download_transform_job_logs(transform_job_name, country_code)

            # Optionally, process the downloaded files as needed
            # For example, combine CSVs, perform post-processing, etc.

            # Download scaled inference data locally if not already saved
            scaled_inference_file = Path(f"scaled_inference_{country_code}.csv")
            if not scaled_inference_file.exists():
                shutil.copy(inference_file, scaled_inference_file)
                self.logger.info(f"Copied scaled inference data to {scaled_inference_file}")

            # Proceed with downloading and processing forecast results
            forecast_df = self.download_forecast_results(country_code)

            # Restore original scale
            restored_forecast_df = self.restore_forecasts_scale(forecast_df, country_code)

            # Validate forecast data
            self.validate_forecast_data(restored_forecast_df)

            # Save final forecasts
            self.save_final_forecasts(restored_forecast_df, country_code)

            # Generate statistical report
            self.generate_statistical_report(restored_forecast_df, country_code)

            self.logger.info(f"Inference pipeline completed successfully for {country_code}")

        except Exception as e:
            self.logger.error(f"Inference pipeline failed for {country_code}: {e}")
            self.logger.error("Traceback:", exc_info=True)
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

        # Create the output directory
        inference_pipeline.output_dir = Path(f"./output/{args.countries[0]}/{inference_pipeline.timestamp}")
        inference_pipeline.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {inference_pipeline.output_dir}")

        # Process each country
        for country_code in args.countries:
            try:
                logger.info(f"\nProcessing country: {country_code}")

                # Load and validate scaling parameters
                inference_pipeline.load_scaling_parameters(country_code, args.model_timestamp)

                # Run inference
                inference_pipeline.run_inference(
                    country_code=country_code,
                    model_name=args.model_name,
                    effective_date=args.effective_date,
                    inference_template_path=args.inference_template
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
