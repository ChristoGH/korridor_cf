"""
Cash Forecasting Inference Script

This script performs inference using a trained forecasting model on new data.
It leverages shared utilities from common.py for configuration management,
data processing, logging, and AWS interactions.

Usage:
python cs_scripts/cash_forecast_inference.py \
    --config cs_scripts/config.yaml \
    --countries ZM \
    --model_timestamp 20241209-035546 \
    --effective_date 2024-12-09 \
    --model_name ZM-model-20241209-035546 \
    --inference_template ./cs_data/cash/ZM_20241209-035546/ZM_inference_template.csv
"""

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
        self.scaling_params: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None
        self.scaling_metadata: Optional[Dict[str, Any]] = None

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
        self.s3_handler.download_file(bucket=self.config.bucket, s3_key=scaling_params_key, local_path=scaling_params_path)
        scaling_params = self.data_processor.load_scaling_params(scaling_file=scaling_params_path)

        # Download scaling metadata
        scaling_metadata_path = Path(f"./temp/{country_code}_scaling_metadata.json")
        self.s3_handler.download_file(bucket=self.config.bucket, s3_key=scaling_metadata_key, local_path=scaling_metadata_path)
        scaling_metadata = self.data_processor.load_scaling_metadata(scaling_metadata_file=scaling_metadata_path)

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
        Prepare the inference data based on the provided template and effective date.
        """
        self.logger.info(f"Preparing inference data for country: {country_code}")
        try:
            # Load the inference template
            inference_template_df = self.data_processor.load_data(inference_template_path)
            # Ensure the inference template is not empty
            if inference_template_df.empty:
                raise ValueError("Inference template is empty. No combinations available from the training data.")

            # Log the number of combinations to be used
            num_combinations = len(inference_template_df)
            self.logger.info(f"Number of valid combinations for inference: {num_combinations}")

            # Validate that all combinations in the template have scaling parameters
            valid_combinations = set(self.scaling_params.keys())
            self.logger.debug(f"Valid combinations: {valid_combinations}")
            inference_template_df['IsValid'] = inference_template_df.apply(
                lambda row: (row['Currency'], str(row['BranchId'])) in valid_combinations, axis=1
            )
            invalid_combinations = inference_template_df[~inference_template_df['IsValid']]
            if not invalid_combinations.empty:
                self.logger.error(f"Missing scaling parameters for combinations:\n{invalid_combinations}")
                raise ValueError(f"Missing scaling parameters for some combinations in the inference template.")

            # Proceed with valid combinations only
            inference_template_df = inference_template_df[inference_template_df['IsValid']].drop(columns=['IsValid'])
            self.logger.info(f"Filtered inference template to {len(inference_template_df)} valid combinations.")

            # Generate future dates based on effective_date and forecast_horizon
            effective_date_dt = pd.to_datetime(effective_date)
            future_dates = [effective_date_dt + pd.Timedelta(days=i) for i in range(1, self.config.forecast_horizon + 1)]

            # Create the inference DataFrame using the entire template
            inference_data = pd.DataFrame({
                'ProductId': np.tile(inference_template_df['ProductId'].values, self.config.forecast_horizon),
                'BranchId': np.tile(inference_template_df['BranchId'].values, self.config.forecast_horizon),
                'Currency': np.tile(inference_template_df['Currency'].values, self.config.forecast_horizon),
                'EffectiveDate': effective_date_dt,
                'ForecastDate': np.repeat(future_dates, len(inference_template_df)),
                'Demand': np.nan  # Placeholder for unknown demand
            })
            self.logger.info(f"Inference data generated with shape {inference_data.shape}")

            # Save inference data to CSV
            self.output_dir = Path(f"./inference_results/{country_code}_{self.timestamp}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            inference_file = self.output_dir / f"{country_code}_inference_{self.timestamp}.csv"
            inference_data.to_csv(inference_file, index=False)
            self.logger.info(f"Inference data saved to {inference_file}")

            return str(inference_file)

        except Exception as e:
            self.logger.error(f"Failed to prepare inference data: {e}")
            raise

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
            self.s3_handler.safe_upload(local_path=inference_file, bucket=self.config.bucket, s3_key=s3_inference_key, overwrite=True)
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
                        raise RuntimeError(
                            f"Transform job {transform_job_name} failed with status: {status}. Reason: {failure_reason}")
                    self.logger.info(f"Transform job {transform_job_name} completed successfully.")
                    break

                sleep(sleep_time)
                sleep_time = min(sleep_time * 1.5, 600)  # Exponential backoff up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring transform job {transform_job_name}: {e}")
                sleep(60)  # Wait before retrying

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

        Args:
            forecast_df (pd.DataFrame): DataFrame containing forecasted quantiles (p10, p50, p90).
            country_code (str): The country code.

        Returns:
            pd.DataFrame: DataFrame with restored Demand values.
        """
        self.logger.info("Restoring original scale to forecasted Demand values.")
        try:
            # Load the inference data
            inference_csv = self.output_dir / f"{country_code}_inference_{self.timestamp}.csv"
            inference_data = self.data_processor.load_data(str(inference_csv))

            # Check for data length mismatch
            if len(forecast_df) != len(inference_data):
                self.logger.warning(
                    f"Forecast data length {len(forecast_df)} does not match inference data length {len(inference_data)}"
                )
                # Align the forecast data to match inference data length
                if len(forecast_df) > len(inference_data):
                    self.logger.warning("Forecast data has extra rows. Dropping the extra rows.")
                    forecast_df = forecast_df.iloc[:len(inference_data)]
                else:
                    self.logger.warning("Forecast data has fewer rows. Filling the missing rows with NaN.")
                    missing_rows = len(inference_data) - len(forecast_df)
                    additional_df = pd.DataFrame({
                        'p10': [np.nan] * missing_rows,
                        'p50': [np.nan] * missing_rows,
                        'p90': [np.nan] * missing_rows
                    })
                    forecast_df = pd.concat([forecast_df, additional_df], ignore_index=True)

            # Re-attach necessary columns: ProductId and ForecastDate
            forecast_df = inference_data[['ProductId', 'BranchId', 'Currency', 'ForecastDate']].reset_index(drop=True).join(forecast_df)

            # Restore scaling for each group
            for (currency, branch), params in self.scaling_params.items():
                mask = (forecast_df['Currency'] == currency) & (forecast_df['BranchId'] == branch)
                for quantile in self.config.quantiles:
                    if quantile in forecast_df.columns:
                        # Check if scaling parameters exist
                        if (currency, branch) in self.scaling_params:
                            mean = self.scaling_params[(currency, branch)]['mean']
                            std = self.scaling_params[(currency, branch)]['std'] if self.scaling_params[(currency, branch)]['std'] != 0 else 1
                            forecast_df.loc[mask, quantile] = (
                                (forecast_df.loc[mask, quantile] * std) + mean
                            )
                            self.logger.info(f"Inverse scaled {quantile} for Currency={currency}, Branch={branch}")
                        else:
                            self.logger.warning(
                                f"Scaling parameters not found for Currency={currency}, Branch={branch}. Skipping scaling."
                            )
                    else:
                        self.logger.warning(
                            f"Quantile '{quantile}' not found in forecast data for Currency={currency}, Branch={branch}"
                        )

            self.logger.info("Scale restoration completed successfully.")
            return forecast_df

        except Exception as e:
            self.logger.error(f"Failed to restore forecast scales: {e}")
            raise

    def validate_forecast_data(self, forecast_df: pd.DataFrame) -> None:
        """
        Validate the forecast DataFrame for required columns and data types.

        Args:
            forecast_df (pd.DataFrame): DataFrame containing forecasted quantiles.

        Raises:
            ValueError: If validation fails.
        """
        required_columns = ['ProductId', 'BranchId', 'Currency', 'ForecastDate', 'p10', 'p50', 'p90']
        missing_columns = set(required_columns) - set(forecast_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in forecast data: {missing_columns}")

        # Additional type checks can be added here
        if not pd.api.types.is_numeric_dtype(forecast_df['p10']):
            raise TypeError("Quantile 'p10' must be numeric.")
        if not pd.api.types.is_numeric_dtype(forecast_df['p50']):
            raise TypeError("Quantile 'p50' must be numeric.")
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
            forecast_df = forecast_df.rename(columns={'p50': 'Demand'})

            final_forecast_path = self.output_dir / f"{country_code}_final_forecast_{self.timestamp}.csv"
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
            report_path = self.output_dir / f"{country_code}_forecast_report_{self.timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Statistical report saved to {report_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate statistical report: {e}")
            raise

    def run_inference(self, country_code: str, model_name: str, effective_date: str, inference_template_path: str) -> None:
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

            # Download forecast results
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
