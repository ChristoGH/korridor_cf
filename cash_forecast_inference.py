# cash_forecast_inference.py
# Usage:
# python cs_scripts/cash_forecast_inference.py \
#     --model_timestamp 20241203-054112 \
#     --effective_date 2024-07-15 \
#     --config cs_scripts/config.yaml \
#     --countries ZM \
#     --model_name ZM-model-20241203-054112 \
#     --inference_template ./cs_data/cash/ZM_20241203-054112/ZM_inference_template.csv

import argparse
import boto3
import pandas as pd
import numpy as np
import os
import sys
import json
import shutil
from time import gmtime, strftime, sleep
from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from sagemaker import Session
import yaml

from common import (
    Config,
    load_and_validate_config,
    setup_logging,
    safe_s3_upload,
    load_scaling_parameters
)
from cash_forecast_lib import (
    load_scaling_metadata  # Newly added
)
import logging  # Ensure logging is imported


@dataclass
class ScalingState:
    """State container for scaling operations"""
    original_data: pd.DataFrame
    scaled_data: pd.DataFrame
    params: Dict[str, Any]
    timestamp: str
    successful: bool = False


def validate_scaling_parameters(scaling_params: Dict[Tuple[str, str], Any]) -> None:
    """Validate the structure and content of scaling parameters."""
    if not scaling_params:
        raise ValueError("Scaling parameters dictionary is empty")

    required_keys = {'mean', 'std', 'min', 'max', 'last_value', 'n_observations'}

    for (currency, branch), params in scaling_params.items():
        if not isinstance(currency, str) or not isinstance(branch, str):
            raise ValueError(f"Invalid key format: ({currency}, {branch})")

        missing_keys = required_keys - set(params.keys())
        if missing_keys:
            raise ValueError(f"Missing required scaling parameters {missing_keys} for {currency}-{branch}")

        # Validate numeric values
        for key, value in params.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Non-numeric value {value} for {key} in {currency}-{branch}")

        # Validate statistical coherence
        if params['std'] < 0:
            raise ValueError(f"Negative standard deviation for {currency}-{branch}")
        if params['max'] < params['min']:
            raise ValueError(f"Max value less than min for {currency}-{branch}")
        if params['n_observations'] <= 0:
            raise ValueError(f"Invalid number of observations for {currency}-{branch}")


class CashForecastingInference:
    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize the inference pipeline with configuration."""
        self.config = config
        self.logger = logger
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=config.region)
        self.s3_client = boto3.client('s3')
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = config.role_arn
        self.output_dir = None
        self.scaling_params = None
        self.scaling_metadata = None
        self.scaling_state = None

    def _load_scaling_parameters(self, country_code: str, model_timestamp: str) -> None:
        """
        Load and validate scaling parameters and metadata from S3.

        Args:
            country_code (str): The country code.
            model_timestamp (str): The timestamp of the training model.

        Raises:
            FileNotFoundError: If scaling parameters or metadata are not found.
            ValueError: If scaling parameters or metadata are invalid.
        """
        try:
            # Construct S3 paths
            scaling_params_key = f"{self.config.prefix}-{country_code}/{model_timestamp}/scaling/{country_code}_scaling_params.json"
            scaling_metadata_key = f"{self.config.prefix}-{country_code}/{model_timestamp}/scaling/{country_code}_scaling_metadata.json"

            # Download scaling parameters
            self.logger.info(f"Downloading scaling parameters from s3://{self.config.bucket}/{scaling_params_key}")
            self.scaling_params = load_scaling_parameters(
                s3_client=self.s3_client,
                logger=self.logger,
                bucket=self.config.bucket,
                scaling_params_key=scaling_params_key
            )

            # Download scaling metadata using the new function
            self.scaling_metadata = load_scaling_metadata(
                s3_client=self.s3_client,
                logger=self.logger,
                bucket=self.config.bucket,
                scaling_metadata_key=scaling_metadata_key
            )

            # Validate scaling parameters
            validate_scaling_parameters(self.scaling_params)

            # Validate metadata structure
            required_metadata = {'scaling_method', 'scaling_level', 'scaled_column', 'scaling_stats', 'generated_at', 'country_code'}
            if not all(key in self.scaling_metadata for key in required_metadata):
                self.logger.error("Invalid scaling metadata structure.")
                raise ValueError("Invalid scaling metadata structure.")

            # Additional validations can be added here if necessary

            self.logger.info(f"Successfully loaded and validated scaling parameters and metadata for {country_code}")

        except self.s3_client.exceptions.NoSuchKey:
            self.logger.error(f"Scaling parameters or metadata file not found in S3 for {country_code}")
            raise FileNotFoundError(f"Scaling parameters or metadata file not found in S3 for {country_code}")
        except ValueError as ve:
            self.logger.error(f"Validation error: {ve}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load scaling parameters and metadata: {e}")
            raise

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

    def prepare_inference_data(self, inference_template_file: str, country_code: str, effective_date: str) -> str:
        """Prepare the inference data based on the template."""
        try:
            # Load and validate inference template
            inference_template = pd.read_csv(inference_template_file)
            required_columns = {'ProductId', 'BranchId', 'Currency'}
            if not required_columns.issubset(inference_template.columns):
                missing = required_columns - set(inference_template.columns)
                raise ValueError(f"Missing required columns in template: {missing}")

            # Ensure correct data types
            inference_template['ProductId'] = inference_template['ProductId'].astype(str)
            inference_template['BranchId'] = inference_template['BranchId'].astype(str)
            inference_template['Currency'] = inference_template['Currency'].astype(str)

            # Parse and validate effective date
            try:
                effective_date_dt = pd.to_datetime(effective_date)
            except ValueError:
                raise ValueError(f"Invalid effective date format: {effective_date}")

            # Generate future dates
            future_dates = [effective_date_dt + pd.Timedelta(days=i)
                            for i in range(1, self.config.forecast_horizon + 1)]

            # Create inference data
            inference_data = pd.DataFrame({
                'ProductId': np.tile(inference_template['ProductId'].values, self.config.forecast_horizon),
                'BranchId': np.tile(inference_template['BranchId'].values, self.config.forecast_horizon),
                'Currency': np.tile(inference_template['Currency'].values, self.config.forecast_horizon),
                'EffectiveDate': effective_date_dt,
                'ForecastDate': np.repeat(future_dates, len(inference_template)),
                'Demand': np.nan  # Demand is unknown for future dates
            })

            # Validate currency-branch combinations against scaling parameters
            template_combinations = set(zip(inference_template['Currency'], inference_template['BranchId']))
            scaling_combinations = set(self.scaling_params.keys())
            missing_combinations = template_combinations - scaling_combinations

            if missing_combinations:
                self.logger.error(f"Template contains combinations without scaling parameters: {missing_combinations}")
                raise ValueError(
                    f"Template contains combinations without scaling parameters: {missing_combinations}"
                )

            # Create output directory
            self.output_dir = f"./cs_data/cash/{country_code}_{self.timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)

            # Save inference data
            inference_file = os.path.join(self.output_dir, f"{country_code}_inference.csv")
            inference_data.to_csv(inference_file, index=False)
            self.logger.info(f"Inference data prepared and saved to {inference_file}")

            # Store state for potential rollback
            self.scaling_state = ScalingState(
                original_data=inference_data.copy(),
                scaled_data=inference_data.copy(),
                params=self.scaling_params.copy(),
                timestamp=self.timestamp
            )

            return inference_file

        except Exception as e:
            self.logger.error(f"Error preparing inference data: {e}")
            raise

    def run_inference(self, country_code: str, model_name: str, inference_file: str) -> None:
        """Run the complete inference process."""
        try:
            # Upload inference data to S3
            s3_inference_key = (f"{self.config.prefix}-{country_code}/"
                                f"{self.timestamp}/inference/{os.path.basename(inference_file)}")
            self._safe_s3_upload(inference_file, s3_inference_key, overwrite=True)
            s3_inference_data_uri = f"s3://{self.config.bucket}/{s3_inference_key}"
            self.logger.info(f"Inference data uploaded to {s3_inference_data_uri}")

            # Read inference data into DataFrame
            inference_df = pd.read_csv(inference_file)
            self.logger.info(f"Inference data loaded from {inference_file}")

            # Run batch transform job with batch_number=1
            self._run_batch_transform_job(country_code, model_name, s3_inference_data_uri, batch_number=1)

            # Get and process results, passing inference_df for 'ForecastDate' assignment
            forecast_df = self._get_forecast_result(country_code, inference_df)

            # Generate statistical report
            self._generate_statistical_report(forecast_df, country_code)

            # Save final forecasts
            self._save_forecasts(country_code, forecast_df)

            # Mark scaling state as successful
            if self.scaling_state:
                self.scaling_state.successful = True

        except Exception as e:
            self.logger.error(f"Error in inference pipeline: {e}")
            if self.scaling_state and not self.scaling_state.successful:
                self._rollback_scaling()
            raise

    def forecast(self, country_code: str, model_name: str, template_file: str,
                backtesting: bool = False) -> pd.DataFrame:
        """Generate forecasts starting from specified dates."""
        try:
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

                    # Save to CSV using self.output_dir
                    inference_file = os.path.join(self.output_dir, f"{country_code}_inference_batch_{batch_number}.csv")
                    temp_df.to_csv(inference_file, index=False)

                    # Upload to S3
                    s3_inference_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference/{os.path.basename(inference_file)}"
                    self._safe_s3_upload(inference_file, s3_inference_key, overwrite=True)
                    s3_inference_data_uri = f"s3://{self.config.bucket}/{s3_inference_key}"

                    # Run batch transform
                    self._run_batch_transform_job(country_code, model_name, s3_inference_data_uri, batch_number)

            # After processing all batches, retrieve and combine forecast results
            forecast_df = self._get_forecast_result(country_code, inference_df=None)  # Pass inference_df if needed

            # Save forecasts
            self._save_forecasts(country_code, forecast_df)

            return forecast_df

        except Exception as e:
            self.logger.error(f"Error during forecasting: {e}")
            raise

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
                    'ContentType': 'text/csv',  # Ensure this matches your input format
                    'SplitType': 'Line'
                },
                TransformOutput={
                    'S3OutputPath': output_s3_uri,
                    'AssembleWith': 'Line',
                    'Accept': 'text/csv'  # Set to 'text/csv' if model outputs CSV
                },
                TransformResources={
                    'InstanceType': self.config.instance_type,
                    'InstanceCount': self.config.instance_count
                }
            )
            self.logger.info(f"Created transform job {transform_job_name}")
            self._monitor_transform_job(transform_job_name)
        except Exception as e:
            self.logger.error(f"Failed to create transform job {transform_job_name}: {e}")
            raise

    def _monitor_transform_job(self, transform_job_name: str, max_wait_time: int = 24 * 60 * 60) -> None:
        """Monitor the batch transform job until completion with a timeout."""
        self.logger.info(f"Monitoring transform job {transform_job_name}")
        sleep_time = 30  # Start with 30 seconds
        elapsed_time = 0
        while elapsed_time < max_wait_time:
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
                    return
                sleep(sleep_time)
                elapsed_time += sleep_time
                sleep_time = min(int(sleep_time * 1.5), 600)  # Increase sleep time up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring transform job: {e}")
                sleep_time = min(int(sleep_time * 1.5), 600)
                elapsed_time += sleep_time
                sleep(60)  # Wait before retrying
        raise TimeoutError(f"Transform job {transform_job_name} did not complete within {max_wait_time} seconds.")

    def _get_forecast_result(self, country_code: str, inference_df: pd.DataFrame = None) -> pd.DataFrame:
        """Download and process forecast results."""
        output_s3_prefix = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/"
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.config.bucket, Prefix=output_s3_prefix)
            if 'Contents' not in response:
                raise FileNotFoundError(f"No inference results found in S3 for {output_s3_prefix}")

            temp_dir = f"./temp/{country_code}_{self.timestamp}/"
            os.makedirs(temp_dir, exist_ok=True)

            try:
                forecast_data = []
                for obj in response['Contents']:
                    s3_key = obj['Key']
                    if s3_key.endswith('.out'):
                        local_file = os.path.join(temp_dir, os.path.basename(s3_key))
                        try:
                            self.s3_client.download_file(self.config.bucket, s3_key, local_file)
                            self.logger.info(f"Downloaded forecast file {s3_key} to {local_file}")

                            # Read and process the forecast data with explicit dtypes
                            df = pd.read_csv(local_file, header=None)
                            # Assign column names based on quantiles
                            df.columns = self.config.quantiles
                            # Convert to numeric immediately
                            for col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
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
                self.logger.info(f"Combined forecast data shape: {forecast_df.shape}")

                # Add metadata columns from inference_df if provided
                if inference_df is not None:
                    for col in ['ForecastDate', 'ProductId', 'BranchId', 'Currency', 'EffectiveDate']:
                        forecast_df[col] = inference_df[col].values
                else:
                    # If inference_df is not provided, ensure that these columns exist or handle accordingly
                    for col in ['ForecastDate', 'ProductId', 'BranchId', 'Currency', 'EffectiveDate']:
                        if col not in forecast_df.columns:
                            forecast_df[col] = np.nan
                            self.logger.warning(f"Column '{col}' not found in forecast data. Filled with NaN.")

                # Debug logging
                self.logger.info(f"Column dtypes before scaling: {forecast_df.dtypes}")

                # Inverse scaling with type validation
                self.logger.info("Restoring original scale to forecasts...")
                for (currency, branch), params in self.scaling_params.items():
                    mask = (forecast_df['Currency'] == currency) & (forecast_df['BranchId'] == branch)
                    if mask.sum() == 0:
                        self.logger.warning(f"No forecasts found for Currency={currency}, Branch={branch}")
                        continue

                    std_val = float(params.std) if params.std != 0 else 1.0
                    mean_val = float(params.mean)

                    for quantile in self.config.quantiles:
                        if quantile in forecast_df.columns:
                            try:
                                values = forecast_df.loc[mask, quantile]
                                forecast_df.loc[mask, quantile] = values * std_val + mean_val
                                self.logger.info(f"Scaled {quantile} for {currency}-{branch}")
                            except Exception as e:
                                self.logger.error(f"Error scaling {quantile} for {currency}-{branch}: {e}")
                                raise

                self.logger.info("Inverse scaling completed successfully")
                return forecast_df

            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    self.logger.info(f"Removed temporary directory {temp_dir}")

        except Exception as e:
            self.logger.error(f"Error in _get_forecast_result: {e}")
            raise

    def _generate_statistical_report(self, result_df: pd.DataFrame, country_code: str) -> None:
        """Generate detailed statistical report for forecasts."""
        report_file = os.path.join(self.output_dir, f"{country_code}_forecast_stats.json")
        try:
            stats = {
                'global_stats': {
                    'total_forecasts': len(result_df),
                    'timestamp': self.timestamp,
                    'scaling_method': self.scaling_metadata.get('scaling_method', 'N/A'),
                    'forecast_horizon': self.config.forecast_horizon,
                    'quantiles': self.config.quantiles
                },
                'group_stats': {}
            }

            for (currency, branch), params in self.scaling_params.items():
                mask = (result_df['Currency'] == currency) & (result_df['BranchId'] == branch)
                group_data = result_df[mask]

                group_stats = {
                    'n_forecasts': len(group_data),
                    'scaling_params': params.__dict__,
                    'quantile_stats': {},
                    'validation': {
                        'negative_values': False,
                        'extreme_outliers': False
                    }
                }

                for quantile in self.config.quantiles:
                    if quantile not in group_data.columns:
                        self.logger.warning(f"Quantile '{quantile}' missing in group {currency}-{branch}")
                        continue

                    values = group_data[quantile]

                    # Calculate basic statistics
                    stats_dict = values.describe().to_dict()

                    # Calculate additional metrics
                    zscore = np.abs((values - params.mean) / (params.std if params.std != 0 else 1))
                    outliers = (zscore > 5).sum()

                    group_stats['quantile_stats'][quantile] = {
                        'basic_stats': stats_dict,
                        'outliers': int(outliers),
                        'negative_values': int((values < 0).sum())
                    }

                    # Update validation flags
                    group_stats['validation']['negative_values'] |= bool((values < 0).any())
                    group_stats['validation']['extreme_outliers'] |= bool(outliers > 0)

                stats['group_stats'][f"{currency}_{branch}"] = group_stats

            # Save report
            with open(report_file, 'w') as f:
                json.dump(stats, f, indent=2)

            self.logger.info(f"Statistical report generated: {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to generate statistical report: {e}")
            raise

    def _save_forecasts(self, country_code: str, forecasts_df: pd.DataFrame) -> None:
        """Save the forecasts with appropriate format and validation."""
        # Required columns
        required_columns = [
            'ProductId', 'BranchId', 'Currency',
            'EffectiveDate', 'ForecastDate'] + self.config.quantiles

        self.logger.info(f"Forecast df columns: {forecasts_df.columns.tolist()}")

        missing_columns = set(required_columns) - set(forecasts_df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in forecasts DataFrame: {missing_columns}")

        # Convert date columns to datetime
        forecasts_df['EffectiveDate'] = pd.to_datetime(forecasts_df['EffectiveDate'])
        forecasts_df['ForecastDate'] = pd.to_datetime(forecasts_df['ForecastDate'])

        # Ensure quantile columns are numeric
        try:
            forecasts_df[self.config.quantiles] = forecasts_df[self.config.quantiles].apply(pd.to_numeric, errors='coerce')
        except Exception as e:
            self.logger.error(f"Failed to convert quantile columns to numeric: {e}")
            raise ValueError("Quantile columns contain non-numeric values.")

        # Calculate 'ForecastDay'
        forecasts_df['ForecastDay'] = (forecasts_df['ForecastDate'] - forecasts_df['EffectiveDate']).dt.days + 1

        # Filter forecasts within horizon
        forecasts_df = forecasts_df[
            (forecasts_df['ForecastDay'] >= 1) & (forecasts_df['ForecastDay'] <= self.config.forecast_horizon)
        ]

        # Pivot the data
        try:
            forecasts_pivot = forecasts_df.pivot_table(
                index=['ProductId', 'BranchId', 'Currency', 'EffectiveDate'],
                columns='ForecastDay',
                values=self.config.quantiles,
                aggfunc='mean'  # Explicitly specify the aggregation function
            )
        except Exception as e:
            self.logger.error(f"Pivot table aggregation failed: {e}")
            raise ValueError("Aggregation function failed due to non-numeric quantile columns.")

        # Rename columns to include quantile and day information
        forecasts_pivot.columns = [f"{quantile}_Day{int(day)}" for quantile, day in forecasts_pivot.columns]

        # Reset index
        forecasts_pivot.reset_index(inplace=True)

        # Save results
        output_file = f"./results/{country_code}_{self.timestamp}/final_forecast.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        forecasts_pivot.to_csv(output_file, index=False)
        self.logger.info(f"Final forecast saved to {output_file}")

    def _rollback_scaling(self) -> None:
        """Rollback scaling operations in case of failure."""
        if self.scaling_state and not self.scaling_state.successful:
            self.logger.warning("Rolling back scaling operations...")
            try:
                rollback_file = os.path.join(
                    self.output_dir,
                    f"rollback_{self.scaling_state.timestamp}.csv"
                )
                self.scaling_state.original_data.to_csv(rollback_file, index=False)
                self.logger.info(f"Scaling rollback successful. Data saved to {rollback_file}")
            except Exception as e:
                self.logger.error(f"Failed to rollback scaling: {e}")


def setup_logging_custom(timestamp: str) -> logging.Logger:
    """Setup logging for the main process."""
    return setup_logging('CashForecastInference', timestamp)


def validate_args(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Validate command line arguments."""
    # Validate model timestamp format
    try:
        datetime.strptime(args.model_timestamp, "%Y%m%d-%H%M%S")
    except ValueError:
        raise ValueError(
            f"Invalid model_timestamp format: {args.model_timestamp}. "
            f"Expected format: YYYYMMDD-HHMMSS"
        )

    # Validate effective date format
    try:
        datetime.strptime(args.effective_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Invalid effective_date format: {args.effective_date}. "
            f"Expected format: YYYY-MM-DD"
        )

    # Validate config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Validate inference template
    if not os.path.exists(args.inference_template):
        raise FileNotFoundError(f"Inference template not found: {args.inference_template}")

    # Validate country codes
    if not args.countries:
        raise ValueError("No country codes provided")

    # Log validated arguments
    logger.info("Arguments validated successfully:")
    logger.info(f"  Model Timestamp: {args.model_timestamp}")
    logger.info(f"  Effective Date: {args.effective_date}")
    logger.info(f"  Countries: {args.countries}")
    logger.info(f"  Model Name: {args.model_name}")


def main():
    """Main function for cash forecasting inference."""
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())

    # Setup logging
    logger = setup_logging_custom(timestamp)
    logger.info("Starting Cash Forecasting Inference Pipeline")

    # Parse arguments
    parser = argparse.ArgumentParser(description='Cash Forecasting Inference')
    parser.add_argument('--model_timestamp', type=str, required=True,
                        help='Timestamp of the training run (YYYYMMDD-HHMMSS)')
    parser.add_argument('--effective_date', type=str, required=True,
                        help='Effective date for forecasts (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--countries', nargs='+', default=['ZM'],
                        help='Country codes to process')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the existing SageMaker model to use for inference')
    parser.add_argument('--inference_template', type=str, required=True,
                        help='Path to the inference template CSV file')

    args = parser.parse_args()

    try:
        # Validate arguments
        validate_args(args, logger)

        # Load and validate configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_and_validate_config(args.config)
        logger.info("Configuration validated successfully")

        # Process each country
        for country_code in args.countries:
            logger.info(f"\nProcessing country: {country_code}")
            try:
                # Initialize pipeline
                inference_pipeline = CashForecastingInference(config, logger)

                # Load scaling parameters and metadata
                logger.info(f"Loading scaling parameters and metadata for {country_code}")
                inference_pipeline._load_scaling_parameters(
                    country_code,
                    args.model_timestamp
                )

                # Prepare inference data
                logger.info(f"Preparing inference data for {country_code}")
                inference_file = inference_pipeline.prepare_inference_data(
                    args.inference_template,
                    country_code,
                    args.effective_date
                )

                # Run inference
                logger.info(f"Running inference for {country_code}")
                inference_pipeline.run_inference(
                    country_code,
                    args.model_name,
                    inference_file
                )

                logger.info(f"Successfully completed processing for {country_code}")

            except Exception as e:
                logger.error(f"Failed to process {country_code}: {e}")
                logger.debug("Exception details:", exc_info=True)
                continue  # Continue with next country even if one fails

        logger.info("\nInference pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Critical error in main process: {e}")
        logger.debug("Traceback:", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up any temporary files or resources
        logger.info("Cleaning up resources...")
        try:
            # Add any cleanup code here if necessary
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()
