# cash_forecast_inference.py

import argparse
import boto3
import pandas as pd
import numpy as np
import os
import logging
import sys
import json
import shutil
from time import gmtime, strftime, sleep
from sagemaker import Session
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import yaml
from pydantic import BaseModel, ValidationError
from urllib.parse import urlparse
from datetime import datetime, timedelta


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


@dataclass
class ScalingState:
    """State container for scaling operations"""
    original_data: pd.DataFrame
    scaled_data: pd.DataFrame
    params: Dict
    timestamp: str
    successful: bool = False


def validate_scaling_parameters(scaling_params: Dict) -> None:
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
    def __init__(self, config: Config):
        """Initialize the inference pipeline with configuration."""
        self.config = config
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=config.region)
        self.s3_client = boto3.client('s3')
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = config.role_arn
        self._setup_logging()
        self.output_dir = None
        self.scaling_params = None
        self.scaling_metadata = None
        self.scaling_state = None

    def _setup_logging(self):
        """Configure logging for the pipeline."""
        self.logger = logging.getLogger('CashForecastInference')
        self.logger.setLevel(logging.INFO)

        # Create formatters and handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(f'{log_dir}/cash_forecast_inference_{self.timestamp}.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # Stream handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # Add handlers
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

    def _load_scaling_parameters(self, country_code: str, model_timestamp: str) -> None:
        """
        Load and validate scaling parameters from S3.

        Args:
            country_code (str): The country code.
            model_timestamp (str): The timestamp of the training model.

        Raises:
            FileNotFoundError: If scaling parameters are not found.
            ValueError: If scaling parameters are invalid.
        """
        try:
            # Construct S3 paths
            scaling_params_key = f"{self.config.prefix}-{country_code}/{model_timestamp}/scaling/{country_code}_scaling_params.json"
            scaling_metadata_key = f"{self.config.prefix}-{country_code}/{model_timestamp}/scaling/{country_code}_scaling_metadata.json"

            # Download scaling parameters
            self.logger.info(f"Downloading scaling parameters from s3://{self.config.bucket}/{scaling_params_key}")
            scaling_params_obj = self.s3_client.get_object(Bucket=self.config.bucket, Key=scaling_params_key)
            scaling_params = json.loads(scaling_params_obj['Body'].read().decode('utf-8'))
            self.scaling_params = {
                tuple(k.split('_')): v
                for k, v in scaling_params.items()
            }

            # Download scaling metadata
            self.logger.info(f"Downloading scaling metadata from s3://{self.config.bucket}/{scaling_metadata_key}")
            scaling_metadata_obj = self.s3_client.get_object(Bucket=self.config.bucket, Key=scaling_metadata_key)
            self.scaling_metadata = json.loads(scaling_metadata_obj['Body'].read().decode('utf-8'))

            # Validate scaling parameters
            validate_scaling_parameters(self.scaling_params)

            # Validate metadata structure
            required_metadata = {'scaling_method', 'scaling_level', 'scaled_column', 'scaling_stats'}
            if not all(key in self.scaling_metadata for key in required_metadata):
                raise ValueError("Invalid scaling metadata structure")

            self.logger.info(f"Successfully loaded and validated scaling parameters for {country_code}")

        except self.s3_client.exceptions.NoSuchKey:
            self.logger.error(f"Scaling parameters file not found in S3 for {country_code}")
            raise FileNotFoundError(f"Scaling parameters file not found in S3 for {country_code}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format in scaling files: {e}")
            raise ValueError(f"Invalid JSON format in scaling files: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load scaling parameters: {e}")
            raise

    def prepare_inference_data(self,
                               inference_template_file: str,
                               country_code: str,
                               effective_date: str) -> str:
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

            # Run batch transform job
            self._run_batch_transform_job(country_code, model_name, s3_inference_data_uri)

            # Get and process results
            forecast_df = self._get_forecast_result(country_code)

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
            self.logger.info(f"Transform job {transform_job_name} created successfully.")
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
                if s3_key.endswith('.out'):  # Assuming forecast output files have .out extension
                    local_file = os.path.join(temp_dir, os.path.basename(s3_key))
                    try:
                        self.s3_client.download_file(self.config.bucket, s3_key, local_file)
                        self.logger.info(f"Downloaded forecast file {s3_key} to {local_file}")

                        # Read forecast data
                        df = pd.read_csv(local_file)
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

            # Load scaling parameters (Already loaded in self.scaling_params)

            # Inverse scaling
            self.logger.info("Restoring original scale to forecasts...")
            for (currency, branch), params in self.scaling_params.items():
                mask = (forecast_df['Currency'] == currency) & (forecast_df['BranchId'] == branch)
                if mask.sum() == 0:
                    self.logger.warning(f"No forecasts found for Currency={currency}, Branch={branch}")
                    continue

                for quantile in self.config.quantiles:
                    if quantile in forecast_df.columns:
                        forecast_df.loc[mask, quantile] = (
                            (forecast_df.loc[mask, quantile] * (params['std'] if params['std'] != 0 else 1)) +
                            params['mean']
                        )
                        self.logger.info(
                            f"Inverse scaled {quantile} for Currency={currency}, Branch={branch}"
                        )
                    else:
                        self.logger.warning(f"Quantile '{quantile}' not found in forecast data for Currency={currency}, Branch={branch}")

            self.logger.info("Inverse scaling completed successfully.")
            return forecast_df

        except Exception as e:
            self.logger.error(f"Error in _get_forecast_result: {e}")
            raise
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    self.logger.info(f"Removed temporary directory {temp_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp directory {temp_dir}: {e}")

    def _generate_statistical_report(self, result_df: pd.DataFrame, country_code: str) -> None:
        """Generate detailed statistical report for forecasts."""
        report_file = os.path.join(self.output_dir, f"{country_code}_forecast_stats.json")
        try:
            stats = {
                'global_stats': {
                    'total_forecasts': len(result_df),
                    'timestamp': self.timestamp,
                    'scaling_method': self.scaling_metadata['scaling_method'],
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
                    'scaling_params': params,
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
                    zscore = np.abs((values - params['mean']) / (params['std'] if params['std'] != 0 else 1))
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
        output_path = f"./results/{country_code}_{self.timestamp}"

        try:
            # Create output directory first
            os.makedirs(output_path, exist_ok=True)

            # Required columns to validate
            required_columns = (
                    ['ProductId', 'BranchId', 'Currency', 'EffectiveDate', 'ForecastDate'] +
                    self.config.quantiles
            )

            # Initial DataFrame validation
            missing_columns = set(required_columns) - set(forecasts_df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in forecasts: {missing_columns}")

            # Create working copy to avoid modifying original
            working_df = forecasts_df.copy()

            # Convert dates
            try:
                for date_col in ['EffectiveDate', 'ForecastDate']:
                    if not pd.api.types.is_datetime64_any_dtype(working_df[date_col]):
                        working_df[date_col] = pd.to_datetime(working_df[date_col])
            except Exception as e:
                raise ValueError(f"Error converting date columns: {e}")

            # Calculate ForecastDay
            working_df['ForecastDay'] = (
                    (working_df['ForecastDate'] - working_df['EffectiveDate']).dt.days + 1
            )

            # Filter forecasts within horizon
            valid_forecasts = working_df[
                (working_df['ForecastDay'] >= 1) &
                (working_df['ForecastDay'] <= self.config.forecast_horizon)
                ]

            if len(valid_forecasts) < len(working_df):
                self.logger.warning(
                    f"Filtered out {len(working_df) - len(valid_forecasts)} "
                    f"forecasts outside horizon range"
                )

            if len(valid_forecasts) == 0:
                raise ValueError("No valid forecasts within specified horizon")

            # Ensure quantiles are numeric and handle any conversion issues
            numeric_conversion_issues = []
            for quantile in self.config.quantiles:
                try:
                    valid_forecasts[quantile] = pd.to_numeric(
                        valid_forecasts[quantile],
                        errors='coerce'
                    )
                    # Check for NaN values after conversion
                    nan_count = valid_forecasts[quantile].isna().sum()
                    if nan_count > 0:
                        numeric_conversion_issues.append(
                            f"{quantile}: {nan_count} non-numeric values"
                        )
                except Exception as e:
                    raise ValueError(f"Error converting {quantile} to numeric: {e}")

            if numeric_conversion_issues:
                self.logger.warning(
                    "Numeric conversion issues detected:\n" +
                    "\n".join(numeric_conversion_issues)
                )

            # Pivot the forecasts for final output format
            try:
                forecasts_pivot = valid_forecasts.pivot_table(
                    index=['ProductId', 'BranchId', 'Currency', 'EffectiveDate'],
                    columns='ForecastDay',
                    values=self.config.quantiles,
                    aggfunc='first'  # Use first value if there are duplicates
                )

                # Rename columns to include quantile and day information
                forecasts_pivot.columns = [
                    f"{quantile}_Day{int(day)}"
                    for quantile, day in forecasts_pivot.columns
                ]
                self.logger.info(f"Renamed forecast columns to include quantile and day information.")

                # Reset index for flat structure
                forecasts_pivot.reset_index(inplace=True)

                # Validate pivoted data
                if len(forecasts_pivot) == 0:
                    raise ValueError("Pivot operation resulted in empty DataFrame")

                expected_cols = len(self.config.quantiles) * self.config.forecast_horizon
                actual_cols = len([col for col in forecasts_pivot.columns
                                   if any(q in col for q in self.config.quantiles)])
                if actual_cols != expected_cols:
                    self.logger.warning(
                        f"Unexpected number of forecast columns: {actual_cols} "
                        f"(expected {expected_cols})"
                    )

                # Save files
                detailed_file = os.path.join(output_path, "detailed_forecast.csv")
                final_file = os.path.join(output_path, "final_forecast.csv")

                valid_forecasts.to_csv(detailed_file, index=False)
                forecasts_pivot.to_csv(final_file, index=False)

                # Validate saved files
                for file_path in [detailed_file, final_file]:
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"Failed to save forecast file: {file_path}")
                    if os.path.getsize(file_path) == 0:
                        raise ValueError(f"Generated empty forecast file: {file_path}")

                # Create and save metadata
                metadata = {
                    'timestamp': self.timestamp,
                    'country_code': country_code,
                    'forecast_horizon': self.config.forecast_horizon,
                    'total_forecasts': len(valid_forecasts),
                    'unique_combinations': len(forecasts_pivot),
                    'quantiles': self.config.quantiles,
                    'scaling_method': self.scaling_metadata['scaling_method'],
                    'files': {
                        'detailed': os.path.basename(detailed_file),
                        'final': os.path.basename(final_file)
                    },
                    'validation': {
                        'original_rows': len(forecasts_df),
                        'valid_rows': len(valid_forecasts),
                        'numeric_issues': len(numeric_conversion_issues) > 0,
                        'column_count_match': actual_cols == expected_cols
                    }
                }

                metadata_file = os.path.join(output_path, "forecast_metadata.json")
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                self.logger.info(f"Saved detailed forecasts to: {detailed_file}")
                self.logger.info(f"Saved final forecasts to: {final_file}")
                self.logger.info(f"Saved forecast metadata to: {metadata_file}")

                return metadata

            except Exception as e:
                self.logger.error(f"Error in pivot and save operations: {e}")
                raise

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


def setup_logging(timestamp: str) -> logging.Logger:
    """Setup logging for the main process."""
    logger = logging.getLogger('CashForecastMain')
    logger.setLevel(logging.INFO)

    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(f'{log_dir}/cash_forecast_main_{timestamp}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Stream handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


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
    logger = setup_logging(timestamp)
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
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Validate configuration
        try:
            config_model = ConfigModel(**config_dict)
            config = Config(**config_model.dict())
            logger.info("Configuration validated successfully")
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected configuration error: {e}")
            sys.exit(1)

        # Process each country
        for country_code in args.countries:
            logger.info(f"\nProcessing country: {country_code}")
            try:
                # Initialize pipeline
                inference_pipeline = CashForecastingInference(config)

                # Load scaling parameters
                logger.info(f"Loading scaling parameters for {country_code}")
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
                logger.error("Traceback:", exc_info=True)
                continue  # Continue with next country even if one fails

        logger.info("\nInference pipeline completed")

    except Exception as e:
        logger.error(f"Critical error in main process: {e}")
        logger.error("Traceback:", exc_info=True)
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
