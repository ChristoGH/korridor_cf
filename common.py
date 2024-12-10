"""
Common utilities and shared functionality for cash forecasting pipeline.

This module contains shared components for both model training and inference,
including configuration management, data processing, logging, and AWS interactions.
"""

import argparse
import boto3
import pandas as pd
import numpy as np
import os
import logging
import sys
import json
from time import gmtime, strftime, sleep
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, field_validator, ValidationError
import yaml
from pathlib import Path
import shutil


class MonitoringDatasetFormat(BaseModel):
    """Pydantic model for monitoring dataset format."""
    json_format: str  # Renamed from 'json' to 'json_format' to avoid shadowing
    # Add other fields as necessary

    @field_validator('json_format')
    def json_format_must_be_valid(cls, v):
        if not v.startswith('{') or not v.endswith('}'):
            raise ValueError('json_format must be a valid JSON string')
        return v


class ConfigModel(BaseModel):
    """Pydantic model for validating configuration parameters."""
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


class S3Handler:
    """Handles S3 operations with proper error handling and logging."""

    def __init__(self, region: str, logger: logging.Logger):
        self.client = boto3.client('s3', region_name=region)
        self.logger = logger

    def safe_upload(self,
                    local_path: str | Path,
                    bucket: str,
                    s3_key: str,
                    overwrite: bool = False) -> None:
        """
        Safely upload file to S3 with existence check.

        Args:
            local_path: Local file path
            bucket: S3 bucket name
            s3_key: S3 object key
            overwrite: Whether to overwrite existing file

        Raises:
            FileExistsError: If file exists and overwrite=False
            Exception: For other upload errors
        """
        try:
            if not overwrite:
                existing = self.client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=s3_key
                )
                if 'Contents' in existing:
                    raise FileExistsError(f"S3 object already exists: {s3_key}")

            self.client.upload_file(
                Filename=str(local_path),
                Bucket=bucket,
                Key=s3_key
            )
            self.logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
        except Exception as e:
            self.logger.error(f"Failed to upload {local_path} to S3: {e}")
            raise

    def list_s3_objects(self, bucket: str, prefix: str) -> List[Dict[str, Any]]:
        """
        List objects in an S3 bucket under a specific prefix.

        Args:
            bucket: S3 bucket name
            prefix: S3 prefix

        Returns:
            List of S3 object metadata dictionaries
        """
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
            objects = []
            for page in page_iterator:
                if 'Contents' in page:
                    objects.extend(page['Contents'])
            self.logger.info(f"Listed {len(objects)} objects in s3://{bucket}/{prefix}")
            return objects
        except Exception as e:
            self.logger.error(f"Failed to list objects in s3://{bucket}/{prefix}: {e}")
            raise

    def download_file(self, bucket: str, s3_key: str, local_path: str | Path) -> None:
        """
        Download a file from S3 to a local path.

        Args:
            bucket: S3 bucket name
            s3_key: S3 object key
            local_path: Local file path to save the downloaded file

        Raises:
            Exception: If download fails
        """
        try:
            self.client.download_file(bucket, s3_key, str(local_path))
            self.logger.info(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")
        except Exception as e:
            self.logger.error(f"Failed to download s3://{bucket}/{s3_key}: {e}")
            raise


class DataProcessor:
    """Handles data preprocessing and scaling operations."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.required_columns = ['ProductId', 'BranchId', 'Currency', 'EffectiveDate', 'Demand']

    def load_data(self, file_path: str | Path) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file does not exist
            pd.errors.EmptyDataError: If file is empty
            Exception: For other loading errors
        """
        try:
            self.logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            self.logger.info(f"Loaded data with shape {data.shape}")
            return data
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error(f"No data: {file_path} is empty")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            raise

    def validate_input_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data structure and required columns.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            self.logger.error(f"Input data is missing required columns: {missing_cols}")
            raise ValueError(f"Input data is missing required columns: {missing_cols}")
        self.logger.info("Input data validation passed.")

    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, float]]]:
        """
        Prepare and scale data for forecasting.

        Args:
            data: Input DataFrame

        Returns:
            Tuple containing scaled DataFrame and scaling parameters
        """
        self.validate_input_data(data)

        # Use only required columns and ensure correct types
        data = data[self.required_columns].copy()
        data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate']).dt.tz_localize(None)
        data['ProductId'] = data['ProductId'].astype(str)
        data['BranchId'] = data['BranchId'].astype(str)
        data['Currency'] = data['Currency'].astype(str)
        data.sort_values('EffectiveDate', inplace=True)
        self.logger.info("Data types enforced and sorted by EffectiveDate.")

        # Calculate scaling parameters per Currency-BranchId combination
        scaling_params = self._calculate_scaling_params(data)
        self.logger.info(f"Calculated scaling parameters for {len(scaling_params)} groups.")

        # Apply scaling
        scaled_data = self._apply_scaling(data, scaling_params)
        self.logger.info("Applied scaling to Demand values.")

        return scaled_data, scaling_params

    def _calculate_scaling_params(self, data: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Calculate scaling parameters for each Currency-BranchId group."""
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
        return scaling_params

    def _apply_scaling(self,
                       data: pd.DataFrame,
                       scaling_params: Dict[Tuple[str, str], Dict[str, float]]) -> pd.DataFrame:
        """Apply scaling to data using provided parameters."""
        scaled_data = data.copy()
        for (currency, branch), params in scaling_params.items():
            mask = (data['Currency'] == currency) & (data['BranchId'] == branch)
            scaled_data.loc[mask, 'Demand'] = (
                (data.loc[mask, 'Demand'] - params['mean']) /
                (params['std'] if params['std'] != 0 else 1)
            )
        return scaled_data

    def load_scaling_metadata(self, scaling_metadata_file: Path) -> Dict[str, Any]:
        """
        Load scaling metadata from a JSON file.
        """
        try:
            with open(scaling_metadata_file, 'r') as f:
                metadata = json.load(f)
            self.logger.info(f"Loaded scaling metadata from {scaling_metadata_file}")
            return metadata
        except FileNotFoundError:
            self.logger.error(f"Scaling metadata file not found: {scaling_metadata_file}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load scaling metadata from {scaling_metadata_file}: {e}")
            raise

    def save_scaling_params(self, scaling_params: Dict[Tuple[str, str], Dict[str, float]], scaling_file: Path) -> None:
        """
        Save scaling parameters to a JSON file.

        Args:
            scaling_params: Scaling parameters dictionary
            scaling_file: Path to save the JSON file
        """
        try:
            with open(scaling_file, 'w') as f:
                # Convert scaling_params keys to strings for JSON serialization
                serializable_params = {
                    f"{currency}_{branch}": params
                    for (currency, branch), params in scaling_params.items()
                }
                json.dump(serializable_params, f, indent=2)
            self.logger.info(f"Saved scaling parameters to {scaling_file}")
        except Exception as e:
            self.logger.error(f"Failed to save scaling parameters to {scaling_file}: {e}")
            raise

    def validate_scaling_parameters(self, scaling_params: Dict[Tuple[str, str], Dict[str, float]]) -> None:
        """
        Validate the structure and content of scaling parameters.

        Args:
            scaling_params (dict): Scaling parameters to validate.

        Raises:
            ValueError: If validation fails.
        """
        required_keys = {'mean', 'std', 'min', 'max', 'last_value', 'n_observations'}
        for group, params in scaling_params.items():
            if not required_keys.issubset(params.keys()):
                missing = required_keys - set(params.keys())
                raise ValueError(f"Scaling parameters for group {group} are missing keys: {missing}")

    def load_scaling_params(self, scaling_file: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Load scaling parameters from a JSON file.

        Args:
            scaling_file: Path to the JSON file

        Returns:
            Scaling parameters dictionary

        Raises:
            FileNotFoundError: If the scaling file does not exist
            Exception: For other loading errors
        """
        try:
            with open(scaling_file, 'r') as f:
                serialized_params = json.load(f)
            scaling_params = {
                tuple(k.split('_')): v
                for k, v in serialized_params.items()
            }
            self.logger.info(f"Loaded scaling parameters from {scaling_file}")
            return scaling_params
        except FileNotFoundError:
            self.logger.error(f"Scaling parameters file not found: {scaling_file}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load scaling parameters from {scaling_file}: {e}")
            raise

    def restore_scaling(self,
                        forecast_df: pd.DataFrame,
                        scaling_params: Dict[Tuple[str, str], Dict[str, float]],
                        quantiles: List[str],
                        logger: logging.Logger) -> pd.DataFrame:
        """
        Restore original scale for forecasted Demand values.

        Args:
            forecast_df: DataFrame containing forecasted values
            scaling_params: Scaling parameters dictionary
            quantiles: List of quantile names
            logger: Logger instance

        Returns:
            DataFrame with restored Demand values
        """
        logger.info("Restoring original scale to forecasts...")
        result_df = forecast_df.copy()

        # Iterate over each group and restore scaling
        for (currency, branch), params in scaling_params.items():
            mask = (result_df['Currency'] == currency) & (result_df['BranchId'] == branch)
            if not mask.any():
                logger.warning(f"No forecasts found for Currency={currency}, Branch={branch}")
                continue

            # Inverse transform the forecasts for each quantile
            for quantile in quantiles:
                if quantile in result_df.columns:
                    result_df.loc[mask, quantile] = (
                        (result_df.loc[mask, quantile] *
                         (params['std'] if params['std'] != 0 else 1)) +
                        params['mean']
                    )
                    logger.info(
                        f"Inverse scaled {quantile} for Currency={currency}, Branch={branch}"
                    )
                else:
                    logger.warning(
                        f"Quantile '{quantile}' not found in forecast data for Currency={currency}, Branch={branch}"
                    )

            # Validate the scaling restoration
            for quantile in quantiles:
                if quantile not in result_df.columns:
                    continue
                scaled_stats = forecast_df.loc[mask, quantile].describe()
                restored_stats = result_df.loc[mask, quantile].describe()

                logger.info(
                    f"Scaling restoration for {currency}-{branch} {quantile}:\n"
                    f"  Scaled   -> mean: {scaled_stats['mean']:.2f}, std: {scaled_stats['std']:.2f}\n"
                    f"  Restored -> mean: {restored_stats['mean']:.2f}, std: {restored_stats['std']:.2f}\n"
                    f"  Expected -> mean: {params['mean']:.2f}, std: {params['std']:.2f}"
                )

                # Check for anomalies
                if params['std'] != 0:
                    zscore = np.abs((result_df.loc[mask, quantile] - params['mean']) / params['std'])
                    outliers = (zscore > 5).sum()
                    if outliers > 0:
                        logger.warning(
                            f"{outliers} extreme outliers in restored forecasts for "
                            f"{currency}-{branch} {quantile} (>5 std from mean)"
                        )

        # Add scaling metadata to result
        result_df.attrs['scaling_restored'] = True
        result_df.attrs['scaling_timestamp'] = strftime("%Y%m%d-%H%M%S", gmtime())

        return result_df

    def generate_scaling_metadata(self, data: pd.DataFrame,
                                  scaling_params: Dict[Tuple[str, str], Dict[str, float]]) -> Dict[str, Any]:
        """
        Generate metadata about the scaling process.

        Args:
            data: Original input DataFrame
            scaling_params: Scaling parameters dictionary

        Returns:
            Metadata dictionary
        """
        metadata = {
            'scaling_method': 'standardization',
            'scaling_level': 'currency_branch',
            'scaled_column': 'Demand',
            'timestamp': strftime("%Y%m%d-%H%M%S", gmtime()),
            'scaling_stats': {
                'global_mean': data['Demand'].mean(),
                'global_std': data['Demand'].std(),
                'global_min': data['Demand'].min(),
                'global_max': data['Demand'].max(),
                'n_groups': len(scaling_params),
                'n_total_observations': len(data)
            }
        }
        self.logger.info("Generated scaling metadata.")
        return metadata

    def save_metadata(self, metadata: Dict[str, Any], metadata_file: Path) -> None:
        """
        Save scaling metadata to a JSON file.

        Args:
            metadata: Metadata dictionary
            metadata_file: Path to save the JSON file
        """
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Saved scaling metadata to {metadata_file}")
        except Exception as e:
            self.logger.error(f"Failed to save scaling metadata to {metadata_file}: {e}")
            raise

    def get_effective_dates(self, train_file: str | Path, backtesting: bool) -> List[pd.Timestamp]:
        """
        Retrieve effective dates from training data for forecasting.

        Args:
            train_file: Path to the scaled training CSV file
            backtesting: Whether to use past dates for backtesting

        Returns:
            List of effective dates
        """
        data = self.load_data(train_file)
        data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate']).dt.tz_localize(None)
        if backtesting:
            effective_dates = sorted(data['EffectiveDate'].unique())
            self.logger.info(f"Using {len(effective_dates)} effective dates for backtesting.")
        else:
            effective_dates = [data['EffectiveDate'].max()]
            self.logger.info(f"Using the most recent effective date: {effective_dates[0]}")
        return effective_dates

    def chunk_list(self, data_list: List[Any], chunk_size: int) -> List[List[Any]]:
        """
        Split a list into smaller chunks.

        Args:
            data_list: The list to split
            chunk_size: Size of each chunk

        Returns:
            List of chunks
        """
        return [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

    def generate_inference_data(self, inference_template: pd.DataFrame,
                                batch_dates: List[pd.Timestamp],
                                forecast_horizon: int) -> pd.DataFrame:
        """
        Generate inference data for a batch of dates.

        Args:
            inference_template: DataFrame with unique ProductId, BranchId, Currency
            batch_dates: List of effective dates
            forecast_horizon: Number of days to forecast

        Returns:
            Inference DataFrame
        """
        inference_data_list = []
        num_combinations = len(inference_template)

        for effective_date in batch_dates:
            # Generate future dates for the forecast horizon
            future_dates = [effective_date + pd.Timedelta(days=j) for j in range(1, forecast_horizon + 1)]

            # Create inference data for each combination and future dates
            temp_df = pd.DataFrame({
                'ProductId': np.tile(inference_template['ProductId'].values, forecast_horizon),
                'BranchId': np.tile(inference_template['BranchId'].values, forecast_horizon),
                'Currency': np.tile(inference_template['Currency'].values, forecast_horizon),
                'EffectiveDate': effective_date,
                'ForecastDate': np.repeat(future_dates, num_combinations),
                'Demand': np.nan  # Demand is unknown for future dates
            })
            inference_data_list.append(temp_df)

        # Combine batch inference data
        inference_data = pd.concat(inference_data_list, ignore_index=True).drop(columns=['Demand'])
        self.logger.info(f"Generated inference data with shape {inference_data.shape}")
        return inference_data

    def retrieve_forecast_results(self, s3_handler: S3Handler, bucket: str, prefix: str, output_dir: Path, quantiles: List[str]) -> pd.DataFrame:
        """
        Download and combine forecast result files from S3.

        Args:
            s3_handler: Instance of S3Handler
            bucket: S3 bucket name
            prefix: S3 prefix where forecast results are stored
            output_dir: Local directory to download files
            quantiles: List of quantile names

        Returns:
            Combined forecast DataFrame

        Raises:
            FileNotFoundError: If no forecast files are found
            Exception: For other download or processing errors
        """
        try:
            objects = s3_handler.list_s3_objects(bucket=bucket, prefix=prefix)
            forecast_keys = [obj['Key'] for obj in objects if obj['Key'].endswith('.out')]

            if not forecast_keys:
                self.logger.error(f"No forecast output files found in s3://{bucket}/{prefix}")
                raise FileNotFoundError(f"No forecast output files found in s3://{bucket}/{prefix}")

            # Create a temporary directory for downloads
            temp_dir = output_dir / "temp_forecast"
            temp_dir.mkdir(parents=True, exist_ok=True)

            forecast_data = []
            for key in forecast_keys:
                local_file = temp_dir / Path(key).name
                s3_handler.download_file(bucket=bucket, s3_key=key, local_path=local_file)
                self.logger.info(f"Downloaded forecast file {key} to {local_file}")

                # Read forecast data (assuming CSV format without headers)
                try:
                    df = pd.read_csv(local_file, header=None)
                    num_columns = len(df.columns)
                    self.logger.info(f"Forecast file {key} has {num_columns} columns.")

                    # Assign quantile columns based on config.quantiles
                    if num_columns != len(quantiles):
                        self.logger.warning(f"Expected {len(quantiles)} quantiles, but got {num_columns} in file {key}")
                        # Adjust column names accordingly
                        df.columns = [f"Quantile_{i+1}" for i in range(num_columns)]
                    else:
                        df.columns = quantiles

                    forecast_data.append(df)
                    self.logger.info(f"Processed forecast file {key} with shape {df.shape}")
                except Exception as e:
                    self.logger.error(f"Failed to process forecast file {key}: {e}")
                    continue
                finally:
                    # Remove the local file after processing
                    local_file.unlink(missing_ok=True)

            if not forecast_data:
                self.logger.error("No valid forecast data found.")
                raise FileNotFoundError("No valid forecast data found.")

            # Combine all forecast data
            forecast_df = pd.concat(forecast_data, ignore_index=True)
            self.logger.info(f"Combined forecast data shape: {forecast_df.shape}")

            # Cleanup temporary directory
            shutil.rmtree(temp_dir)
            self.logger.info(f"Cleaned up temporary forecast directory {temp_dir}")

            return forecast_df

        except Exception as e:
            self.logger.error(f"Failed to retrieve forecast results: {e}")
            raise

    def save_final_forecasts(self,
                             forecasts_df: pd.DataFrame,
                             country_code: str,
                             timestamp: str,
                             config: Config,
                             logger: logging.Logger) -> None:
        """
        Save the final forecasts in the desired format.

        Args:
            forecasts_df: DataFrame containing the final forecasts
            country_code: Country code identifier
            timestamp: Timestamp string
            config: Configuration object
            logger: Logger instance

        Raises:
            ValueError: If required columns are missing or data types are incorrect
        """
        try:
            required_columns = [
                'ProductId', 'BranchId', 'Currency',
                'EffectiveDate', 'ForecastDate'
            ] + config.quantiles

            logger.info(f"Forecast df columns: {forecasts_df.columns.tolist()}")

            for col in required_columns:
                if col not in forecasts_df.columns:
                    logger.error(f"Column '{col}' is missing from forecasts DataFrame.")
                    raise ValueError(f"Column '{col}' is missing from forecasts DataFrame.")

            # Convert date columns to datetime
            for date_col in ['EffectiveDate', 'ForecastDate']:
                forecasts_df[date_col] = pd.to_datetime(forecasts_df[date_col]).dt.tz_localize(None)

            # Ensure quantile columns are numeric
            try:
                forecasts_df[config.quantiles] = forecasts_df[config.quantiles].apply(pd.to_numeric, errors='coerce')
                logger.info("Converted quantile columns to numeric.")
            except Exception as e:
                logger.error(f"Failed to convert quantile columns to numeric: {e}")
                raise ValueError("Quantile columns contain non-numeric values.")

            # Calculate 'ForecastDay'
            forecasts_df['ForecastDay'] = (forecasts_df['ForecastDate'] - forecasts_df['EffectiveDate']).dt.days + 1

            # Filter forecasts within horizon
            forecasts_df = forecasts_df[
                (forecasts_df['ForecastDay'] >= 1) & (forecasts_df['ForecastDay'] <= config.forecast_horizon)
            ]
            logger.info(f"Filtered forecasts within horizon, resulting shape: {forecasts_df.shape}")

            # Pivot the data
            try:
                forecasts_pivot = forecasts_df.pivot_table(
                    index=['ProductId', 'BranchId', 'Currency', 'EffectiveDate'],
                    columns='ForecastDay',
                    values=config.quantiles,
                    aggfunc='mean'  # Explicitly specify the aggregation function
                )
                logger.info("Pivoted forecast data successfully.")
            except Exception as e:
                logger.error(f"Pivot table aggregation failed: {e}")
                raise ValueError("Aggregation function failed due to non-numeric quantile columns.")

            # Rename columns to include quantile and day information
            forecasts_pivot.columns = [f"{quantile}_Day{int(day)}"
                                       for quantile, day in forecasts_pivot.columns]
            logger.info("Renamed forecast columns with quantile and day information.")

            # Reset index
            forecasts_pivot.reset_index(inplace=True)

            # Save results
            output_file = Path(f"./results/{country_code}_{timestamp}/final_forecast.csv")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            forecasts_pivot.to_csv(output_file, index=False)
            logger.info(f"Final forecast saved to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save final forecasts: {e}")
            raise


def setup_logging(timestamp: str, name: str = 'CashForecast') -> logging.Logger:
    """
    Configure logging with file and console handlers.

    Args:
        timestamp: Timestamp for log file name
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if logger already has them
    if not logger.handlers:
        # Create formatters and handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        log_dir = Path('./logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f'cash_forecast_{timestamp}.log')
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


def load_config(config_path: str | Path) -> Config:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated Config instance

    Raises:
        Exception: If configuration is invalid
    """
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Validate with Pydantic
        config_model = ConfigModel(**config_dict)

        # Create Config instance
        config = Config(**config_model.dict())
        return config
    except ValidationError as ve:
        raise Exception(f"Configuration validation error: {ve}")
    except Exception as e:
        raise Exception(f"Failed to load configuration: {str(e)}")


def parse_arguments(inference: bool = False) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        inference (bool): Flag indicating if inference arguments should be included.

    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(description='Cash Forecasting Pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--countries', nargs='+', default=['ZM'],
                        help='Country codes to process')
    parser.add_argument('--resume', action='store_true',
                        help='Resume pipeline from last checkpoint')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to the input CSV file (overrides default)')

    if inference:
        parser.add_argument('--model_timestamp', type=str, required=True,
                            help='Timestamp of the training run (YYYYMMDD-HHMMSS)')
        parser.add_argument('--effective_date', type=str, required=True,
                            help='Effective date for forecasts (YYYY-MM-DD)')
        parser.add_argument('--model_name', type=str, required=True,
                            help='Name of the SageMaker model to use for inference')
        parser.add_argument('--inference_template', type=str, required=True,
                            help='Path to the inference template CSV file')

    return parser.parse_args()
