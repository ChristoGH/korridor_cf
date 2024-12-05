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
from time import gmtime, strftime
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel
import yaml
from pathlib import Path


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


class DataProcessor:
    """Handles data preprocessing and scaling operations."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.required_columns = ['ProductId', 'BranchId', 'Currency', 'EffectiveDate', 'Demand']

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
            raise ValueError(f"Input data is missing required columns: {missing_cols}")

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

        # Calculate scaling parameters per Currency-BranchId combination
        scaling_params = self._calculate_scaling_params(data)

        # Apply scaling
        scaled_data = self._apply_scaling(data, scaling_params)

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

    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    fh = logging.FileHandler(f'cash_forecast_{timestamp}.log')
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


def load_config(config_path: str) -> Config:
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
        return Config(**config_model.dict())
    except Exception as e:
        raise Exception(f"Failed to load configuration: {str(e)}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

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

    return parser.parse_args()