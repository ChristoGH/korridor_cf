# common.py
from botocore.exceptions import ClientError
import os
import sys
import logging
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from time import strftime, gmtime

import boto3
import yaml
import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError

# Configuration Models

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


# Logging Setup

def setup_logging(logger_name: str, timestamp: str, log_dir: str = './logs') -> logging.Logger:
    """Setup logging for the pipeline."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    fh = logging.FileHandler(f'{log_dir}/{logger_name.lower()}_{timestamp}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Stream handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # Avoid adding multiple handlers if they already exist
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# Utility Functions

def load_and_validate_config(config_path: str) -> Config:
    """Load and validate configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    try:
        config_model = ConfigModel(**config_dict)
        config = Config(**config_model.dict())
    except ValidationError as e:
        # Initialize a temporary logger for configuration errors
        temp_logger = logging.getLogger('ConfigValidation')
        temp_logger.setLevel(logging.ERROR)
        temp_logger.addHandler(logging.StreamHandler(sys.stdout))
        temp_logger.error(f"Configuration validation error: {e}")
        sys.exit(1)
    except Exception as e:
        temp_logger = logging.getLogger('ConfigValidation')
        temp_logger.setLevel(logging.ERROR)
        temp_logger.addHandler(logging.StreamHandler(sys.stdout))
        temp_logger.error(f"Unexpected configuration error: {e}")
        sys.exit(1)

    return config

# common.py

def safe_s3_upload(s3_client, logger, local_path, s3_key, bucket, overwrite=False):
    """Safely upload a file to S3, checking for existence if overwrite is False."""
    if not overwrite:
        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
            logger.warning(f"S3 key {s3_key} already exists in bucket {bucket}. Skipping upload.")
            return
        except ClientError as e:
            if e.response['Error']['Code'] != '404':
                logger.error(f"Error checking existence of s3://{bucket}/{s3_key}: {e}")
                raise
    try:
        s3_client.upload_file(local_path, bucket, s3_key)
        logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
    except ClientError as e:
        logger.error(f"Failed to upload {local_path} to s3://{bucket}/{s3_key}: {e}")
        raise

# common.py

def load_scaling_parameters(s3_client, logger, bucket, scaling_params_key):
    """Load scaling parameters from S3."""
    try:
        scaling_params_obj = s3_client.get_object(Bucket=bucket, Key=scaling_params_key)
        scaling_params_content = scaling_params_obj['Body'].read().decode('utf-8')
        scaling_params = json.loads(scaling_params_content)
        # Convert string keys back to tuple keys if necessary
        scaling_params = {tuple(k.split('_')): v for k, v in scaling_params.items()}
        return scaling_params
    except ClientError as e:
        logger.error(f"Failed to get scaling parameters from s3://{bucket}/{scaling_params_key}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in scaling parameters file: {e}")
        raise
