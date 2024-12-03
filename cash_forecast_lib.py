# cash_forecast_lib.py

import os
import json
import logging
import boto3
import pandas as pd
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
from time import strftime, gmtime, sleep
import numpy as np
import shutil

from common import Config, load_and_validate_config, setup_logging, safe_s3_upload, load_scaling_parameters


@dataclass
class ScalingParameters:
    """Data class to hold scaling parameters."""
    mean: float
    std: float
    min: float
    max: float
    last_value: float
    n_observations: int


def setup_custom_logging(name: str, timestamp: str) -> logging.Logger:
    """Setup custom logging."""
    return setup_logging(name, timestamp)


def upload_file_to_s3(
        local_path: str,
        s3_key: str,
        s3_client,
        bucket: str,
        logger: logging.Logger,
        overwrite: bool = False
) -> None:
    """Upload a file to S3 safely."""
    safe_s3_upload(
        s3_client=s3_client,
        logger=logger,
        local_path=local_path,
        s3_key=s3_key,
        bucket=bucket,
        overwrite=overwrite
    )


def download_file_from_s3(
        s3_client,
        bucket: str,
        s3_key: str,
        local_path: str,
        logger: logging.Logger
) -> None:
    """Download a file from S3."""
    try:
        s3_client.download_file(bucket, s3_key, local_path)
        logger.info(f"Downloaded {s3_key} to {local_path}")
    except Exception as e:
        logger.error(f"Failed to download {s3_key} from S3: {e}")
        raise


def load_scaling_params(
        s3_client,
        bucket: str,
        scaling_params_key: str,
        logger: logging.Logger
) -> Dict[Tuple[str, str], ScalingParameters]:
    """Load and parse scaling parameters from S3."""
    scaling_params_raw = load_scaling_parameters(
        s3_client=s3_client,
        logger=logger,
        bucket=bucket,
        scaling_params_key=scaling_params_key
    )

    # Convert to ScalingParameters dataclass
    scaling_params = {
        tuple(key.split('_')): ScalingParameters(**params)
        for key, params in scaling_params_raw.items()
    }
    return scaling_params


def save_scaling_params(
        scaling_params: Dict[Tuple[str, str], ScalingParameters],
        filepath: str
) -> None:
    """Save scaling parameters to a JSON file."""
    serializable_params = {
        f"{currency}_{branch}": params.__dict__
        for (currency, branch), params in scaling_params.items()
    }
    with open(filepath, 'w') as f:
        json.dump(serializable_params, f, indent=2)


def load_scaling_metadata(
        s3_client,
        bucket: str,
        scaling_metadata_key: str,
        logger: logging.Logger
) -> Dict[str, Any]:
    """Load scaling metadata from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=scaling_metadata_key)
        scaling_metadata = json.loads(response['Body'].read().decode('utf-8'))
        return scaling_metadata
    except Exception as e:
        logger.error(f"Failed to load scaling metadata from S3: {e}")
        raise


def prepare_training_data(
        input_file: str,
        required_columns: List[str],
        categorical_columns: List[str],
        logger: logging.Logger
) -> pd.DataFrame:
    """Load and preprocess training data."""
    data = pd.read_csv(input_file)
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    # Check for missing values in required columns
    if data[required_columns].isnull().any().any():
        missing = data[required_columns].isnull().sum()
        raise ValueError(f"Input data contains missing values: {missing[missing > 0]}")

    # Use only the required columns
    data = data[required_columns]

    # Ensure correct data types
    data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate'], errors='coerce').dt.tz_localize(None)
    if data['EffectiveDate'].isnull().any():
        raise ValueError("Some 'EffectiveDate' entries could not be parsed as datetime.")

    for col in categorical_columns:
        data[col] = data[col].astype(str)

    data.sort_values('EffectiveDate', inplace=True)
    logger.info("Data loaded and sorted by EffectiveDate.")

    return data


def calculate_scaling_parameters(
        data: pd.DataFrame,
        group_columns: List[str],
        target_column: str,
        logger: logging.Logger
) -> Dict[Tuple[str, str], ScalingParameters]:
    """Calculate scaling parameters per group."""
    scaling_params = {}
    for group, group_data in data.groupby(group_columns):
        scaling_params[group] = ScalingParameters(
            mean=group_data[target_column].mean(),
            std=group_data[target_column].std(),
            min=group_data[target_column].min(),
            max=group_data[target_column].max(),
            last_value=group_data[target_column].iloc[-1],
            n_observations=len(group_data)
        )
        logger.debug(f"Calculated scaling for group {group}.")
    return scaling_params


def apply_scaling(
        data: pd.DataFrame,
        scaling_params: Dict[Tuple[str, str], ScalingParameters],
        group_columns: List[str],
        target_column: str,
        logger: logging.Logger
) -> pd.DataFrame:
    """Apply scaling to the target column based on scaling parameters."""
    scaled_data = data.copy()
    for group, params in scaling_params.items():
        mask = np.ones(len(data), dtype=bool)
        for col, val in zip(group_columns, group):
            mask &= data[col] == val
        scaled_data.loc[mask, target_column] = (
                (data.loc[mask, target_column] - params.mean) / (params.std if params.std != 0 else 1)
        )
        logger.debug(f"Applied scaling for group {group}.")
    return scaled_data


def create_output_directory(base_path: str, country_code: str, timestamp: str) -> str:
    """Create and return the output directory path."""
    output_dir = os.path.join(base_path, f"{country_code}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_dataframe_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """Save a DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)


def load_inference_template(inference_template_file: str, logger: logging.Logger) -> pd.DataFrame:
    """Load and validate the inference template."""
    inference_template = pd.read_csv(inference_template_file)
    required_columns = {'ProductId', 'BranchId', 'Currency'}
    if not required_columns.issubset(inference_template.columns):
        missing = required_columns - set(inference_template.columns)
        raise ValueError(f"Missing required columns in inference template: {missing}")

    for col in ['ProductId', 'BranchId', 'Currency']:
        inference_template[col] = inference_template[col].astype(str)

    logger.info("Inference template loaded and validated.")
    return inference_template


def generate_inference_data(
        inference_template: pd.DataFrame,
        effective_date: pd.Timestamp,
        forecast_horizon: int,
        logger: logging.Logger
) -> pd.DataFrame:
    """Generate inference data based on the template and forecast horizon."""
    future_dates = [effective_date + pd.Timedelta(days=i) for i in range(1, forecast_horizon + 1)]
    inference_data = pd.DataFrame({
        'ProductId': np.tile(inference_template['ProductId'].values, forecast_horizon),
        'BranchId': np.tile(inference_template['BranchId'].values, forecast_horizon),
        'Currency': np.tile(inference_template['Currency'].values, forecast_horizon),
        'EffectiveDate': effective_date,
        'ForecastDate': np.repeat(future_dates, len(inference_template)),
        'Demand': np.nan  # Demand is unknown for future dates
    })
    logger.info("Inference data generated based on the template and forecast horizon.")
    return inference_data


def validate_scaling_parameters(scaling_params: Dict[Tuple[str, str], ScalingParameters]) -> None:
    """Validate the structure and content of scaling parameters."""
    if not scaling_params:
        raise ValueError("Scaling parameters dictionary is empty")

    for group, params in scaling_params.items():
        currency, branch = group
        if not isinstance(currency, str) or not isinstance(branch, str):
            raise ValueError(f"Invalid group format: ({currency}, {branch})")

        if params.std < 0:
            raise ValueError(f"Negative standard deviation for group {group}")
        if params.max < params.min:
            raise ValueError(f"Max value less than min for group {group}")
        if params.n_observations <= 0:
            raise ValueError(f"Invalid number of observations for group {group}")


def monitor_auto_ml_job(
        sm_client,
        job_name: str,
        logger: logging.Logger,
        max_wait_time: int = 24 * 60 * 60  # 24 hours
) -> str:
    """Monitor the AutoML job until completion."""
    logger.info(f"Monitoring AutoML job: {job_name}")
    sleep_time = 60  # Start with 1 minute
    elapsed_time = 0
    while elapsed_time < max_wait_time:
        try:
            response = sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
            status = response['AutoMLJobStatus']
            logger.info(f"AutoML Job {job_name} Status: {status}")
            if status in ['Completed', 'Failed', 'Stopped']:
                if status != 'Completed':
                    failure_reason = response.get('FailureReason', 'No failure reason provided.')
                    logger.error(f"AutoML Job {job_name} failed: {failure_reason}")
                return status
            sleep(sleep_time)
            elapsed_time += sleep_time
            sleep_time = min(int(sleep_time * 1.5), 600)  # Exponential backoff up to 10 minutes
        except Exception as e:
            logger.error(f"Error monitoring AutoML job {job_name}: {e}")
            sleep_time = min(int(sleep_time * 1.5), 600)
            elapsed_time += sleep_time
            sleep(60)  # Wait before retrying
    raise TimeoutError(f"AutoML job {job_name} did not complete within {max_wait_time} seconds.")


def retrieve_best_model(
        sm_client,
        job_name: str,
        country_code: str,
        timestamp: str,
        role_arn: str,
        logger: logging.Logger
) -> str:
    """Retrieve the best model from the completed AutoML job."""
    logger.info(f"Retrieving best model for AutoML job: {job_name}")
    try:
        response = sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
        best_candidate = response.get('BestCandidate')
        if not best_candidate:
            raise ValueError(f"No BestCandidate found for AutoML Job {job_name}")
    except Exception as e:
        logger.error(f"Error retrieving BestCandidate from AutoML job {job_name}: {e}")
        raise

    model_name = f"{country_code}-model-{timestamp}"

    # Check if the model already exists to prevent overwriting
    try:
        existing_models = sm_client.list_models(NameContains=model_name)
        if existing_models.get('Models'):
            raise ValueError(
                f"A model with name {model_name} already exists. Choose a different timestamp or model name.")
    except Exception as e:
        logger.error(f"Error checking existing models: {e}")
        raise

    # Create the SageMaker model
    try:
        sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            Containers=best_candidate['InferenceContainers']
        )
        logger.info(f"Created SageMaker model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to create SageMaker model {model_name}: {e}")
        raise

    return model_name
