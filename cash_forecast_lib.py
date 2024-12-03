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

from common import (
    Config,
    load_and_validate_config,
    setup_logging,
    safe_s3_upload,
    load_scaling_parameters
)


@dataclass
class ScalingParameters:
    """Data class to hold scaling parameters."""
    mean: float
    std: float
    min: float
    max: float
    last_value: float
    n_observations: int


def save_scaling_metadata(
    scaling_metadata: Dict[str, Any],
    filepath: str,
    logger: logging.Logger
) -> None:
    """
    Save scaling metadata to a JSON file.

    Args:
        scaling_metadata (Dict[str, Any]): The scaling metadata to save.
        filepath (str): The local path where the JSON file will be saved.
        logger (logging.Logger): Logger for logging information.
    """
    try:
        with open(filepath, 'w') as meta_file:
            json.dump(scaling_metadata, meta_file, indent=2)
        logger.info(f"Scaling metadata saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save scaling metadata to {filepath}: {e}")
        raise


def load_scaling_metadata(
    s3_client: boto3.client,
    logger: logging.Logger,
    bucket: str,
    scaling_metadata_key: str
) -> Dict[str, Any]:
    """
    Load scaling metadata from S3.

    Args:
        s3_client (boto3.client): The S3 client.
        logger (logging.Logger): Logger for logging information.
        bucket (str): The S3 bucket name.
        scaling_metadata_key (str): The S3 key for the scaling metadata JSON file.

    Returns:
        Dict[str, Any]: The loaded scaling metadata.
    """
    try:
        logger.info(f"Downloading scaling metadata from s3://{bucket}/{scaling_metadata_key}")
        scaling_metadata_obj = s3_client.get_object(Bucket=bucket, Key=scaling_metadata_key)
        scaling_metadata = json.loads(scaling_metadata_obj['Body'].read().decode('utf-8'))
        logger.info("Scaling metadata loaded successfully.")
        return scaling_metadata
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Scaling metadata file not found in S3: s3://{bucket}/{scaling_metadata_key}")
        raise FileNotFoundError(f"Scaling metadata file not found in S3: s3://{bucket}/{scaling_metadata_key}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in scaling metadata: {e}")
        raise ValueError(f"Invalid JSON format in scaling metadata: {e}")
    except Exception as e:
        logger.error(f"Failed to load scaling metadata from S3: {e}")
        raise


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
    """
    Upload a file to S3 safely.

    Args:
        local_path (str): Path to the local file.
        s3_key (str): S3 key where the file will be uploaded.
        s3_client (boto3.client): The S3 client.
        bucket (str): The S3 bucket name.
        logger (logging.Logger): Logger for logging information.
        overwrite (bool, optional): Whether to overwrite if the file exists. Defaults to False.
    """
    safe_s3_upload(
        s3_client=s3_client,
        logger=logger,
        local_path=local_path,
        s3_key=s3_key,
        bucket=bucket,
        overwrite=overwrite
    )


def download_file_from_s3(
    s3_client: boto3.client,
    bucket: str,
    s3_key: str,
    local_path: str,
    logger: logging.Logger
) -> None:
    """
    Download a file from S3.

    Args:
        s3_client (boto3.client): The S3 client.
        bucket (str): The S3 bucket name.
        s3_key (str): The S3 key of the file to download.
        local_path (str): The local path where the file will be saved.
        logger (logging.Logger): Logger for logging information.
    """
    try:
        s3_client.download_file(bucket, s3_key, local_path)
        logger.info(f"Downloaded {s3_key} to {local_path}")
    except Exception as e:
        logger.error(f"Failed to download {s3_key} from S3: {e}")
        raise


def load_scaling_params(
    s3_client: boto3.client,
    bucket: str,
    scaling_params_key: str,
    logger: logging.Logger
) -> Dict[Tuple[str, str], ScalingParameters]:
    """
    Load and parse scaling parameters from S3.

    Args:
        s3_client (boto3.client): The S3 client.
        bucket (str): The S3 bucket name.
        scaling_params_key (str): The S3 key for the scaling parameters JSON file.
        logger (logging.Logger): Logger for logging information.

    Returns:
        Dict[Tuple[str, str], ScalingParameters]: A dictionary mapping (Currency, BranchId) to ScalingParameters.
    """
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
    logger.info("Scaling parameters loaded and converted to dataclass.")
    return scaling_params


def save_scaling_params(
    scaling_params: Dict[Tuple[str, str], ScalingParameters],
    filepath: str
) -> None:
    """
    Save scaling parameters to a JSON file.

    Args:
        scaling_params (Dict[Tuple[str, str], ScalingParameters]): The scaling parameters.
        filepath (str): The local path where the JSON file will be saved.
    """
    serializable_params = {
        f"{currency}_{branch}": params.__dict__
        for (currency, branch), params in scaling_params.items()
    }
    with open(filepath, 'w') as f:
        json.dump(serializable_params, f, indent=2)


def prepare_training_data(
    input_file: str,
    required_columns: List[str],
    categorical_columns: List[str],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load and preprocess training data.

    Args:
        input_file (str): Path to the input CSV file.
        required_columns (List[str]): Columns required in the data.
        categorical_columns (List[str]): Columns to treat as categorical.
        logger (logging.Logger): Logger for logging information.

    Returns:
        pd.DataFrame: The preprocessed training data.
    """
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
    """
    Calculate scaling parameters per group.

    Args:
        data (pd.DataFrame): The training data.
        group_columns (List[str]): Columns to group by for scaling.
        target_column (str): The target column to scale.
        logger (logging.Logger): Logger for logging information.

    Returns:
        Dict[Tuple[str, str], ScalingParameters]: Scaling parameters for each group.
    """
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
    logger.info("All scaling parameters calculated successfully.")
    return scaling_params


def apply_scaling(
    data: pd.DataFrame,
    scaling_params: Dict[Tuple[str, str], ScalingParameters],
    group_columns: List[str],
    target_column: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Apply scaling to the target column based on scaling parameters.

    Args:
        data (pd.DataFrame): The training data.
        scaling_params (Dict[Tuple[str, str], ScalingParameters]): Scaling parameters per group.
        group_columns (List[str]): Columns to group by for scaling.
        target_column (str): The target column to scale.
        logger (logging.Logger): Logger for logging information.

    Returns:
        pd.DataFrame: The scaled data.
    """
    scaled_data = data.copy()
    for group, params in scaling_params.items():
        mask = np.ones(len(data), dtype=bool)
        for col, val in zip(group_columns, group):
            mask &= data[col] == val
        scaled_values = (data.loc[mask, target_column] - params.mean) / (params.std if params.std != 0 else 1)
        scaled_data.loc[mask, target_column] = scaled_values
        logger.debug(f"Applied scaling for group {group}.")
    logger.info("Scaling applied to all relevant groups.")
    return scaled_data


def create_output_directory(
    base_path: str,
    country_code: str,
    timestamp: str
) -> str:
    """
    Create and return the output directory path.

    Args:
        base_path (str): The base directory path.
        country_code (str): The country code being processed.
        timestamp (str): The current timestamp.

    Returns:
        str: The full path to the created output directory.
    """
    output_dir = os.path.join(base_path, f"{country_code}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_dataframe_to_csv(
    df: pd.DataFrame,
    filepath: str
) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filepath (str): The path where the CSV will be saved.
    """
    df.to_csv(filepath, index=False)


def load_inference_template(
    inference_template_file: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load and validate the inference template.

    Args:
        inference_template_file (str): Path to the inference template CSV file.
        logger (logging.Logger): Logger for logging information.

    Returns:
        pd.DataFrame: The validated inference template.
    """
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
    """
    Generate inference data based on the template and forecast horizon.

    Args:
        inference_template (pd.DataFrame): The inference template DataFrame.
        effective_date (pd.Timestamp): The effective date for forecasts.
        forecast_horizon (int): The number of days to forecast.
        logger (logging.Logger): Logger for logging information.

    Returns:
        pd.DataFrame: The generated inference data.
    """
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


def validate_scaling_parameters(
    scaling_params: Dict[Tuple[str, str], ScalingParameters]
) -> None:
    """
    Validate the structure and content of scaling parameters.

    Args:
        scaling_params (Dict[Tuple[str, str], ScalingParameters]): Scaling parameters to validate.

    Raises:
        ValueError: If any scaling parameter is invalid.
    """
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
    sm_client: boto3.client,
    job_name: str,
    logger: logging.Logger,
    max_wait_time: int = 24 * 60 * 60  # 24 hours
) -> str:
    """
    Monitor the AutoML job until completion.

    Args:
        sm_client (boto3.client): The SageMaker client.
        job_name (str): The name of the AutoML job.
        logger (logging.Logger): Logger for logging information.
        max_wait_time (int, optional): Maximum wait time in seconds. Defaults to 24 * 60 * 60.

    Returns:
        str: The final status of the AutoML job.

    Raises:
        TimeoutError: If the job does not complete within the maximum wait time.
    """
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
    sm_client: boto3.client,
    job_name: str,
    country_code: str,
    timestamp: str,
    role_arn: str,
    logger: logging.Logger
) -> str:
    """
    Retrieve the best model from the completed AutoML job.

    Args:
        sm_client (boto3.client): The SageMaker client.
        job_name (str): The name of the AutoML job.
        country_code (str): The country code.
        timestamp (str): The current timestamp.
        role_arn (str): The ARN of the SageMaker execution role.
        logger (logging.Logger): Logger for logging information.

    Returns:
        str: The name of the created SageMaker model.

    Raises:
        ValueError: If no best candidate is found or model creation fails.
    """
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
                f"A model with name {model_name} already exists. Choose a different timestamp or model name."
            )
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
