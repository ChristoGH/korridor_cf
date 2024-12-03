# cash_forecast_lib.py

import os
import json
import logging
import boto3
import pandas as pd
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
from time import strftime, gmtime
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


def submit_batch_transform_job(
        sm_client,
        model_name: str,
        transform_job_name: str,
        input_s3_uri: str,
        output_s3_uri: str,
        instance_type: str,
        instance_count: int,
        content_type: str,
        split_type: str,
        accept_type: str,
        logger: logging.Logger
) -> None:
    """Submit a SageMaker Batch Transform job."""
    try:
        sm_client.create_transform_job(
            TransformJobName=transform_job_name,
            ModelName=model_name,
            BatchStrategy='MultiRecord',
            TransformInput={
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': input_s3_uri
                    }
                },
                'ContentType': content_type,  # e.g., 'text/csv'
                'SplitType': split_type  # e.g., 'Line'
            },
            TransformOutput={
                'S3OutputPath': output_s3_uri,
                'AssembleWith': 'Line',
                'Accept': accept_type  # e.g., 'text/csv'
            },
            TransformResources={
                'InstanceType': instance_type,
                'InstanceCount': instance_count
            }
        )
        logger.info(f"Submitted Batch Transform job: {transform_job_name}")
    except Exception as e:
        logger.error(f"Failed to submit Batch Transform job {transform_job_name}: {e}")
        raise


def monitor_batch_transform_job(
        sm_client,
        transform_job_name: str,
        logger: logging.Logger,
        max_wait_time: int = 24 * 60 * 60  # 24 hours
) -> None:
    """Monitor the Batch Transform job until completion."""
    logger.info(f"Monitoring Batch Transform job: {transform_job_name}")
    sleep_time = 60  # Start with 1 minute
    elapsed_time = 0
    while elapsed_time < max_wait_time:
        try:
            response = sm_client.describe_transform_job(TransformJobName=transform_job_name)
            status = response['TransformJobStatus']
            logger.info(f"Transform job {transform_job_name} status: {status}")
            if status in ['Completed', 'Failed', 'Stopped']:
                if status != 'Completed':
                    failure_reason = response.get('FailureReason', 'No failure reason provided.')
                    logger.error(f"Transform job {transform_job_name} failed: {failure_reason}")
                    raise RuntimeError(f"Transform job {transform_job_name} failed: {failure_reason}")
                logger.info(f"Transform job {transform_job_name} completed successfully.")
                return
            sleep(sleep_time)
            elapsed_time += sleep_time
            sleep_time = min(int(sleep_time * 1.5), 600)  # Increase sleep time up to 10 minutes
        except Exception as e:
            logger.error(f"Error monitoring Batch Transform job {transform_job_name}: {e}")
            sleep_time = min(int(sleep_time * 1.5), 600)
            elapsed_time += sleep_time
            sleep(60)  # Wait before retrying
    raise TimeoutError(f"Transform job {transform_job_name} did not complete within {max_wait_time} seconds.")


def download_forecast_results(
        s3_client,
        bucket: str,
        output_s3_prefix: str,
        local_output_dir: str,
        logger: logging.Logger
) -> pd.DataFrame:
    """Download and combine forecast result files from S3."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=output_s3_prefix)
    except Exception as e:
        logger.error(f"Failed to list objects in S3: {e}")
        raise

    if 'Contents' not in response:
        raise FileNotFoundError(f"No forecast results found in S3 for {output_s3_prefix}")

    # Create a temporary directory
    temp_dir = os.path.join(local_output_dir, "temp_forecast")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Download and process forecast files
        forecast_data = []
        for obj in response['Contents']:
            s3_key = obj['Key']
            if s3_key.endswith('.out'):  # Assuming forecast output files have .out extension
                local_file = os.path.join(temp_dir, os.path.basename(s3_key))
                try:
                    download_file_from_s3(s3_client, bucket, s3_key, local_file, logger)
                    # Read forecast data as CSV
                    df = pd.read_csv(local_file, header=None)
                    forecast_data.append(df)
                except Exception as e:
                    logger.error(f"Failed to process {s3_key}: {e}")
                    continue
                finally:
                    if os.path.exists(local_file):
                        os.remove(local_file)

        if not forecast_data:
            raise FileNotFoundError("No forecast output files found.")

        # Combine forecast data
        forecast_df = pd.concat(forecast_data, ignore_index=True)
        logger.info(f"Combined forecast data shape: {forecast_df.shape}")

        # Assign 'ForecastDate' and other columns from inference_df
        # Note: You need to pass 'inference_df' or relevant mapping to this function
        # For simplicity, assuming 'inference_df' is available globally or passed as a parameter
        # Here, we'll skip assigning additional columns as it's context-dependent

        return forecast_df

    except Exception as e:
        logger.error(f"Error in download_forecast_results: {e}")
        raise
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Removed temporary directory {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove temp directory {temp_dir}: {e}")


def save_forecasts_locally(
        forecast_df: pd.DataFrame,
        output_file: str,
        quantiles: List[str],
        logger: logging.Logger
) -> None:
    """Process and save forecast DataFrame to a CSV file locally."""
    try:
        # Assign quantile column names if not present
        if not all(col in forecast_df.columns for col in quantiles):
            raise ValueError("Forecast DataFrame does not contain all quantile columns.")

        # Inverse scaling should be applied here if necessary
        # This depends on your specific scaling logic

        # Save to CSV
        forecast_df.to_csv(output_file, index=False)
        logger.info(f"Final forecast saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save forecasts locally: {e}")
        raise
