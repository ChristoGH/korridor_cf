assistant, I build a timeseriesforescating model with the following script.  It seemingly does what it is supposed to do.


Here  is the input data:

zm.head()=
BranchId	CountryCode	CountryName	ProductId	Currency	EffectiveDate	Demand	DayOfWeekName
0	11	ZM	Zambia	18	USD	2023-05-01	144458.0	Monday
1	11	ZM	Zambia	71	ZMW	2023-05-01	95736.2	Monday
2	13	ZM	Zambia	21	USD	2023-05-01	1445.0	Monday
3	13	ZM	Zambia	70	ZMW	2023-05-01	14326.0	Monday
4	13	ZM	Zambia	341	BWP	2023-05-01	18260.0	Monday

I call the model from the commandline with

python cs_scripts/cash_forecast_model.py --config cs_scripts/config.yaml --countries ZM

this is the config.yaml is

# filename config.yaml
region: eu-west-1
bucket: 'sagemaker-eu-west-1-717377802724'
prefix: 'cash-forecasting'
role_arn: 'arn:aws:iam::717377802724:role/service-role/AmazonSageMaker-ExecutionRole-20231102T230107'
instance_type: 'ml.c5.2xlarge'  # Change this line
instance_count: 1
forecast_horizon: 10  # Changed to 10 days ahead forecasting
forecast_frequency: '1D'
batch_size: 100
quantiles:
  - 'p10'
  - 'p50'
  - 'p90'
iterative_forecast_steps: 1  # If applicable


Here is cash_forecast_model.py:
# cash_forecast_model.py
# python cs_scripts/cash_forecast_model.py --config cs_scripts/config.yaml --countries ZM

import argparse
import boto3
import pandas as pd
import numpy as np
import os
import sys
import json
import shutil
from time import gmtime, strftime, sleep
from typing import Dict, Tuple, List
from dataclasses import dataclass, field

import logging  # <-- Added import for logging
from sagemaker import Session
import yaml
import glob

from common import Config, load_and_validate_config, setup_logging, safe_s3_upload, load_scaling_parameters


class CashForecastingPipeline:
    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize the forecasting pipeline with configuration."""
        self.config = config
        self.logger = logger
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=config.region)
        self.s3_client = boto3.client('s3')
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = config.role_arn
        self.train_file = None
        self.output_dir = None  # Will be set in prepare_data

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

    def _load_scaling_parameters_from_s3(self, country_code: str) -> Dict[Tuple[str, str], Dict]:
        """Load scaling parameters from S3."""
        scaling_params_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/scaling/{country_code}_scaling_params.json"
        return load_scaling_parameters(
            s3_client=self.s3_client,
            logger=self.logger,
            bucket=self.config.bucket,
            scaling_params_key=scaling_params_key
        )

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str]:
        """Prepare data for training and create inference template."""
        # Load and preprocess data
        data = pd.read_csv(input_file)
        required_columns = ['ProductId', 'BranchId', 'Currency', 'EffectiveDate', 'Demand']
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

        data['ProductId'] = data['ProductId'].astype(str)
        data['BranchId'] = data['BranchId'].astype(str)
        data['Currency'] = data['Currency'].astype(str)
        data.sort_values('EffectiveDate', inplace=True)

        # Calculate scaling parameters per Currency-BranchId combination
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

        # Apply scaling to Demand values
        scaled_data = data.copy()
        for (currency, branch), params in scaling_params.items():
            mask = (data['Currency'] == currency) & (data['BranchId'] == branch)
            # Standardize the data
            scaled_data.loc[mask, 'Demand'] = (
                (data.loc[mask, 'Demand'] - params['mean']) /
                (params['std'] if params['std'] != 0 else 1)
            )

        # Create directories for output
        self.output_dir = f"./cs_data/cash/{country_code}_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Save scaling parameters
        scaling_file = os.path.join(self.output_dir, f"{country_code}_scaling_params.json")
        with open(scaling_file, 'w') as f:
            # Convert scaling_params keys to strings for JSON serialization
            serializable_params = {
                f"{currency}_{branch}": params
                for (currency, branch), params in scaling_params.items()
            }
            json.dump(serializable_params, f, indent=2)

        # Save metadata about the scaling
        metadata = {
            'scaling_method': 'standardization',
            'scaling_level': 'currency_branch',
            'scaled_column': 'Demand',
            'timestamp': self.timestamp,
            'scaling_stats': {
                'global_mean': data['Demand'].mean(),
                'global_std': data['Demand'].std(),
                'global_min': data['Demand'].min(),
                'global_max': data['Demand'].max(),
                'n_groups': len(scaling_params),
                'n_total_observations': len(data)
            }
        }

        metadata_file = os.path.join(self.output_dir, f"{country_code}_scaling_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Define file paths
        train_file = os.path.join(self.output_dir, f"{country_code}_train.csv")
        inference_template_file = os.path.join(self.output_dir, f"{country_code}_inference_template.csv")

        # Save training data
        scaled_data.to_csv(train_file, index=False)
        self.train_file = train_file  # Save train file path for later use

        # Create inference template with unique combinations
        inference_template = data.drop_duplicates(
            subset=['ProductId', 'BranchId', 'Currency']
        )[['ProductId', 'BranchId', 'Currency']]

        # Save inference template
        inference_template.to_csv(inference_template_file, index=False)

        # Log scaling information
        self.logger.info(f"Applied standardization scaling at currency-branch level")
        self.logger.info(f"Scaling parameters saved to {scaling_file}")
        self.logger.info(f"Scaling metadata saved to {metadata_file}")
        self.logger.info(f"Global scale: mean={metadata['scaling_stats']['global_mean']:.2f}, "
                         f"std={metadata['scaling_stats']['global_std']:.2f}")

        return train_file, inference_template_file

    def train_model(self, country_code: str, train_data_s3_uri: str) -> str:
        """Train the forecasting model."""
        self.logger.info(f"Starting model training for {country_code}")

        job_name = f"{country_code}-ts-{self.timestamp}"

        # Configure AutoML job
        automl_config = {
            'AutoMLJobName': job_name,
            'AutoMLJobInputDataConfig': [{
                'ChannelType': 'training',
                'ContentType': 'text/csv;header=present',
                'CompressionType': 'None',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': train_data_s3_uri
                    }
                }
            }],
            'OutputDataConfig': {
                'S3OutputPath': f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/output"
            },
            'AutoMLProblemTypeConfig': {
                'TimeSeriesForecastingJobConfig': {
                    'ForecastFrequency': self.config.forecast_frequency,
                    'ForecastHorizon': self.config.forecast_horizon,
                    'ForecastQuantiles': self.config.quantiles,
                    'TimeSeriesConfig': {
                        'TargetAttributeName': 'Demand',
                        'TimestampAttributeName': 'EffectiveDate',
                        'ItemIdentifierAttributeName': 'ProductId',
                        'GroupingAttributeNames': ['BranchId', 'Currency']
                    }
                }
            },
            'RoleArn': self.role
        }

        # Start training
        try:
            self.sm_client.create_auto_ml_job_v2(**automl_config)
            self.logger.info(f"AutoML job {job_name} initiated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to start AutoML job: {e}")
            raise
        return job_name

    def _monitor_job(self, job_name: str, max_wait_time: int = 24 * 60 * 60) -> str:
        """Monitor the AutoML job until completion with a timeout."""
        self.logger.info(f"Monitoring job {job_name}")
        sleep_time = 60  # Start with 1 minute
        elapsed_time = 0
        while elapsed_time < max_wait_time:
            try:
                response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
                status = response['AutoMLJobStatus']
                self.logger.info(f"Job {job_name} status: {status}")
                if status in ['Completed', 'Failed', 'Stopped']:
                    if status != 'Completed':
                        failure_reason = response.get('FailureReason', 'No failure reason provided.')
                        self.logger.error(f"AutoML job {job_name} failed: {failure_reason}")
                    return status
                sleep(sleep_time)
                elapsed_time += sleep_time
                sleep_time = min(int(sleep_time * 1.5), 600)  # Increase sleep time up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring AutoML job: {e}")
                sleep_time = min(int(sleep_time * 1.5), 600)
                elapsed_time += sleep_time
                sleep(60)  # Wait before retrying
        raise TimeoutError(f"AutoML job {job_name} did not complete within {max_wait_time} seconds.")

    def _get_best_model(self, job_name: str, country_code: str) -> str:
        """Retrieve the best model from the AutoML job."""
        self.logger.info(f"Retrieving best model for job {job_name}")
        while True:
            try:
                response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
                if 'BestCandidate' in response:
                    best_candidate = response['BestCandidate']
                    break
                else:
                    self.logger.info(f"Waiting for best candidate model to be available for job {job_name}")
                    sleep(60)
            except Exception as e:
                self.logger.error(f"Error retrieving best model: {e}")
                sleep(60)

        model_name = f"{country_code}-model-{self.timestamp}"

        # Check if model already exists
        try:
            existing_models = self.sm_client.list_models(NameContains=model_name)
            if existing_models['Models']:
                raise ValueError(f"A model with name {model_name} already exists. Aborting to prevent overwriting.")
        except Exception as e:
            self.logger.error(f"Error checking for existing models: {e}")
            raise

        # Create model
        try:
            self.sm_client.create_model(
                ModelName=model_name,
                ExecutionRoleArn=self.role,
                Containers=best_candidate['InferenceContainers']
            )
            self.logger.info(f"Created model {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to create model {model_name}: {e}")
            raise
        return model_name

    def run_pipeline(self, country_code: str, input_file: str, backtesting: bool = False) -> None:
        """Run the complete forecasting pipeline."""
        try:
            # Prepare data
            train_file, template_file = self.prepare_data(input_file, country_code)

            # Upload training data to S3
            s3_train_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/train/{os.path.basename(train_file)}"
            self._safe_s3_upload(train_file, s3_train_key)
            train_data_s3_uri = f"s3://{self.config.bucket}/{os.path.dirname(s3_train_key)}/"

            # Upload scaling parameters to S3
            s3_scaling_params_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/scaling/{country_code}_scaling_params.json"
            scaling_params_file = os.path.join(self.output_dir, f"{country_code}_scaling_params.json")
            self._safe_s3_upload(scaling_params_file, s3_scaling_params_key, overwrite=True)
            self.logger.info(f"Uploaded scaling parameters to s3://{self.config.bucket}/{s3_scaling_params_key}")

            # Upload scaling metadata to S3
            s3_scaling_metadata_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/scaling/{country_code}_scaling_metadata.json"
            scaling_metadata_file = os.path.join(self.output_dir, f"{country_code}_scaling_metadata.json")
            self._safe_s3_upload(scaling_metadata_file, s3_scaling_metadata_key, overwrite=True)
            self.logger.info(f"Uploaded scaling metadata to s3://{self.config.bucket}/{s3_scaling_metadata_key}")

            # Train model
            job_name = self.train_model(
                country_code,
                train_data_s3_uri
            )

            # Monitor training
            status = self._monitor_job(job_name)
            if status != 'Completed':
                raise RuntimeError(f"Training failed with status: {status}")

            # Get best model
            model_name = self._get_best_model(job_name, country_code)

            # (Optional) Forecasting step can be removed if not needed in the training script

        except Exception as e:
            self.logger.error(f"Pipeline failed for {country_code}: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Cash Forecasting Pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--countries', nargs='+', default=['ZM'],
                        help='Country codes to process')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to the input CSV file (overrides default)')
    args = parser.parse_args()

    # Load and validate configuration
    config = load_and_validate_config(args.config)

    # Setup logging
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
    logger = setup_logging('CashForecast', timestamp)

    for country_code in args.countries:
        pipeline = CashForecastingPipeline(config, logger)
        try:
            if args.input_file:
                input_file = args.input_file
            else:
                input_file = f"./data/cash/{country_code}.csv"
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            pipeline.run_pipeline(country_code, input_file, backtesting=False)

        except Exception as e:
            logger.error(f"Failed to process {country_code}: {str(e)}")


if __name__ == "__main__":
    main()
 

Next I do inference with this script:

# cash_forecast_inference.py


import argparse
import boto3
import pandas as pd
import numpy as np
import os
import sys
import json
import shutil
from time import gmtime, strftime, sleep
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from sagemaker import Session
import yaml

from common import Config, load_and_validate_config, setup_logging, safe_s3_upload, load_scaling_parameters

import logging  # Ensure logging is imported


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
            self.scaling_params = load_scaling_parameters(
                s3_client=self.s3_client,
                logger=self.logger,
                bucket=self.config.bucket,
                scaling_params_key=scaling_params_key
            )

            # Download scaling metadata
            self.logger.info(f"Downloading scaling metadata from s3://{self.config.bucket}/{scaling_metadata_key}")
            try:
                scaling_metadata_obj = self.s3_client.get_object(Bucket=self.config.bucket, Key=scaling_metadata_key)
                self.scaling_metadata = json.loads(scaling_metadata_obj['Body'].read().decode('utf-8'))
            except Exception as e:
                self.logger.error(f"Failed to load scaling metadata from S3: {e}")
                raise

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
        forecast_df = self._get_forecast_result(country_code)

        # Save forecasts
        self._save_forecasts(country_code, forecast_df)

        return forecast_df


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

    def _get_forecast_result(self, country_code: str, inference_df: pd.DataFrame) -> pd.DataFrame:
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
                if s3_key.endswith('.out'):
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

            # Assign 'ForecastDate' from inference_df
            if len(forecast_df) != len(inference_df):
                self.logger.error(f"Mismatch between inference data and forecast results. "
                                  f"Inference records: {len(inference_df)}, Forecast records: {len(forecast_df)}")
                raise ValueError("Mismatch between inference data and forecast results.")

            forecast_df['ForecastDate'] = inference_df['ForecastDate'].values
            self.logger.info("Assigned 'ForecastDate' to forecast results based on inference data.")

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
                        self.logger.warning(
                            f"Quantile '{quantile}' not found in forecast data for Currency={currency}, Branch={branch}")

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

with the inferenece template:

filtered_zm[['ProductId','BranchId','Currency','EffectiveDate','DayOfWeekName']].to_csv('~/cs_data/cash/ZM_custom_inference/ZM_inference_custom.csv', index=False)

ProductId	BranchId	Currency	EffectiveDate	DayOfWeekName
16758	18	11	USD	2024-07-15	Monday


The following a screenshot of the last attempt:

sagemaker-user@default:~$ python cs_scripts/cash_forecast_inference.py   --model_timestamp 20241201-153323   --effective_date 2024-07-15   --config cs_scripts/config.yaml   --countries ZM   --model_name ZM-model-20241201-153323   --inference_template ./cs_data/cash/ZM_custom_inference/ZM_inference_custom.csv
/opt/conda/lib/python3.10/site-packages/pydantic/_internal/_fields.py:172: UserWarning: Field name "json" in "MonitoringDatasetFormat" shadows an attribute in parent "Base"
  warnings.warn(
sagemaker.config INFO - Not applying SDK defaults from location: /etc/xdg/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /home/sagemaker-user/.config/sagemaker/config.yaml
2024-12-02 04:19:40,723 - CashForecastInference - INFO - Starting Cash Forecasting Inference Pipeline
[12/02/24 04:19:40] INFO     Starting Cash Forecasting Inference Pipeline                                                                cash_forecast_inference.py:681
2024-12-02 04:19:40,732 - CashForecastInference - INFO - Arguments validated successfully:
                    INFO     Arguments validated successfully:                                                                           cash_forecast_inference.py:668
2024-12-02 04:19:40,733 - CashForecastInference - INFO -   Model Timestamp: 20241201-153323
                    INFO       Model Timestamp: 20241201-153323                                                                          cash_forecast_inference.py:669
2024-12-02 04:19:40,735 - CashForecastInference - INFO -   Effective Date: 2024-07-15
                    INFO       Effective Date: 2024-07-15                                                                                cash_forecast_inference.py:670
2024-12-02 04:19:40,738 - CashForecastInference - INFO -   Countries: ['ZM']
                    INFO       Countries: ['ZM']                                                                                         cash_forecast_inference.py:671
2024-12-02 04:19:40,740 - CashForecastInference - INFO -   Model Name: ZM-model-20241201-153323
                    INFO       Model Name: ZM-model-20241201-153323                                                                      cash_forecast_inference.py:672
2024-12-02 04:19:40,742 - CashForecastInference - INFO - Loading configuration from cs_scripts/config.yaml
                    INFO     Loading configuration from cs_scripts/config.yaml                                                           cash_forecast_inference.py:705
2024-12-02 04:19:40,747 - CashForecastInference - INFO - Configuration validated successfully
                    INFO     Configuration validated successfully                                                                        cash_forecast_inference.py:707
2024-12-02 04:19:40,750 - CashForecastInference - INFO - 
Processing country: ZM
                    INFO                                                                                                                 cash_forecast_inference.py:711
                             Processing country: ZM                                                                                                                    
2024-12-02 04:19:41,245 - CashForecastInference - INFO - Loading scaling parameters for ZM
[12/02/24 04:19:41] INFO     Loading scaling parameters for ZM                                                                           cash_forecast_inference.py:717
2024-12-02 04:19:41,247 - CashForecastInference - INFO - Downloading scaling parameters from s3://sagemaker-eu-west-1-717377802724/cash-forecasting-ZM/20241201-153323/scaling/ZM_scaling_params.json
                    INFO     Downloading scaling parameters from                                                                         cash_forecast_inference.py:108
                             s3://sagemaker-eu-west-1-717377802724/cash-forecasting-ZM/20241201-153323/scaling/ZM_scaling_params.json                                  
2024-12-02 04:19:41,310 - CashForecastInference - INFO - Downloading scaling metadata from s3://sagemaker-eu-west-1-717377802724/cash-forecasting-ZM/20241201-153323/scaling/ZM_scaling_metadata.json
                    INFO     Downloading scaling metadata from                                                                           cash_forecast_inference.py:117
                             s3://sagemaker-eu-west-1-717377802724/cash-forecasting-ZM/20241201-153323/scaling/ZM_scaling_metadata.json                                
2024-12-02 04:19:41,335 - CashForecastInference - INFO - Successfully loaded and validated scaling parameters for ZM
                    INFO     Successfully loaded and validated scaling parameters for ZM                                                 cash_forecast_inference.py:133
2024-12-02 04:19:41,337 - CashForecastInference - INFO - Preparing inference data for ZM
                    INFO     Preparing inference data for ZM                                                                             cash_forecast_inference.py:724
2024-12-02 04:19:41,349 - CashForecastInference - INFO - Inference data prepared and saved to ./cs_data/cash/ZM_20241202-041941/ZM_inference.csv
                    INFO     Inference data prepared and saved to ./cs_data/cash/ZM_20241202-041941/ZM_inference.csv                     cash_forecast_inference.py:198
2024-12-02 04:19:41,350 - CashForecastInference - INFO - Running inference for ZM
                    INFO     Running inference for ZM                                                                                    cash_forecast_inference.py:732
2024-12-02 04:19:41,381 - CashForecastInference - INFO - Uploaded ./cs_data/cash/ZM_20241202-041941/ZM_inference.csv to s3://sagemaker-eu-west-1-717377802724/cash-forecasting-ZM/20241202-041941/inference/ZM_inference.csv
                    INFO     Uploaded ./cs_data/cash/ZM_20241202-041941/ZM_inference.csv to                                                               common.py:120
                             s3://sagemaker-eu-west-1-717377802724/cash-forecasting-ZM/20241202-041941/inference/ZM_inference.csv                                      
2024-12-02 04:19:41,383 - CashForecastInference - INFO - Inference data uploaded to s3://sagemaker-eu-west-1-717377802724/cash-forecasting-ZM/20241202-041941/inference/ZM_inference.csv
                    INFO     Inference data uploaded to                                                                                  cash_forecast_inference.py:222
                             s3://sagemaker-eu-west-1-717377802724/cash-forecasting-ZM/20241202-041941/inference/ZM_inference.csv                                      
2024-12-02 04:19:41,387 - CashForecastInference - INFO - Inference data loaded from ./cs_data/cash/ZM_20241202-041941/ZM_inference.csv
                    INFO     Inference data loaded from ./cs_data/cash/ZM_20241202-041941/ZM_inference.csv                               cash_forecast_inference.py:226
2024-12-02 04:19:41,885 - CashForecastInference - INFO - Created transform job ZM-model-20241201-153323-transform-20241202-041941-1
                    INFO     Created transform job ZM-model-20241201-153323-transform-20241202-041941-1                                  cash_forecast_inference.py:352
2024-12-02 04:19:41,888 - CashForecastInference - INFO - Monitoring transform job ZM-model-20241201-153323-transform-20241202-041941-1
                    INFO     Monitoring transform job ZM-model-20241201-153323-transform-20241202-041941-1                               cash_forecast_inference.py:360
2024-12-02 04:19:41,918 - CashForecastInference - INFO - Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress
                    INFO     Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress                       cash_forecast_inference.py:367
2024-12-02 04:20:11,970 - CashForecastInference - INFO - Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress
[12/02/24 04:20:11] INFO     Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress                       cash_forecast_inference.py:367
2024-12-02 04:20:57,085 - CashForecastInference - INFO - Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress
[12/02/24 04:20:57] INFO     Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress                       cash_forecast_inference.py:367
2024-12-02 04:22:04,237 - CashForecastInference - INFO - Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress
[12/02/24 04:22:04] INFO     Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress                       cash_forecast_inference.py:367
2024-12-02 04:23:44,427 - CashForecastInference - INFO - Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress
[12/02/24 04:23:44] INFO     Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress                       cash_forecast_inference.py:367
2024-12-02 04:26:14,616 - CashForecastInference - INFO - Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress
[12/02/24 04:26:14] INFO     Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: InProgress                       cash_forecast_inference.py:367
2024-12-02 04:29:59,781 - CashForecastInference - INFO - Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: Completed
[12/02/24 04:29:59] INFO     Transform job ZM-model-20241201-153323-transform-20241202-041941-1 status: Completed                        cash_forecast_inference.py:367
2024-12-02 04:29:59,956 - CashForecastInference - INFO - Downloaded forecast file cash-forecasting-ZM/20241202-041941/inference-output/ZM_inference.csv.out to ./temp/ZM_20241202-041941/ZM_inference.csv.out
                    INFO     Downloaded forecast file cash-forecasting-ZM/20241202-041941/inference-output/ZM_inference.csv.out to       cash_forecast_inference.py:411
                             ./temp/ZM_20241202-041941/ZM_inference.csv.out                                                                                            
2024-12-02 04:29:59,967 - CashForecastInference - INFO - Combined forecast data shape: (10, 8)
                    INFO     Combined forecast data shape: (10, 8)                                                                       cash_forecast_inference.py:428
2024-12-02 04:29:59,975 - CashForecastInference - INFO - Forecast df columns after merging: ['ProductId', 'BranchId', 'Currency', 'EffectiveDate', 'ForecastDate', 'Demand', 'p10', 'p50', 'p90', 'mean']
                    INFO     Forecast df columns after merging: ['ProductId', 'BranchId', 'Currency', 'EffectiveDate', 'ForecastDate',   cash_forecast_inference.py:451
                             'Demand', 'p10', 'p50', 'p90', 'mean']                                                                                                    
2024-12-02 04:29:59,982 - CashForecastInference - INFO - Restoring original scale to forecasts...
                    INFO     Restoring original scale to forecasts...                                                                    cash_forecast_inference.py:459
2024-12-02 04:29:59,987 - CashForecastInference - WARNING - No forecasts found for Currency=BWP, Branch=13
                    WARNING  No forecasts found for Currency=BWP, Branch=13                                                              cash_forecast_inference.py:463
2024-12-02 04:29:59,991 - CashForecastInference - WARNING - No forecasts found for Currency=BWP, Branch=244
                    WARNING  No forecasts found for Currency=BWP, Branch=244                                                             cash_forecast_inference.py:463
2024-12-02 04:29:59,999 - CashForecastInference - WARNING - No forecasts found for Currency=BWP, Branch=43
                    WARNING  No forecasts found for Currency=BWP, Branch=43                                                              cash_forecast_inference.py:463
2024-12-02 04:30:00,026 - CashForecastInference - WARNING - No forecasts found for Currency=NAD, Branch=244
[12/02/24 04:30:00] WARNING  No forecasts found for Currency=NAD, Branch=244                                                             cash_forecast_inference.py:463
2024-12-02 04:30:00,040 - CashForecastInference - WARNING - No forecasts found for Currency=TZS, Branch=65
                    WARNING  No forecasts found for Currency=TZS, Branch=65                                                              cash_forecast_inference.py:463
2024-12-02 04:30:00,068 - CashForecastInference - ERROR - Error in _get_forecast_result: can't multiply sequence by non-int of type 'float'
Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 220, in _na_arithmetic_op
    result = func(left, right)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/computation/expressions.py", line 242, in evaluate
    return _evaluate(op, op_str, a, b)  # type: ignore[misc]
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/computation/expressions.py", line 73, in _evaluate_standard
    return op(a, b)
TypeError: can't multiply sequence by non-int of type 'float'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/sagemaker-user/cs_scripts/cash_forecast_inference.py", line 469, in _get_forecast_result
    (forecast_df.loc[mask, quantile] * (params['std'] if params['std'] != 0 else 1)) +
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/common.py", line 76, in new_method
    return method(self, other)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/arraylike.py", line 202, in __mul__
    return self._arith_method(other, operator.mul)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/series.py", line 5819, in _arith_method
    return base.IndexOpsMixin._arith_method(self, other, op)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/base.py", line 1381, in _arith_method
    result = ops.arithmetic_op(lvalues, rvalues, op)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 285, in arithmetic_op
    res_values = _na_arithmetic_op(left, right, op)  # type: ignore[arg-type]
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 229, in _na_arithmetic_op
    result = _masked_arith_op(left, right, op)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 184, in _masked_arith_op
    result[mask] = op(xrav[mask], y)
TypeError: can't multiply sequence by non-int of type 'float'
                    ERROR    Error in _get_forecast_result: can't multiply sequence by non-int of type 'float'                           cash_forecast_inference.py:483
                             Traceback (most recent call last):                                                                                                        
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 220, in                                               
                             _na_arithmetic_op                                                                                                                         
                                 result = func(left, right)                                                                                                            
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/computation/expressions.py", line 242, in                                     
                             evaluate                                                                                                                                  
                                 return _evaluate(op, op_str, a, b)  # type: ignore                                                                                    
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/computation/expressions.py", line 73, in                                      
                             _evaluate_standard                                                                                                                        
                                 return op(a, b)                                                                                                                       
                             TypeError: can't multiply sequence by non-int of type 'float'                                                                             
                                                                                                                                                                       
                             During handling of the above exception, another exception occurred:                                                                       
                                                                                                                                                                       
                             Traceback (most recent call last):                                                                                                        
                               File "/home/sagemaker-user/cs_scripts/cash_forecast_inference.py", line 469, in _get_forecast_result                                    
                                 (forecast_df.loc * (params['std'] if params['std'] != 0 else 1)) +                                                                    
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/common.py", line 76, in new_method                                        
                                 return method(self, other)                                                                                                            
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/arraylike.py", line 202, in __mul__                                           
                                 return self._arith_method(other, operator.mul)                                                                                        
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/series.py", line 5819, in _arith_method                                       
                                 return base.IndexOpsMixin._arith_method(self, other, op)                                                                              
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/base.py", line 1381, in _arith_method                                         
                                 result = ops.arithmetic_op(lvalues, rvalues, op)                                                                                      
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 285, in arithmetic_op                                 
                                 res_values = _na_arithmetic_op(left, right, op)  # type: ignore                                                                       
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 229, in                                               
                             _na_arithmetic_op                                                                                                                         
                                 result = _masked_arith_op(left, right, op)                                                                                            
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 184, in                                               
                             _masked_arith_op                                                                                                                          
                                 result = op(xrav, y)                                                                                                                  
                             TypeError: can't multiply sequence by non-int of type 'float'                                                                             
2024-12-02 04:30:00,143 - CashForecastInference - INFO - Removed temporary directory ./temp/ZM_20241202-041941/
                    INFO     Removed temporary directory ./temp/ZM_20241202-041941/                                                      cash_forecast_inference.py:490
2024-12-02 04:30:00,182 - CashForecastInference - ERROR - Error in inference pipeline: can't multiply sequence by non-int of type 'float'
                    ERROR    Error in inference pipeline: can't multiply sequence by non-int of type 'float'                             cash_forecast_inference.py:245
2024-12-02 04:30:00,192 - CashForecastInference - WARNING - Rolling back scaling operations...
                    WARNING  Rolling back scaling operations...                                                                          cash_forecast_inference.py:618
2024-12-02 04:30:00,209 - CashForecastInference - INFO - Scaling rollback successful. Data saved to ./cs_data/cash/ZM_20241202-041941/rollback_20241202-041941.csv
                    INFO     Scaling rollback successful. Data saved to ./cs_data/cash/ZM_20241202-041941/rollback_20241202-041941.csv   cash_forecast_inference.py:625
2024-12-02 04:30:00,221 - CashForecastInference - ERROR - Failed to process ZM: can't multiply sequence by non-int of type 'float'
                    ERROR    Failed to process ZM: can't multiply sequence by non-int of type 'float'                                    cash_forecast_inference.py:742
2024-12-02 04:30:00,228 - CashForecastInference - ERROR - Traceback:
Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 220, in _na_arithmetic_op
    result = func(left, right)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/computation/expressions.py", line 242, in evaluate
    return _evaluate(op, op_str, a, b)  # type: ignore[misc]
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/computation/expressions.py", line 73, in _evaluate_standard
    return op(a, b)
TypeError: can't multiply sequence by non-int of type 'float'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/sagemaker-user/cs_scripts/cash_forecast_inference.py", line 733, in main
    inference_pipeline.run_inference(
  File "/home/sagemaker-user/cs_scripts/cash_forecast_inference.py", line 232, in run_inference
    forecast_df = self._get_forecast_result(country_code, inference_df)
  File "/home/sagemaker-user/cs_scripts/cash_forecast_inference.py", line 469, in _get_forecast_result
    (forecast_df.loc[mask, quantile] * (params['std'] if params['std'] != 0 else 1)) +
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/common.py", line 76, in new_method
    return method(self, other)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/arraylike.py", line 202, in __mul__
    return self._arith_method(other, operator.mul)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/series.py", line 5819, in _arith_method
    return base.IndexOpsMixin._arith_method(self, other, op)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/base.py", line 1381, in _arith_method
    result = ops.arithmetic_op(lvalues, rvalues, op)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 285, in arithmetic_op
    res_values = _na_arithmetic_op(left, right, op)  # type: ignore[arg-type]
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 229, in _na_arithmetic_op
    result = _masked_arith_op(left, right, op)
  File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 184, in _masked_arith_op
    result[mask] = op(xrav[mask], y)
TypeError: can't multiply sequence by non-int of type 'float'
                    ERROR    Traceback:                                                                                                  cash_forecast_inference.py:743
                             Traceback (most recent call last):                                                                                                        
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 220, in                                               
                             _na_arithmetic_op                                                                                                                         
                                 result = func(left, right)                                                                                                            
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/computation/expressions.py", line 242, in                                     
                             evaluate                                                                                                                                  
                                 return _evaluate(op, op_str, a, b)  # type: ignore                                                                                    
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/computation/expressions.py", line 73, in                                      
                             _evaluate_standard                                                                                                                        
                                 return op(a, b)                                                                                                                       
                             TypeError: can't multiply sequence by non-int of type 'float'                                                                             
                                                                                                                                                                       
                             During handling of the above exception, another exception occurred:                                                                       
                                                                                                                                                                       
                             Traceback (most recent call last):                                                                                                        
                               File "/home/sagemaker-user/cs_scripts/cash_forecast_inference.py", line 733, in main                                                    
                                 inference_pipeline.run_inference(                                                                                                     
                               File "/home/sagemaker-user/cs_scripts/cash_forecast_inference.py", line 232, in run_inference                                           
                                 forecast_df = self._get_forecast_result(country_code, inference_df)                                                                   
                               File "/home/sagemaker-user/cs_scripts/cash_forecast_inference.py", line 469, in _get_forecast_result                                    
                                 (forecast_df.loc * (params['std'] if params['std'] != 0 else 1)) +                                                                    
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/common.py", line 76, in new_method                                        
                                 return method(self, other)                                                                                                            
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/arraylike.py", line 202, in __mul__                                           
                                 return self._arith_method(other, operator.mul)                                                                                        
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/series.py", line 5819, in _arith_method                                       
                                 return base.IndexOpsMixin._arith_method(self, other, op)                                                                              
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/base.py", line 1381, in _arith_method                                         
                                 result = ops.arithmetic_op(lvalues, rvalues, op)                                                                                      
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 285, in arithmetic_op                                 
                                 res_values = _na_arithmetic_op(left, right, op)  # type: ignore                                                                       
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 229, in                                               
                             _na_arithmetic_op                                                                                                                         
                                 result = _masked_arith_op(left, right, op)                                                                                            
                               File "/opt/conda/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 184, in                                               
                             _masked_arith_op                                                                                                                          
                                 result = op(xrav, y)                                                                                                                  
                             TypeError: can't multiply sequence by non-int of type 'float'                                                                             
2024-12-02 04:30:00,257 - CashForecastInference - INFO - 
Inference pipeline completed
                    INFO                                                                                                                 cash_forecast_inference.py:746
                             Inference pipeline completed                                                                                                              
2024-12-02 04:30:00,269 - CashForecastInference - INFO - Cleaning up resources...
                    INFO     Cleaning up resources...                                                                                    cash_forecast_inference.py:754
sagemaker-user@default:~$ 

This is how the inference script is called:

python cs_scripts/cash_forecast_inference.py   --model_timestamp 20241201-153323   --effective_date 2024-07-15   --config cs_scripts/config.yaml   --countries ZM   --model_name ZM-model-20241201-153323   --inference_template ./cs_data/cash/ZM_custom_inference/ZM_inference_custom.csv

The problem however is that final_forecast.csv is empty

Apply your mind and focus on the most likely solution to this vexing problem.

Be precise and professional in your response.
Use your knowledge and experience to provide a solution.
How would you proceed?

