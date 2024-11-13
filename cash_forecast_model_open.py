import argparse
import boto3
import pandas as pd
import datetime
import os
import logging
import sys
from time import gmtime, strftime, sleep
from sagemaker import get_execution_role, Session
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import yaml
import tempfile
import numpy as np  # Import numpy for array operations


@dataclass
class Config:
    """Configuration parameters for the forecasting pipeline."""
    region: str
    bucket: str
    prefix: str
    instance_type: str = "ml.m5.large"
    instance_count: int = 1
    forecast_horizon: int = 10  # Set to 10 for ten-day ahead forecasts
    forecast_frequency: str = "1D"


class CashForecastingPipeline:
    def __init__(self, config: Config):
        """Initialize the forecasting pipeline with configuration."""
        self.config = config
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=config.region)
        self.s3_client = boto3.client('s3')
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = get_execution_role()
        self._setup_logging()
        # Placeholder for train file path (used later in saving forecasts)
        self.train_file = None

    def _setup_logging(self):
        """Configure logging for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'cash_forecast_{self.timestamp}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('CashForecast')

    def _ensure_column_types(self,
                             data: pd.DataFrame,
                             columns: List[str],
                             prefixes: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Ensure column types and add prefixes where specified."""
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' missing.")
            prefix = (prefixes or {}).get(col, '')
            data[col] = prefix + data[col].astype(str)
        return data

    def _safe_s3_upload(self,
                        local_path: str,
                        s3_key: str,
                        overwrite: bool = False) -> None:
        """Safely upload file to S3 with existence check."""
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

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str]:
        """Prepare data for training and create inference template."""
        # Load and preprocess data
        data = pd.read_csv(input_file)
        data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate'])
        data.sort_values('EffectiveDate', inplace=True)

        # Create directories for output
        output_dir = f"./cs_data/cash/{country_code}_{self.timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Define file paths
        train_file = os.path.join(output_dir, f"{country_code}_train.csv")
        inference_template_file = os.path.join(output_dir, f"{country_code}_inference_template.csv")

        # Save training data
        data.to_csv(train_file, index=False)
        self.train_file = train_file  # Save train file path for later use

        # Create inference template with unique combinations
        inference_template = data.drop_duplicates(
            subset=['ProductId', 'BranchId', 'CountryCode', 'Currency']
        )[['ProductId', 'BranchId', 'CountryCode', 'Currency']]
        inference_template.to_csv(inference_template_file, index=False)

        return train_file, inference_template_file

    def train_model(self, country_code: str, train_data_s3_uri: str) -> str:
        """Train the forecasting model with updated forecast horizon."""
        self.logger.info(f"Starting model training for {country_code}")

        job_name = f"{country_code}-ts-{self.timestamp}"

        # Configure AutoML job with updated forecast horizon
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
                    'ForecastQuantiles': ['0.1', '0.5', '0.9'],  # Quantiles should be in decimal form
                    'TimeSeriesConfig': {
                        'TargetAttributeName': 'Demand',
                        'TimestampAttributeName': 'EffectiveDate',
                        'ItemIdentifierAttributeName': 'ProductId',
                        'GroupingAttributeNames': ['BranchId', 'CountryCode', 'Currency']
                    },
                    'HolidayConfig': [{'CountryCode': country_code}]
                }
            },
            'RoleArn': self.role
        }

        # Start training
        self.sm_client.create_auto_ml_job_v2(**automl_config)
        return job_name

    def _monitor_job(self, job_name: str) -> str:
        """Monitor the AutoML job until completion."""
        self.logger.info(f"Monitoring job {job_name}")
        while True:
            response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
            status = response['AutoMLJobStatus']
            self.logger.info(f"Job {job_name} status: {status}")
            if status in ['Completed', 'Failed', 'Stopped']:
                break
            sleep(120)
        return status

    def _get_best_model(self, job_name: str, country_code: str) -> str:
        """Retrieve the best model from the AutoML job."""
        response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
        best_candidate = response['BestCandidate']
        model_name = f"{country_code}-model-{self.timestamp}"

        # Check if model already exists
        existing_models = self.sm_client.list_models(NameContains=model_name)
        if existing_models['Models']:
            raise ValueError(f"A model with name {model_name} already exists. Aborting to prevent overwriting.")

        # Create model
        self.sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=self.role,
            Containers=best_candidate['InferenceContainers']
        )
        self.logger.info(f"Created model {model_name}")
        return model_name

    def forecast(self, country_code: str, model_name: str, template_file: str) -> pd.DataFrame:
        """Generate multi-step forecasts starting from each EffectiveDate in the training data."""
        # Load inference template
        inference_template = pd.read_csv(template_file)
        inference_template['EffectiveDate'] = pd.to_datetime(inference_template['EffectiveDate'])

        # Load training data to get all EffectiveDates
        training_data = pd.read_csv(self.train_file)
        training_data['EffectiveDate'] = pd.to_datetime(training_data['EffectiveDate'])

        # Get unique EffectiveDates
        effective_dates = training_data['EffectiveDate'].unique()

        # Prepare inference data
        inference_data_list = []

        for effective_date in effective_dates:
            # Generate future dates for the forecast horizon
            future_dates = [effective_date + pd.Timedelta(days=i) for i in range(1, self.config.forecast_horizon + 1)]

            # Create inference data for each combination and future dates
            temp_df = pd.DataFrame({
                'ProductId': np.repeat(inference_template['ProductId'].values, self.config.forecast_horizon),
                'BranchId': np.repeat(inference_template['BranchId'].values, self.config.forecast_horizon),
                'CountryCode': np.repeat(inference_template['CountryCode'].values, self.config.forecast_horizon),
                'Currency': np.repeat(inference_template['Currency'].values, self.config.forecast_horizon),
                'EffectiveDate': effective_date,
                'ForecastDate': np.tile(future_dates, len(inference_template)),
                'Demand': np.nan  # Demand is unknown for future dates
            })

            inference_data_list.append(temp_df)

        # Combine all inference data
        inference_data = pd.concat(inference_data_list, ignore_index=True)

        # Save to CSV
        inference_file = f"./data/inference_{country_code}_{self.timestamp}.csv"
        os.makedirs(os.path.dirname(inference_file), exist_ok=True)
        inference_data.to_csv(inference_file, index=False)

        # Upload to S3
        s3_inference_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference/{os.path.basename(inference_file)}"
        self._safe_s3_upload(inference_file, s3_inference_key, overwrite=True)
        s3_inference_data_uri = f"s3://{self.config.bucket}/{s3_inference_key}"

        # Run batch transform
        self._run_batch_transform_job(country_code, model_name, s3_inference_data_uri)

        # Retrieve and process forecast results
        forecast_df = self._get_forecast_result(country_code)

        # Merge with actual data
        self._save_forecasts(country_code, forecast_df)

        return forecast_df

    def _run_batch_transform_job(self, country_code: str, model_name: str, s3_inference_data_uri: str) -> None:
        """Run a batch transform job for the given inference data."""
        transform_job_name = f"{model_name}-transform-{self.timestamp}"

        output_s3_uri = f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/"

        # Start batch transform job
        self.sm_client.create_transform_job(
            TransformJobName=transform_job_name,
            ModelName=model_name,
            BatchStrategy='MultiRecord',
            TransformInput={
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': s3_inference_data_uri,
                        'S3DataDistributionType': 'ShardedByS3Key'
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

        # Monitor the transform job
        self._monitor_transform_job(transform_job_name)

    def _monitor_transform_job(self, transform_job_name: str) -> None:
        """Monitor the batch transform job until completion."""
        self.logger.info(f"Monitoring transform job {transform_job_name}")
        while True:
            response = self.sm_client.describe_transform_job(TransformJobName=transform_job_name)
            status = response['TransformJobStatus']
            self.logger.info(f"Transform job {transform_job_name} status: {status}")
            if status in ['Completed', 'Failed', 'Stopped']:
                break
            sleep(60)
        if status != 'Completed':
            raise RuntimeError(f"Transform job {transform_job_name} failed with status: {status}")

    def _get_forecast_result(self, country_code: str) -> pd.DataFrame:
        """Download and process forecast results."""
        output_s3_prefix = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/"
        response = self.s3_client.list_objects_v2(Bucket=self.config.bucket, Prefix=output_s3_prefix)

        if 'Contents' not in response:
            raise FileNotFoundError(f"No inference results found in S3 for {output_s3_prefix}")

        # Create a temporary directory to store downloaded files
        temp_dir = f"./temp/{country_code}_{self.timestamp}/"
        os.makedirs(temp_dir, exist_ok=True)

        # Download all output files
        forecast_data = []
        for obj in response['Contents']:
            s3_key = obj['Key']
            if s3_key.endswith('.out'):
                local_file = os.path.join(temp_dir, os.path.basename(s3_key))
                self.s3_client.download_file(self.config.bucket, s3_key, local_file)
                df = pd.read_csv(local_file, header=None)
                forecast_data.append(df)

        if not forecast_data:
            raise FileNotFoundError("No forecast output files found.")

        # Combine all forecast data
        forecast_df = pd.concat(forecast_data, ignore_index=True)

        # The forecast output may not have headers, so assign them
        forecast_df.columns = ['p10', 'p50', 'p90']

        # Load inference data to get identifiers
        inference_file = f"./data/inference_{country_code}_{self.timestamp}.csv"
        inference_data = pd.read_csv(inference_file)

        # Combine inference data with forecasts
        forecast_df = pd.concat(
            [inference_data[['ProductId', 'BranchId', 'CountryCode', 'Currency', 'EffectiveDate', 'ForecastDate']],
             forecast_df], axis=1)

        return forecast_df

    def _save_forecasts(self, country_code: str, forecasts_df: pd.DataFrame) -> None:
        """Save the forecasts with appropriate format."""
        # Required columns
        required_columns = [
            'ProductId', 'BranchId', 'CountryCode', 'Currency',
            'EffectiveDate', 'ForecastDate', 'p10', 'p50', 'p90'
        ]
        for col in required_columns:
            if col not in forecasts_df.columns:
                raise ValueError(f"Column '{col}' is missing from forecasts DataFrame.")

        # Convert date columns to datetime
        forecasts_df['EffectiveDate'] = pd.to_datetime(forecasts_df['EffectiveDate'])
        forecasts_df['ForecastDate'] = pd.to_datetime(forecasts_df['ForecastDate'])

        # Calculate 'ForecastDay' as the difference in days plus 1
        forecasts_df['ForecastDay'] = (forecasts_df['ForecastDate'] - forecasts_df['EffectiveDate']).dt.days + 1

        # Filter out any forecasts that are beyond the forecast horizon
        forecasts_df = forecasts_df[forecasts_df['ForecastDay'] <= self.config.forecast_horizon]

        # Pivot the data
        forecasts_pivot = forecasts_df.pivot_table(
            index=['ProductId', 'BranchId', 'CountryCode', 'Currency', 'EffectiveDate'],
            columns='ForecastDay',
            values=['p10', 'p50', 'p90']
        )

        # Flatten the MultiIndex columns
        forecasts_pivot.columns = [f"{quantile}_Day{int(day)}" for quantile, day in forecasts_pivot.columns]

        # Reset index to turn MultiIndex into columns
        forecasts_pivot.reset_index(inplace=True)

        # Load the historical data (training data)
        historical_data = pd.read_csv(self.train_file)
        historical_data['EffectiveDate'] = pd.to_datetime(historical_data['EffectiveDate'])

        # Ensure necessary columns are present in historical data
        required_actual_columns = [
            'ProductId', 'BranchId', 'CountryCode', 'Currency', 'EffectiveDate', 'Demand'
        ]
        for col in required_actual_columns:
            if col not in historical_data.columns:
                raise ValueError(f"Column '{col}' is missing from historical DataFrame.")

        # Merge actual demand with forecasts
        final_df = pd.merge(
            historical_data,
            forecasts_pivot,
            on=['ProductId', 'BranchId', 'CountryCode', 'Currency', 'EffectiveDate'],
            how='left'
        )

        # Save to CSV
        output_file = f"./results/{country_code}_{self.timestamp}/final_forecast.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_csv(output_file, index=False)
        self.logger.info(f"Final forecast saved to {output_file}")

    def run_pipeline(self, country_code: str, input_file: str) -> None:
        """Run the complete forecasting pipeline with multi-step forecasting."""
        try:
            # Prepare data
            train_file, template_file = self.prepare_data(input_file, country_code)

            # Upload training data to S3
            s3_train_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/train/{os.path.basename(train_file)}"
            self._safe_s3_upload(train_file, s3_train_key)
            train_data_s3_uri = f"s3://{self.config.bucket}/{os.path.dirname(s3_train_key)}/"

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

            # Multi-step Forecasting
            self.forecast(country_code, model_name, template_file)

        except Exception as e:
            self.logger.error(f"Pipeline failed for {country_code}: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Cash Forecasting Pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--countries', nargs='+', default=['ZM'],
                        help='Country codes to process')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)

    # Initialize and run pipeline
    pipeline = CashForecastingPipeline(config)

    for country_code in args.countries:
        try:
            input_file = f"./data/cash/{country_code}.csv"
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            pipeline.run_pipeline(country_code, input_file)

        except Exception as e:
            logging.error(f"Failed to process {country_code}: {str(e)}")


if __name__ == "__main__":
    main()
