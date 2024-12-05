# cash_forecast_model.py

"""
Cash Forecasting Model Building Script

This script builds and trains forecasting models for cash demand using the provided configuration.
It leverages shared utilities from common.py for configuration management, data processing,
logging, and AWS interactions.
"""

import argparse
import sys
from pathlib import Path
from time import gmtime, strftime

import boto3
import pandas as pd
import numpy as np
from sagemaker import Session

from common import (
    ConfigModel,
    Config,
    S3Handler,
    DataProcessor,
    setup_logging,
    load_config,
    parse_arguments
)


class CashForecastingPipeline:
    """Pipeline for training and forecasting cash demand models."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize the forecasting pipeline with configuration."""
        self.config = config
        self.logger = logger
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=self.config.region)
        self.s3_handler = S3Handler(region=self.config.region, logger=self.logger)
        self.data_processor = DataProcessor(logger=self.logger)
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = self.config.role_arn
        self.train_file = None
        self.output_dir = None  # Will be set in prepare_data

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str]:
        """Prepare data for training and create inference template."""
        self.logger.info(f"Preparing data for country: {country_code}")

        # Load data using DataProcessor
        data = self.data_processor.load_data(input_file)

        # Prepare and scale data
        scaled_data, scaling_params = self.data_processor.prepare_data(data)

        # Create directories for output
        self.output_dir = Path(f"./cs_data/cash/{country_code}_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save scaling parameters
        scaling_file = self.output_dir / f"{country_code}_scaling_params.json"
        self.data_processor.save_scaling_params(scaling_params, scaling_file)

        # Generate and save scaling metadata
        metadata = self.data_processor.generate_scaling_metadata(data, scaling_params)
        metadata_file = self.output_dir / f"{country_code}_scaling_metadata.json"
        self.data_processor.save_metadata(metadata, metadata_file)

        # Save scaled training data
        train_file = self.output_dir / f"{country_code}_train.csv"
        scaled_data.to_csv(train_file, index=False)
        self.train_file = str(train_file)
        self.logger.info(f"Saved scaled training data to {train_file}")

        # Create and save inference template
        inference_template = data.drop_duplicates(
            subset=['ProductId', 'BranchId', 'Currency']
        )[['ProductId', 'BranchId', 'Currency']]
        inference_template_file = self.output_dir / f"{country_code}_inference_template.csv"
        inference_template.to_csv(inference_template_file, index=False)
        self.logger.info(f"Saved inference template to {inference_template_file}")

        return str(train_file), str(inference_template_file)

    def upload_training_data(self, train_file: str, country_code: str) -> str:
        """Upload training data to S3 and return the S3 URI."""
        s3_train_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/train/{Path(train_file).name}"
        self.s3_handler.safe_upload(local_path=train_file, bucket=self.config.bucket, s3_key=s3_train_key)
        train_data_s3_uri = f"s3://{self.config.bucket}/{s3_train_key}"
        self.logger.info(f"Training data uploaded to {train_data_s3_uri}")
        return train_data_s3_uri

    def train_model(self, country_code: str, train_data_s3_uri: str) -> str:
        """Train the forecasting model using SageMaker AutoML."""
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
            self.logger.info(f"AutoML job {job_name} created successfully.")
        except Exception as e:
            self.logger.error(f"Failed to start AutoML job: {e}")
            raise

        return job_name

    def _monitor_job(self, job_name: str) -> str:
        """Monitor the AutoML job until completion."""
        self.logger.info(f"Monitoring AutoML job: {job_name}")
        sleep_time = 60  # Start with 1 minute

        while True:
            try:
                response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
                status = response['AutoMLJobStatus']
                self.logger.info(f"Job {job_name} status: {status}")

                if status in ['Completed', 'Failed', 'Stopped']:
                    if status != 'Completed':
                        failure_reason = response.get('FailureReason', 'No failure reason provided.')
                        self.logger.error(f"AutoML job {job_name} failed: {failure_reason}")
                    break

                sleep(sleep_time)
                sleep_time = min(sleep_time * 1.5, 600)  # Increase sleep time up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring AutoML job: {e}")
                sleep(60)  # Wait before retrying

        return status

    def _get_best_model(self, job_name: str, country_code: str) -> str:
        """Retrieve and create the best model from the AutoML job."""
        self.logger.info(f"Retrieving best model for job: {job_name}")

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
            self.logger.info(f"Created model {model_name} successfully.")
        except Exception as e:
            self.logger.error(f"Failed to create model {model_name}: {e}")
            raise

        return model_name

    def forecast(self, country_code: str, model_name: str, template_file: str,
                 backtesting: bool = False) -> pd.DataFrame:
        """Generate forecasts using the trained model."""
        self.logger.info(f"Starting forecasting for country: {country_code}")

        # Load inference template using DataProcessor
        inference_template = self.data_processor.load_data(template_file)

        # Retrieve effective dates using DataProcessor
        effective_dates = self.data_processor.get_effective_dates(train_file=self.train_file, backtesting=backtesting)

        # Generate inference data in batches
        batch_size = self.config.batch_size
        forecast_df = pd.DataFrame()

        for batch_number, batch_dates in enumerate(
                self.data_processor.chunk_list(effective_dates, batch_size), start=1):
            self.logger.info(f"Processing batch {batch_number} with {len(batch_dates)} dates.")

            # Generate inference data
            inference_data = self.data_processor.generate_inference_data(
                inference_template=inference_template,
                batch_dates=batch_dates,
                forecast_horizon=self.config.forecast_horizon
            )

            # Save inference data to CSV
            inference_file = self.output_dir / f"{country_code}_inference_batch_{batch_number}.csv"
            inference_data.to_csv(inference_file, index=False)
            self.logger.info(f"Saved inference batch to {inference_file}")

            # Upload inference data to S3 using S3Handler
            s3_inference_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/inference/{inference_file.name}"
            self.s3_handler.safe_upload(local_path=str(inference_file), bucket=self.config.bucket,
                                        s3_key=s3_inference_key, overwrite=True)
            s3_inference_data_uri = f"s3://{self.config.bucket}/{s3_inference_key}"
            self.logger.info(f"Uploaded inference batch to {s3_inference_data_uri}")

            # Run batch transform
            self._run_batch_transform_job(country_code, model_name, s3_inference_data_uri, batch_number)

        # Retrieve and combine forecast results
        forecast_df = self.data_processor.retrieve_forecast_results(
            s3_client=self.s3_handler.client,
            bucket=self.config.bucket,
            prefix=f"{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/",
            output_dir=self.output_dir
        )

        # Load scaling parameters
        scaling_file = self.output_dir / f"{country_code}_scaling_params.json"
        scaling_params = self.data_processor.load_scaling_params(scaling_file=scaling_file)

        # Restore scaling
        restored_forecast_df = self.data_processor.restore_scaling(
            forecast_df=forecast_df,
            scaling_params=scaling_params,
            quantiles=self.config.quantiles,
            logger=self.logger
        )

        # Save final forecasts
        self.data_processor.save_final_forecasts(
            forecasts_df=restored_forecast_df,
            country_code=country_code,
            timestamp=self.timestamp,
            config=self.config,
            logger=self.logger
        )

        self.logger.info(f"Forecasting completed for country: {country_code}")
        return restored_forecast_df

    def _run_batch_transform_job(self, country_code: str, model_name: str, s3_inference_data_uri: str,
                                 batch_number: int) -> None:
        """Run a batch transform job for the given inference data."""
        transform_job_name = f"{model_name}-transform-{self.timestamp}-{batch_number}"
        output_s3_uri = f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/"

        # Start batch transform job using SageMaker
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
            self.logger.info(f"Created transform job {transform_job_name} successfully.")
            self._monitor_transform_job(transform_job_name)
        except Exception as e:
            self.logger.error(f"Failed to create transform job {transform_job_name}: {e}")
            raise

    def _monitor_transform_job(self, transform_job_name: str) -> None:
        """Monitor the batch transform job until completion."""
        self.logger.info(f"Monitoring transform job: {transform_job_name}")
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
                            f"Transform job {transform_job_name} failed with status: {status}. Reason: {failure_reason}"
                        )
                    break

                sleep(sleep_time)
                sleep_time = min(sleep_time * 1.5, 600)  # Increase sleep time up to 10 minutes
            except Exception as e:
                self.logger.error(f"Error monitoring transform job {transform_job_name}: {e}")
                sleep(60)  # Wait before retrying

    def run_pipeline(self, country_code: str, input_file: str, backtesting: bool = False) -> None:
        """Run the complete forecasting pipeline."""
        try:
            self.logger.info(f"Running pipeline for country: {country_code}")

            # Load data using DataProcessor
            data = self.data_processor.load_data(input_file)

            # Prepare and scale data
            scaled_data, scaling_params = self.data_processor.prepare_data(data)

            # Create directories for output
            self.output_dir = Path(f"./cs_data/cash/{country_code}_{self.timestamp}")
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Save scaling parameters
            scaling_file = self.output_dir / f"{country_code}_scaling_params.json"
            self.data_processor.save_scaling_params(scaling_params, scaling_file)

            # Generate and save scaling metadata
            metadata = self.data_processor.generate_scaling_metadata(data, scaling_params)
            metadata_file = self.output_dir / f"{country_code}_scaling_metadata.json"
            self.data_processor.save_metadata(metadata, metadata_file)

            # Save scaled training data
            train_file = self.output_dir / f"{country_code}_train.csv"
            scaled_data.to_csv(train_file, index=False)
            self.train_file = str(train_file)
            self.logger.info(f"Saved scaled training data to {train_file}")

            # Create and save inference template
            inference_template = data.drop_duplicates(
                subset=['ProductId', 'BranchId', 'Currency']
            )[['ProductId', 'BranchId', 'Currency']]
            inference_template_file = self.output_dir / f"{country_code}_inference_template.csv"
            inference_template.to_csv(inference_template_file, index=False)
            self.logger.info(f"Saved inference template to {inference_template_file}")

            # Upload training data to S3
            train_data_s3_uri = self.upload_training_data(str(train_file), country_code)

            # Train model
            job_name = self.train_model(country_code, train_data_s3_uri)

            # Monitor training
            status = self._monitor_job(job_name)
            if status != 'Completed':
                raise RuntimeError(f"Training failed with status: {status}")

            # Get best model
            model_name = self._get_best_model(job_name, country_code)

            # Forecasting
            self.forecast(country_code, model_name, str(inference_template_file), backtesting=backtesting)

            self.logger.info(f"Pipeline completed successfully for {country_code}")

        except Exception as e:
            self.logger.error(f"Pipeline failed for {country_code}: {str(e)}")
            raise


def main():
    """Main entry point for the cash forecasting model script."""
    # Parse command line arguments using common.py's parser
    args = parse_arguments()

    # Load and validate configuration using common.py's load_config
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Configuration error: {str(e)}")
        sys.exit(1)

    # Setup logging using common.py's setup_logging
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
    logger = setup_logging(timestamp)

    # Initialize and run the pipeline for each specified country
    for country_code in args.countries:
        pipeline = CashForecastingPipeline(config=config, logger=logger)
        try:
            # Determine input file path
            input_file = args.input_file if args.input_file else f"./data/cash/{country_code}.csv"
            if not Path(input_file).exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")

            # Run the pipeline
            pipeline.run_pipeline(country_code=country_code, input_file=input_file, backtesting=args.resume)

        except Exception as e:
            logger.error(f"Failed to process {country_code}: {str(e)}")


if __name__ == "__main__":
    main()
