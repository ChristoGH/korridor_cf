# cash_forecast_model.py

"""
Cash Forecasting Model Building Script

This script builds and trains forecasting models for cash demand using the provided configuration.
It leverages shared utilities from common.py for configuration management, data processing,
logging, and AWS interactions.

Approaches to Ensure Long-Term Scalability and Accuracy:

1. Remove or Filter Out Unseen Combinations:
   This script generates an inference template from the training dataset.
   Because of this, all (Currency, BranchId) pairs in the inference template
   are guaranteed to have scaling parameters. No new combinations should appear
   at inference time unless the inference script uses a different template. If
   unexpected combinations appear during inference, you would either remove
   those combinations or ensure the inference template only includes known combinations.

2. Retrain or Update the Model:
   If the business requires forecasting for new (Currency, BranchId) pairs not
   present in the training data, you must retrain the model with historical data
   that includes these new combinations. This ensures scaling parameters and
   model parameters cover these new entities.

3. Handle Missing Combinations Gracefully:
   While this training script does not directly handle inference for unseen combinations,
   you could implement logic during inference to skip or assign default scaling parameters
   for unseen combinations, at the cost of accuracy. The code below focuses on ensuring
   that the training pipeline produces all required artifacts for known combinations.

Created model ZM-model-20241206-031017 successfully.
"""
from time import sleep
import argparse
import sys
from pathlib import Path
from time import gmtime, strftime
import logging
import boto3
import pandas as pd
from sagemaker import Session
from typing import Tuple
from sagemaker.automl import AutoMLV2, AutoMLDataChannel
from sagemaker.automl.automlv2 import AutoMLTimeSeriesForecastingConfig
from sagemaker.transformer import Transformer
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.steps import ProcessingStep, TransformStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet

from common import (
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
        self.output_dir = None  # Will be set in run_pipeline

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str]:
        """
        Prepare data for training with enhanced sorting as per AWS blog best practices.

        The sorting process follows these steps:
        1. Convert timestamp to proper datetime format
        2. Validate timestamp consistency
        3. Multi-level sort by identifiers and timestamp
        4. Verify sort integrity
        5. Handle any gaps in time series
        """
        self.logger.info(f"Preparing data for country: {country_code}")

        # Load data using DataProcessor
        data = self.data_processor.load_data(input_file)

        # 1. Convert timestamp to datetime with validation
        try:
            data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate'])
            self.logger.info("Timestamp conversion successful")
        except Exception as e:
            self.logger.error(f"Failed to convert timestamps: {e}")
            raise ValueError("Timestamp conversion failed. Please ensure timestamp format is consistent.")

        # 2. Validate timestamp consistency
        timestamp_freq = pd.infer_freq(data['EffectiveDate'].sort_values())
        if timestamp_freq is None:
            self.logger.warning("Could not infer consistent timestamp frequency. Check for irregular intervals.")

        # 3. Multi-level sort implementation
        try:
            data = data.sort_values(
                by=['ProductId', 'BranchId', 'Currency', 'EffectiveDate'],
                ascending=[True, True, True, True],
                na_position='first'  # Handle any NAs consistently
            )
            self.logger.info("Multi-level sort completed successfully")
        except Exception as e:
            self.logger.error(f"Sorting failed: {e}")
            raise

        # 4. Verify sort integrity
        for group in data.groupby(['ProductId', 'BranchId', 'Currency']):
            group_data = group[1]  # group[0] is the key, group[1] is the data

            # Check if timestamps are strictly increasing within group
            if not group_data['EffectiveDate'].is_monotonic_increasing:
                self.logger.error(f"Non-monotonic timestamps found in group {group[0]}")
                raise ValueError(f"Time series integrity violated in group {group[0]}")

            # Check for duplicates
            duplicates = group_data.duplicated(subset=['EffectiveDate'], keep=False)
            if duplicates.any():
                self.logger.warning(f"Duplicate timestamps found in group {group[0]}")
                # Log the duplicates for investigation
                self.logger.warning(f"Duplicate records:\n{group_data[duplicates]}")

        # 5. Handle gaps in time series
        groups_with_gaps = []
        for group in data.groupby(['ProductId', 'BranchId', 'Currency']):
            group_data = group[1]
            expected_dates = pd.date_range(
                start=group_data['EffectiveDate'].min(),
                end=group_data['EffectiveDate'].max(),
                freq=timestamp_freq
            )
            if len(expected_dates) != len(group_data):
                groups_with_gaps.append(group[0])

        if groups_with_gaps:
            self.logger.warning(f"Found gaps in time series for groups: {groups_with_gaps}")

        self.logger.info(f"Data preparation completed. Processed {len(data)} records across "
                         f"{len(data.groupby(['ProductId', 'BranchId', 'Currency']))} groups")

        # Proceed with splitting and scaling as before...
        train_df, test_df = self._split_data(data)
        scaled_data, scaling_params = self.data_processor.prepare_data(train_df, country_code)

        # Save the data
        train_file = self.output_dir / f"{country_code}_train.csv"
        test_file = self.output_dir / f"{country_code}_test.csv"

        scaled_data.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        return str(train_file), str(test_file)

    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the sorted data into training and test sets following the blog's methodology.
        """
        train_dfs = []
        test_dfs = []

        for group_key, group_data in data.groupby(['ProductId', 'BranchId', 'Currency']):
            timestamps = group_data['EffectiveDate'].unique()

            if len(timestamps) <= 8:
                self.logger.warning(f"Group {group_key} has insufficient data points (<= 8), skipping")
                continue

            # Split as per blog's specification
            train_end = len(timestamps) - 8
            test_start = len(timestamps) - 8
            test_end = len(timestamps) - 4

            train_mask = group_data['EffectiveDate'] < timestamps[train_end]
            test_mask = (group_data['EffectiveDate'] >= timestamps[test_start]) & \
                        (group_data['EffectiveDate'] < timestamps[test_end])

            train_dfs.append(group_data[train_mask])
            test_dfs.append(group_data[test_mask])

        return pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)

    def upload_training_data(self, train_file: str, country_code: str) -> str:
        """Upload training data to S3 and return the S3 URI."""
        s3_train_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/train/{Path(train_file).name}"
        self.s3_handler.safe_upload(local_path=train_file, bucket=self.config.bucket, s3_key=s3_train_key)
        train_data_s3_uri = f"s3://{self.config.bucket}/{s3_train_key}"
        self.logger.info(f"Training data uploaded to {train_data_s3_uri}")
        return train_data_s3_uri

    def upload_test_data(self, test_file: str, country_code: str) -> str:
        """Upload test data to S3 and return the S3 URI."""
        s3_test_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/test/{Path(test_file).name}"
        self.s3_handler.safe_upload(local_path=test_file, bucket=self.config.bucket, s3_key=s3_test_key)
        test_data_s3_uri = f"s3://{self.config.bucket}/{s3_test_key}"
        self.logger.info(f"Test data uploaded to {test_data_s3_uri}")
        return test_data_s3_uri

    def train_model(self, country_code: str, train_data_s3_uri: str) -> str:
        """Train the forecasting model using SageMaker AutoMLV2 SDK."""
        self.logger.info(f"Starting model training for {country_code}")
        job_name = f"{country_code}-ts-{self.timestamp}"

        time_series_config = AutoMLTimeSeriesForecastingConfig(
            forecast_frequency=self.config.forecast_frequency,
            forecast_horizon=self.config.forecast_horizon,
            forecast_quantiles=self.config.quantiles,
            item_identifier_attribute_name='ProductId',
            target_attribute_name='Demand',
            timestamp_attribute_name='EffectiveDate',
            grouping_attribute_names=['BranchId', 'Currency'],
            filling={
                'imputation_strategy': 'forward_fill',
                'backfill_strategy': 'zero',
            }
        )

        automl_sm_job = AutoMLV2(
            problem_config=time_series_config,
            role=self.role,
            sagemaker_session=self.session,
            base_job_name=job_name,
            output_path=f's3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/output'
        )

        automl_sm_job.fit(
            inputs=[AutoMLDataChannel(s3_data_type='S3Prefix', s3_uri=train_data_s3_uri,
                                      channel_type='training')],
            wait=True,
            logs=True
        )

        return automl_sm_job.best_candidate()['CandidateName']

    def _get_best_model(self, job_name: str, country_code: str) -> str:
        """Retrieve the best model from the AutoML job."""
        self.logger.info(f"Retrieving best model for job: {job_name}")

        automl_sm_job = AutoMLV2.attach(
            auto_ml_job_name=job_name,
            sagemaker_session=self.session
        )
        best_candidate = automl_sm_job.best_candidate()
        model_name = f"{country_code}-model-{self.timestamp}"

        # Create SageMaker model from best candidate
        automl_sm_model = automl_sm_job.create_model(name=model_name, candidate=best_candidate)
        self.logger.info(f"Created model {model_name} successfully.")

        return model_name

    def _run_batch_transform_job(self, country_code: str, model_name: str, s3_inference_data_uri: str,
                                 batch_number: int) -> None:
        """Run a batch transform job for the given inference data."""
        transform_job_name = f"{model_name}-transform-{self.timestamp}-{batch_number}"
        output_s3_uri = f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/inference-output/"

        model = self.session.sagemaker_client.describe_model(ModelName=model_name)
        transformer = Transformer(
            model_name=model_name,
            instance_count=self.config.instance_count,
            instance_type=self.config.instance_type,
            output_path=output_s3_uri,
            sagemaker_session=self.session
        )

        transformer.transform(
            data=s3_inference_data_uri,
            data_type='S3Prefix',
            content_type='text/csv',
            split_type='Line',
            job_name=transform_job_name,
            wait=True
        )
        self.logger.info(f"Created transform job {transform_job_name} successfully.")

    def _create_evaluation_pipeline(self, country_code: str, model_name: str, s3_inference_data_uri: str,
                                    test_s3_uri: str) -> Pipeline:
        """Creates a SageMaker Pipeline for model evaluation and conditional registration."""

        # Define parameters for pipeline execution
        input_data = ParameterString(name="InputData", default_value=s3_inference_data_uri)
        test_data = ParameterString(name="TestData", default_value=test_s3_uri)
        model_name_param = ParameterString(name="ModelName", default_value=model_name)
        instance_type_param = ParameterString(name="InstanceType", default_value=self.config.instance_type)
        instance_count_param = ParameterInteger(name="InstanceCount", default_value=self.config.instance_count)
        evaluation_metric_threshold = ParameterInteger(name="EvalThreshold",
                                                       default_value=self.config.evaluation_threshold)

        # Batch Transform Step
        transformer = Transformer(
            model_name=model_name_param,
            instance_count=instance_count_param,
            instance_type=instance_type_param,
            output_path=f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/{self.timestamp}/batch-output",
            sagemaker_session=self.session
        )
        transform_step = TransformStep(
            name="BatchTransform",
            transformer=transformer,
            inputs={
                "input_data": input_data,
                "content_type": "text/csv",
                "split_type": "Line"
            },
        )

        # Evaluation Step
        evaluation_processor = ScriptProcessor(
            role=self.role,
            image_uri=self.config.evaluation_image_uri,  # EVAL_IMAGE_URI
            command=["python3"],
            instance_type=instance_type_param,
            instance_count=instance_count_param,
            sagemaker_session=self.session,
        )

        evaluation_step = ProcessingStep(
            name="ModelEvaluation",
            processor=evaluation_processor,
            inputs=[
                ProcessingInput(source=transform_step.properties.TransformOutput.S3OutputPath,
                                input_name="predictions",
                                s3_data_type="S3Prefix"),
                ProcessingInput(source=test_data,
                                input_name="test_data",
                                s3_data_type="S3Prefix")
            ],
            outputs=[
                ProcessingOutput(output_name="evaluation",
                                 source=f"/opt/ml/processing/output",
                                 s3_upload_mode="EndOfJob")
            ],
        )

        # Conditional Model Registration
        register_step = RegisterModel(
            name="ConditionalRegisterModel",
            estimator=None,
            model_data=None,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=[instance_type_param],
            transform_instances=[instance_type_param],
            model_package_group_name=f"{country_code}-model-group",
            model_metrics={
                "ModelQuality": {
                    "Statistics": {
                        "S3Uri": evaluation_step.properties.ProcessingOutputConfig.Outputs[
                                     "evaluation"
                                 ].S3OutputPath + "/evaluation_metrics.json"
                    }
                }
            },
            approval_status="PendingManualApproval",
        )

        # Define condition based on eval metrics
        cond_lte = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step=evaluation_step,
                property_file=f"{evaluation_step.properties.ProcessingOutputConfig.Outputs['evaluation'].S3OutputPath}/evaluation_metrics.json",
                json_path="metrics.mae.value"
            ),
            right=evaluation_metric_threshold
        )

        # Model registration step
        register_step.add_condition_step(cond_lte)

        # Build pipeline
        pipeline_name = f"cash-forecast-pipeline-{self.timestamp}-{country_code}"
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=[input_data, test_data, model_name_param, instance_type_param, instance_count_param,
                        evaluation_metric_threshold],
            steps=[transform_step, evaluation_step, register_step],
            sagemaker_session=self.session
        )

        return pipeline

    def run_pipeline(self, country_code: str, input_file: str, backtesting: bool = False) -> None:
        """
        Run the model building pipeline.

        This method handles data preparation, uploads training data to S3,
        initiates model training via SageMaker AutoML, monitors the training job,
        and retrieves the best model.

        Args:
            country_code (str): The country code for which to build the model.
            input_file (str): Path to the input CSV file.
            backtesting (bool): Flag to indicate if backtesting is to be performed.

        Raises:
            RuntimeError: If the training job fails to complete successfully.
        """

        try:
            self.logger.info(f"Running pipeline for country: {country_code}")

            # Set output directory
            self.output_dir = Path(f"./data/output/{country_code}/{self.timestamp}")
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Load and prepare data
            train_file, test_file = self.prepare_data(input_file, country_code)

            # Upload training data to S3
            train_data_s3_uri = self.upload_training_data(train_file, country_code)
            test_data_s3_uri = self.upload_test_data(test_file, country_code)

            # Train model
            job_name = self.train_model(country_code, train_data_s3_uri)

            # Get best model
            model_name = self._get_best_model(job_name, country_code)

            self.logger.info(f"Model trained successfully. Model Name : {model_name}")

            # Run Batch Transform and evaluation using the SageMaker Pipeline
            pipeline = self._create_evaluation_pipeline(
                country_code=country_code,
                model_name=model_name,
                s3_inference_data_uri=test_data_s3_uri,
                test_s3_uri=test_data_s3_uri
            )
            pipeline.upsert(role_arn=self.role)
            execution = pipeline.start()
            execution.wait(logs=True)

            self.logger.info(f"Model building pipeline completed successfully for {country_code}")

        except Exception as e:
            self.logger.error(f"Pipeline failed for {country_code}: {str(e)}")
            raise


def main():
    # Parse command line arguments without inference-specific arguments
    args = parse_arguments()  # inference=False by default

    # Setup logging
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
    logger = setup_logging(timestamp, name='CashForecastModel')
    logger.info("Starting Cash Forecasting Model Training Pipeline")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("Configuration loaded successfully.")

        # Initialize and run the pipeline
        pipeline = CashForecastingPipeline(config=config, logger=logger)
        pipeline.run_pipeline(
            country_code=args.countries[0],
            input_file=args.input_file or f"./data/cash/{args.countries[0]}.csv",
            backtesting=args.resume
        )

        logger.info("Model training pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()