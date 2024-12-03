# cash_forecast_model.py
# python cs_scripts/cash_forecast_model.py --config cs_scripts/config.yaml --countries ZM

import argparse
import os
import json
from time import gmtime, strftime

from sagemaker import Session
import boto3

import logging

from cash_forecast_lib import (
    setup_custom_logging,
    upload_file_to_s3,
    prepare_training_data,
    calculate_scaling_parameters,
    apply_scaling,
    create_output_directory,
    save_scaling_params
)
from common import Config, load_and_validate_config

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
    logger = setup_custom_logging('CashForecast', timestamp)

    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=config.region)

    for country_code in args.countries:
        logger.info(f"Processing country: {country_code}")
        try:
            # Determine input file path
            input_file = args.input_file if args.input_file else f"./data/cash/{country_code}.csv"
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            # Prepare data
            required_columns = ['ProductId', 'BranchId', 'Currency', 'EffectiveDate', 'Demand']
            categorical_columns = ['ProductId', 'BranchId', 'Currency']
            data = prepare_training_data(input_file, required_columns, categorical_columns, logger)

            # Calculate scaling parameters
            group_columns = ['Currency', 'BranchId']
            target_column = 'Demand'
            scaling_params = calculate_scaling_parameters(data, group_columns, target_column, logger)

            # Apply scaling
            scaled_data = apply_scaling(data, scaling_params, group_columns, target_column, logger)

            # Create output directory
            base_output_path = "./cs_data/cash/"
            output_dir = create_output_directory(base_output_path, country_code, timestamp)

            # Save scaling parameters
            scaling_params_file = os.path.join(output_dir, f"{country_code}_scaling_params.json")
            save_scaling_params(scaling_params, scaling_params_file)
            logger.info(f"Scaling parameters saved to {scaling_params_file}")

            # Save scaled training data
            train_file = os.path.join(output_dir, f"{country_code}_train.csv")
            scaled_data.to_csv(train_file, index=False)
            logger.info(f"Scaled training data saved to {train_file}")

            # Create inference template with unique combinations
            inference_template = data.drop_duplicates(subset=['ProductId', 'BranchId', 'Currency'])[['ProductId', 'BranchId', 'Currency']]
            inference_template_file = os.path.join(output_dir, f"{country_code}_inference_template.csv")
            inference_template.to_csv(inference_template_file, index=False)
            logger.info(f"Inference template saved to {inference_template_file}")

            # Upload training data and scaling parameters to S3
            s3_prefix = f"{config.prefix}-{country_code}/{timestamp}"
            s3_train_key = f"{s3_prefix}/train/{os.path.basename(train_file)}"
            upload_file_to_s3(train_file, s3_train_key, s3_client, config.bucket, logger)
            logger.info(f"Uploaded training data to s3://{config.bucket}/{s3_train_key}")

            s3_scaling_params_key = f"{s3_prefix}/scaling/{country_code}_scaling_params.json"
            upload_file_to_s3(scaling_params_file, s3_scaling_params_key, s3_client, config.bucket, logger, overwrite=True)
            logger.info(f"Uploaded scaling parameters to s3://{config.bucket}/{s3_scaling_params_key}")

            s3_inference_template_key = f"{s3_prefix}/inference/{os.path.basename(inference_template_file)}"
            upload_file_to_s3(inference_template_file, s3_inference_template_key, s3_client, config.bucket, logger, overwrite=True)
            logger.info(f"Uploaded inference template to s3://{config.bucket}/{s3_inference_template_key}")

            # Initialize SageMaker session and client
            sm_client = boto3.client('sagemaker', region_name=config.region)
            sm_session = Session()

            # Define AutoML job configuration
            job_name = f"{country_code}-ts-{timestamp}"
            automl_config = {
                'AutoMLJobName': job_name,
                'AutoMLJobInputDataConfig': [{
                    'ChannelType': 'training',
                    'ContentType': 'text/csv;header=present',
                    'CompressionType': 'None',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f"s3://{config.bucket}/{s3_prefix}/train/"
                        }
                    }
                }],
                'OutputDataConfig': {
                    'S3OutputPath': f"s3://{config.bucket}/{s3_prefix}/output"
                },
                'AutoMLProblemTypeConfig': {
                    'TimeSeriesForecastingJobConfig': {
                        'ForecastFrequency': config.forecast_frequency,
                        'ForecastHorizon': config.forecast_horizon,
                        'ForecastQuantiles': config.quantiles,
                        'TimeSeriesConfig': {
                            'TargetAttributeName': 'Demand',
                            'TimestampAttributeName': 'EffectiveDate',
                            'ItemIdentifierAttributeName': 'ProductId',
                            'GroupingAttributeNames': ['BranchId', 'Currency']
                        }
                    }
                },
                'RoleArn': config.role_arn
            }

            # Start AutoML job
            logger.info(f"Starting AutoML job: {job_name}")
            try:
                sm_client.create_auto_ml_job_v2(**automl_config)
                logger.info(f"AutoML job {job_name} initiated successfully.")
            except Exception as e:
                logger.error(f"Failed to start AutoML job: {e}")
                raise

            # TODO: Implement job monitoring and model retrieval as needed
            # This can be further abstracted into the library if desired

        except Exception as e:
            logger.error(f"Failed to process {country_code}: {e}")
            logger.debug("Exception details:", exc_info=True)

    logger.info("Cash Forecasting Model Pipeline completed successfully.")

if __name__ == "__main__":
    main()
