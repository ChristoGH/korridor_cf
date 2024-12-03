# cash_forecast_inference.py
# python cs_scripts/cash_forecast_inference.py --model_timestamp 20241201-153323 --effective_date 2024-07-15 --config cs_scripts/config.yaml --countries ZM --model_name ZM-model-20241201-153323 --inference_template ./cs_data/cash/ZM_custom_inference/ZM_inference_custom.csv

import argparse
import os
import json
from time import gmtime, strftime
from datetime import datetime

import boto3
import pandas as pd
import numpy as np
import shutil
import logging

from cash_forecast_lib import (
    setup_custom_logging,
    upload_file_to_s3,
    download_file_from_s3,
    load_scaling_params,
    load_scaling_metadata,
    load_inference_template,
    generate_inference_data,
    validate_scaling_parameters,
    submit_batch_transform_job,
    monitor_batch_transform_job,
    download_forecast_results,
    save_forecasts_locally
)
from common import Config, load_and_validate_config

def main():
    """Main function for cash forecasting inference."""
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

    # Load and validate configuration
    config = load_and_validate_config(args.config)

    # Setup logging
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
    logger = setup_custom_logging('CashForecastInference', timestamp)

    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=config.region)

    # Validate effective_date format
    try:
        effective_date_dt = pd.to_datetime(args.effective_date)
    except ValueError:
        logger.error(f"Invalid effective_date format: {args.effective_date}. Expected format: YYYY-MM-DD")
        raise

    for country_code in args.countries:
        logger.info(f"Processing inference for country: {country_code}")
        try:
            # Define S3 keys for scaling parameters and metadata
            s3_prefix = f"{config.prefix}-{country_code}/{args.model_timestamp}"
            scaling_params_key = f"{s3_prefix}/scaling/{country_code}_scaling_params.json"
            scaling_metadata_key = f"{s3_prefix}/scaling/{country_code}_scaling_metadata.json"

            # Load scaling parameters from S3
            scaling_params = load_scaling_params(s3_client, config.bucket, scaling_params_key, logger)
            logger.info("Scaling parameters loaded successfully.")

            # Validate scaling parameters
            validate_scaling_parameters(scaling_params)
            logger.info("Scaling parameters validated successfully.")

            # Load scaling metadata from S3
            scaling_metadata = load_scaling_metadata(s3_client, config.bucket, scaling_metadata_key, logger)
            logger.info("Scaling metadata loaded successfully.")

            # Load inference template
            inference_template = load_inference_template(args.inference_template, logger)

            # Generate inference data
            forecast_horizon = config.forecast_horizon
            inference_data = generate_inference_data(inference_template, effective_date_dt, forecast_horizon, logger)

            # Create output directory
            base_output_path = "./cs_data/cash/"
            output_dir = create_output_directory(base_output_path, country_code, timestamp)

            # Save inference data locally
            inference_file = os.path.join(output_dir, f"{country_code}_inference.csv")
            inference_data.to_csv(inference_file, index=False)
            logger.info(f"Inference data saved to {inference_file}")

            # Upload inference data to S3
            s3_inference_key = f"{s3_prefix}/inference/{os.path.basename(inference_file)}"
            upload_file_to_s3(inference_file, s3_inference_key, s3_client, config.bucket, logger, overwrite=True)
            logger.info(f"Uploaded inference data to s3://{config.bucket}/{s3_inference_key}")

            # Define Batch Transform job parameters
            model_name = args.model_name
            transform_job_name = f"{model_name}-transform-{timestamp}"
            input_s3_uri = f"s3://{config.bucket}/{s3_inference_key}"
            output_s3_uri = f"s3://{config.bucket}/{s3_prefix}/inference-output/"
            instance_type = config.instance_type
            instance_count = config.instance_count
            content_type = 'text/csv'
            split_type = 'Line'
            accept_type = 'text/csv'

            # Submit Batch Transform job
            submit_batch_transform_job(
                sm_client=boto3.client('sagemaker', region_name=config.region),
                model_name=model_name,
                transform_job_name=transform_job_name,
                input_s3_uri=input_s3_uri,
                output_s3_uri=output_s3_uri,
                instance_type=instance_type,
                instance_count=instance_count,
                content_type=content_type,
                split_type=split_type,
                accept_type=accept_type,
                logger=logger
            )

            # Monitor Batch Transform job
            monitor_batch_transform_job(
                sm_client=boto3.client('sagemaker', region_name=config.region),
                transform_job_name=transform_job_name,
                logger=logger
            )

            # Download forecast results from S3
            local_forecast_dir = os.path.join(output_dir, "forecast_results")
            os.makedirs(local_forecast_dir, exist_ok=True)
            forecast_df = download_forecast_results(
                s3_client=s3_client,
                bucket=config.bucket,
                output_s3_prefix=f"{s3_prefix}/inference-output/",
                local_output_dir=local_forecast_dir,
                logger=logger
            )

            # Process forecast results (Inverse Scaling)
            # Assuming the forecast_df has quantile columns as per config.quantiles
            for (currency, branch), params in scaling_params.items():
                mask = (forecast_df['Currency'] == currency) & (forecast_df['BranchId'] == branch)
                if mask.sum() == 0:
                    logger.warning(f"No forecasts found for Currency={currency}, Branch={branch}")
                    continue

                for quantile in config.quantiles:
                    if quantile in forecast_df.columns:
                        forecast_df.loc[mask, quantile] = (
                            (forecast_df.loc[mask, quantile] * (params.std if params.std != 0 else 1)) +
                            params.mean
                        )
                        logger.info(f"Inverse scaled {quantile} for Currency={currency}, Branch={branch}")
                    else:
                        logger.warning(f"Quantile '{quantile}' not found in forecast data for Currency={currency}, Branch={branch}")

            logger.info("Inverse scaling completed successfully.")

            # Save forecasts locally
            final_forecast_file = os.path.join(output_dir, "final_forecast.csv")
            save_forecasts_locally(
                forecast_df=forecast_df,
                output_file=final_forecast_file,
                quantiles=config.quantiles,
                logger=logger
            )

            logger.info(f"Inference processing completed for country: {country_code}")

        except Exception as e:
            logger.error(f"Failed to process inference for {country_code}: {e}")
            logger.debug("Exception details:", exc_info=True)
            continue  # Proceed with next country

    logger.info("Cash Forecasting Inference Pipeline completed successfully.")

if __name__ == "__main__":
    main()
