# reuse_cash_forecast_model.py

import argparse
import boto3
import pandas as pd
import numpy as np
import datetime
import os
import logging
import sys
from time import gmtime, strftime, sleep
from sagemaker import Session
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import yaml
import tempfile
import glob
from pydantic import BaseModel, ValidationError


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
    quantiles: List[str] = ('p10', 'p50', 'p90')


class ModelReusePipeline:
    def __init__(self, config: Config):
        self.config = config
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=config.region)
        self.s3_client = boto3.client('s3')
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = config.role_arn
        self._setup_logging()
        self.output_dir = None

    def _setup_logging(self):
        """Configure logging for the pipeline."""
        self.logger = logging.getLogger('CashForecast')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(f'cash_forecast_reuse_{self.timestamp}.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def prepare_data(self, input_file: str, country_code: str) -> Tuple[str, str]:
        """Prepare data for inference."""
        data = pd.read_csv(input_file)
        required_columns = ['ProductId', 'BranchId', 'CountryCode', 'Currency', 'EffectiveDate', 'Demand']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Input data is missing required column: {col}")

        data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate']).dt.tz_localize(None)
        data.sort_values('EffectiveDate', inplace=True)

        self.output_dir = f"./cs_data/cash/{country_code}_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        data_file = os.path.join(self.output_dir, f"{country_code}_data.csv")
        inference_template_file = os.path.join(self.output_dir, f"{country_code}_inference_template.csv")

        data.to_csv(data_file, index=False)

        inference_template = data.drop_duplicates(
            subset=['ProductId', 'BranchId', 'CountryCode', 'Currency']
        )[['ProductId', 'BranchId', 'CountryCode', 'Currency']]

        inference_template.to_csv(inference_template_file, index=False)

        return data_file, inference_template_file

    def generate_forecasts(self, country_code: str, model_name: str, data_file: str,
                           template_file: str) -> pd.DataFrame:
        """Generate forecasts using existing model."""
        inference_template = pd.read_csv(template_file)
        data = pd.read_csv(data_file)
        data['EffectiveDate'] = pd.to_datetime(data['EffectiveDate'])

        effective_dates = sorted(data['EffectiveDate'].unique())

        batch_number = 0
        for i in range(0, len(effective_dates), self.config.batch_size):
            batch_number += 1
            batch_dates = effective_dates[i:i + self.config.batch_size]
            inference_data_list = []

            for effective_date in batch_dates:
                future_dates = [effective_date + pd.Timedelta(days=j)
                                for j in range(1, self.config.forecast_horizon + 1)]

                num_combinations = len(inference_template)
                temp_df = pd.DataFrame({
                    'ProductId': np.tile(inference_template['ProductId'].values, self.config.forecast_horizon),
                    'BranchId': np.tile(inference_template['BranchId'].values, self.config.forecast_horizon),
                    'CountryCode': np.tile(inference_template['CountryCode'].values, self.config.forecast_horizon),
                    'Currency': np.tile(inference_template['Currency'].values, self.config.forecast_horizon),
                    'EffectiveDate': effective_date,
                    'ForecastDate': np.repeat(future_dates, num_combinations),
                    'Demand': np.nan
                })
                inference_data_list.append(temp_df)

            inference_data = pd.concat(inference_data_list, ignore_index=True)

            # Process batch
            forecast_batch = self._process_forecast_batch(
                country_code, model_name, inference_data, batch_number)

            if batch_number == 1:
                all_forecasts = forecast_batch
            else:
                all_forecasts = pd.concat([all_forecasts, forecast_batch], ignore_index=True)

        self._save_results(country_code, all_forecasts, data)
        return all_forecasts

    def _process_forecast_batch(self, country_code: str, model_name: str,
                                inference_data: pd.DataFrame, batch_number: int) -> pd.DataFrame:
        """Process a single batch of forecasts."""
        # Save batch to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            inference_data.to_csv(tmp_file.name, index=False)

        try:
            # Run transform job
            transform_job_name = f"{model_name}-transform-{self.timestamp}-{batch_number}"

            self.sm_client.create_transform_job(
                TransformJobName=transform_job_name,
                ModelName=model_name,
                BatchStrategy='MultiRecord',
                TransformInput={
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/"
                                     f"{self.timestamp}/inference/batch_{batch_number}.csv"
                        }
                    },
                    'ContentType': 'text/csv',
                    'SplitType': 'Line'
                },
                TransformOutput={
                    'S3OutputPath': f"s3://{self.config.bucket}/{self.config.prefix}-{country_code}/"
                                    f"{self.timestamp}/inference-output/",
                    'AssembleWith': 'Line'
                },
                TransformResources={
                    'InstanceType': self.config.instance_type,
                    'InstanceCount': self.config.instance_count
                }
            )

            # Monitor job
            while True:
                response = self.sm_client.describe_transform_job(
                    TransformJobName=transform_job_name)
                status = response['TransformJobStatus']
                self.logger.info(f"Transform job {transform_job_name} status: {status}")
                if status in ['Completed', 'Failed', 'Stopped']:
                    break
                sleep(30)

            if status != 'Completed':
                raise RuntimeError(f"Transform job failed with status: {status}")

            # Get results
            results = self._get_batch_results(country_code, batch_number)
            return results

        finally:
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)

    def _get_batch_results(self, country_code: str, batch_number: int) -> pd.DataFrame:
        """Retrieve and process batch results."""
        output_prefix = (f"{self.config.prefix}-{country_code}/{self.timestamp}/"
                         f"inference-output/batch_{batch_number}")

        response = self.s3_client.list_objects_v2(
            Bucket=self.config.bucket,
            Prefix=output_prefix
        )

        if 'Contents' not in response:
            raise FileNotFoundError(f"No results found for batch {batch_number}")

        # Process results
        results_list = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.out'):
                with tempfile.NamedTemporaryFile(mode='rb', delete=False) as tmp_file:
                    self.s3_client.download_fileobj(
                        self.config.bucket, obj['Key'], tmp_file)
                    df = pd.read_csv(tmp_file.name)
                    results_list.append(df)
                os.remove(tmp_file.name)

        if not results_list:
            raise ValueError(f"No valid results found for batch {batch_number}")

        return pd.concat(results_list, ignore_index=True)

    def _save_results(self, country_code: str, forecasts: pd.DataFrame,
                      historical_data: pd.DataFrame) -> None:
        """Save final results."""
        output_file = (f"./results/{country_code}_reuse_{self.timestamp}/"
                       f"final_forecast.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Merge forecasts with historical data
        final_df = pd.merge(
            historical_data,
            forecasts,
            on=['ProductId', 'BranchId', 'CountryCode', 'Currency', 'EffectiveDate'],
            how='left'
        )

        final_df.to_csv(output_file, index=False)
        self.logger.info(f"Final forecast saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Cash Forecast Model Reuse Pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--countries', nargs='+', default=['ZM'],
                        help='Country codes to process')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of existing model to use')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    try:
        config_model = ConfigModel(**config_dict)
        config = Config(**config_model.dict())
    except Exception as e:
        logging.error(f"Configuration error: {str(e)}")
        sys.exit(1)

    pipeline = ModelReusePipeline(config)

    for country_code in args.countries:
        try:
            input_file = f"./data/cash/{country_code}.csv"
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            data_file, template_file = pipeline.prepare_data(input_file, country_code)
            pipeline.generate_forecasts(country_code, args.model_name,
                                        data_file, template_file)

        except Exception as e:
            logging.error(f"Failed to process {country_code}: {str(e)}")
            raise


if __name__ == "__main__":
    main()