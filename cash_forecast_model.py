import argparse
import boto3
import pandas as pd
import numpy as np
import os
import logging
import sys
import json
from time import gmtime, strftime, sleep
from sagemaker import Session
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import yaml
import glob
from pydantic import BaseModel, ValidationError
import shutil

# Existing imports and classes remain unchanged...

class CashForecastingPipeline:
    def __init__(self, config: Config, state_file: str = "pipeline_state.json"):
        """Initialize the forecasting pipeline with configuration."""
        self.config = config
        self.session = Session()
        self.sm_client = boto3.client('sagemaker', region_name=config.region)
        self.s3_client = boto3.client('s3')
        self.timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
        self.role = config.role_arn
        self._setup_logging()
        self.train_file = None
        self.output_dir = None  # Will be set in prepare_data
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load the pipeline state from a JSON file."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.logger.info(f"Loaded existing state from {self.state_file}")
        else:
            state = {}
            self.logger.info(f"No existing state found. Starting fresh.")
        return state

    def _save_state(self):
        """Save the current pipeline state to a JSON file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)
        self.logger.info(f"State saved to {self.state_file}")

    def run_pipeline(self, country_code: str, input_file: str, backtesting: bool = False) -> None:
        """Run the complete forecasting pipeline with checkpointing."""
        try:
            if country_code not in self.state:
                self.state[country_code] = {}

            # Step 1: Prepare Data
            if not self.state[country_code].get('prepared_data'):
                self.logger.info(f"Preparing data for {country_code}")
                train_file, template_file = self.prepare_data(input_file, country_code)
                self.state[country_code]['prepared_data'] = {
                    'train_file': train_file,
                    'template_file': template_file
                }
                self._save_state()
            else:
                self.logger.info(f"Data already prepared for {country_code}. Skipping.")

            # Step 2: Upload Training Data to S3
            if not self.state[country_code].get('uploaded_train'):
                self.logger.info(f"Uploading training data for {country_code} to S3")
                train_file = self.state[country_code]['prepared_data']['train_file']
                s3_train_key = f"{self.config.prefix}-{country_code}/{self.timestamp}/train/{os.path.basename(train_file)}"
                self._safe_s3_upload(train_file, s3_train_key)
                train_data_s3_uri = f"s3://{self.config.bucket}/{os.path.dirname(s3_train_key)}/"
                self.state[country_code]['uploaded_train'] = {
                    's3_train_uri': train_data_s3_uri,
                    's3_train_key': s3_train_key
                }
                self._save_state()
            else:
                self.logger.info(f"Training data already uploaded for {country_code}. Skipping.")
                train_data_s3_uri = self.state[country_code]['uploaded_train']['s3_train_uri']
                s3_train_key = self.state[country_code]['uploaded_train']['s3_train_key']

            # Step 3: Train Model
            if not self.state[country_code].get('trained_model'):
                self.logger.info(f"Training model for {country_code}")
                job_name = self.train_model(country_code, train_data_s3_uri)
                self.state[country_code]['trained_model'] = {
                    'job_name': job_name
                }
                self._save_state()

                # Step 4: Monitor Training Job
                status = self._monitor_job(job_name)
                self.state[country_code]['training_status'] = status
                self._save_state()
                if status != 'Completed':
                    raise RuntimeError(f"Training failed with status: {status}")
            else:
                self.logger.info(f"Model already trained for {country_code}. Skipping.")
                job_name = self.state[country_code]['trained_model']['job_name']
                status = self.state[country_code].get('training_status')
                if status != 'Completed':
                    self.logger.info(f"Monitoring existing training job {job_name}")
                    status = self._monitor_job(job_name)
                    self.state[country_code]['training_status'] = status
                    self._save_state()
                    if status != 'Completed':
                        raise RuntimeError(f"Training failed with status: {status}")

            # Step 5: Get Best Model
            if not self.state[country_code].get('model_name'):
                self.logger.info(f"Retrieving best model for {country_code}")
                model_name = self._get_best_model(job_name, country_code)
                self.state[country_code]['model_name'] = model_name
                self._save_state()
            else:
                self.logger.info(f"Model already retrieved for {country_code}. Skipping.")
                model_name = self.state[country_code]['model_name']

            # Step 6: Forecasting
            if not self.state[country_code].get('forecasted'):
                self.logger.info(f"Starting forecasting for {country_code}")
                template_file = self.state[country_code]['prepared_data']['template_file']
                self.forecast(country_code, model_name, template_file, backtesting=backtesting)
                self.state[country_code]['forecasted'] = True
                self._save_state()
            else:
                self.logger.info(f"Forecasting already completed for {country_code}. Skipping.")

        except Exception as e:
            self.logger.error(f"Pipeline failed for {country_code}: {str(e)}")
            self._save_state()
            raise

# The rest of the CashForecastingPipeline class remains unchanged

def main():
    parser = argparse.ArgumentParser(description='Cash Forecasting Pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--countries', nargs='+', default=['ZM'],
                        help='Country codes to process')
    parser.add_argument('--resume', action='store_true',
                        help='Resume pipeline from last checkpoint')
    args = parser.parse_args()

    # Load and validate configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    try:
        # First validate with Pydantic
        config_model = ConfigModel(**config_dict)
        # Then create Config dataclass
        config = Config(**config_model.dict())
    except Exception as e:
        logging.error(f"Configuration error: {str(e)}")
        sys.exit(1)

    for country_code in args.countries:
        pipeline = CashForecastingPipeline(config)
        try:
            input_file = f"./data/cash/{country_code}.csv"
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            pipeline.run_pipeline(country_code, input_file, backtesting=False)

        except Exception as e:
            logging.error(f"Failed to process {country_code}: {str(e)}")

if __name__ == "__main__":
    main()
