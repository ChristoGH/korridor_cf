import boto3
import pandas as pd
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DebugForecast')


def debug_forecast_output(
        bucket: str = "sagemaker-eu-west-1-717377802724",
        prefix: str = "cash-forecasting-ZM/20241121-071325/inference-output",
        local_dir: str = "./debug_output"
):
    """
    Debug the forecast output files from a specific transform job.
    """
    s3_client = boto3.client('s3')

    # Create output directory
    os.makedirs(local_dir, exist_ok=True)

    try:
        # List all output files
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )

        if 'Contents' not in response:
            logger.error(f"No files found in s3://{bucket}/{prefix}")
            return

        # Download and examine each output file
        for obj in response['Contents']:
            s3_key = obj['Key']
            if not s3_key.endswith('.out'):
                continue

            local_file = os.path.join(local_dir, os.path.basename(s3_key))
            logger.info(f"Downloading {s3_key}")

            # Download file
            s3_client.download_file(bucket, s3_key, local_file)

            # Read and examine file
            with open(local_file, 'r') as f:
                header = f.readline().strip()
                logger.info(f"Header: {header}")

                # Read first few lines
                sample_lines = [f.readline().strip() for _ in range(5)]
                logger.info("Sample lines:")
                for line in sample_lines:
                    logger.info(line)

            # Try reading with pandas
            try:
                df = pd.read_csv(local_file)
                logger.info(f"\nDataFrame Info:")
                logger.info(f"Shape: {df.shape}")
                logger.info("\nColumns:")
                for col in df.columns:
                    logger.info(f"{col}: {df[col].dtype}")
                    if df[col].dtype != 'object':
                        logger.info(f"Sample values: {df[col].head()}")
                    num_errors = df[df[col] == 'ERROR'].shape[0]
                    if num_errors > 0:
                        logger.info(f"Number of ERROR values: {num_errors}")
            except Exception as e:
                logger.error(f"Failed to read file with pandas: {e}")

        # Also check the input data used for transform
        input_prefix = prefix.replace('inference-output', 'inference')
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=input_prefix
            )

            if 'Contents' in response:
                for obj in response['Contents']:
                    s3_key = obj['Key']
                    if s3_key.endswith('.csv'):
                        local_file = os.path.join(local_dir, f"input_{os.path.basename(s3_key)}")
                        s3_client.download_file(bucket, s3_key, local_file)
                        df = pd.read_csv(local_file)
                        logger.info(f"\nInput Data Info:")
                        logger.info(f"Shape: {df.shape}")
                        logger.info(f"Columns: {df.columns.tolist()}")
                        logger.info(f"Sample rows:\n{df.head()}")
        except Exception as e:
            logger.error(f"Failed to examine input data: {e}")

    except Exception as e:
        logger.error(f"Debug process failed: {e}")
    finally:
        logger.info(f"Debug files saved to {local_dir}")


if __name__ == "__main__":
    debug_forecast_output()