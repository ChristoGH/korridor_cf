# %% 24 May 2024 - Batch Transform inference script
# batch_transform_inference.py
# Imports
import argparse
import sagemaker
import boto3
from sagemaker import get_execution_role
from time import gmtime, strftime, sleep
import datetime




def initiate_batch_transform(sagemaker_ins, model_name, input_source, output_path, timestamp_suffix):
    # timestamp_suffix = strftime("%Y%m%d-%H%M%S", gmtime())
    transform_job_name = f"{model_name}-{timestamp_suffix}"
    print("BatchTransformJob: " + transform_job_name)
    response = sagemaker_ins.create_transform_job(
        TransformJobName=transform_job_name,
        ModelName=model_name,
        MaxPayloadInMB=0,
        ModelClientConfig={
            'InvocationsTimeoutInSeconds': 3600
        },
        TransformInput={
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': input_source  # 's3://{}/{}/batch_predict/'.format(bucket, prefix)
                }
            },
            'ContentType': 'text/csv',
            'SplitType': 'None'
        },
        TransformOutput={
            'S3OutputPath': output_path,  # 's3://{}/{}/batch_predict/output/'.format(bucket, prefix),
            'AssembleWith': 'Line',
        },
        TransformResources={
            'InstanceType': 'ml.m5.12xlarge',
            'InstanceCount': 1
        }
    )
    return transform_job_name, response


def new_initiate_batch_transform(sagemaker_ins, model_name, input_source, output_path, timestamp_suffix):
    # timestamp_suffix = strftime("%Y%m%d-%H%M%S", gmtime())
    transform_job_name = f"{model_name}-{timestamp_suffix}"
    print("BatchTransformJob: " + transform_job_name)

    # Increase MaxPayloadInMB and allow splitting of CSV lines
    response = sagemaker_ins.create_transform_job(
        TransformJobName=transform_job_name,
        ModelName=model_name,
        MaxPayloadInMB=100,  # Set to 100 MB to allow larger batch sizes
        ModelClientConfig={
            'InvocationsTimeoutInSeconds': 1800  # Increase timeout to 2 hours
        },
        TransformInput={
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': input_source
                }
            },
            'ContentType': 'text/csv',
            'SplitType': 'Line'  # Split input data line by line (row by row)
        },
        TransformOutput={
            'S3OutputPath': output_path,
            'AssembleWith': 'Line',
        },
        TransformResources={
            'InstanceType': 'ml.m5.12xlarge',
            'InstanceCount': 1  # Increase instance count if necessary
        }
    )
    return transform_job_name, response


def poll_tranform_job_status(sagemaker_ins, transform_job_name):
    # %% Poll for batch transformation job to complete.
    # Once completed, resulting prediction files are available at the URI shown in the prior cell,
    # S3OutputPath. We use the API method describe_transform_job to complete this step.
    describe_response = sagemaker_ins.describe_transform_job(TransformJobName=transform_job_name)

    job_run_status = describe_response["TransformJobStatus"]

    while job_run_status not in ("Failed", "Completed", "Stopped"):
        describe_response = sagemaker_ins.describe_transform_job(TransformJobName=transform_job_name)
        job_run_status = describe_response["TransformJobStatus"]
        print(datetime.datetime.now(), describe_response["TransformJobStatus"])

        sleep(60)

    return describe_response


def parse_s3_uri(s3_uri):
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI. Must start with 's3://'")
    parts = s3_uri[5:].split("/", 1)
    bucket_name = parts[0]
    object_key = parts[1] if len(parts) > 1 else ""
    return bucket_name, object_key



def download_file(bucket_name, file_key):
    s3 = boto3.resource('s3')
    # s3.meta.client.list_objects(Bucket=input_bucket_name)['Contents']
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=file_key):
        if obj.key[-1] == '/':
            continue
        file_name = obj.key.split('/')[-1]
        bucket.download_file(obj.key, file_name)
        print(f"Downloaded {bucket_name}/{obj.key} to {file_name}")


def copy_file_to_s3(source_file, bucket_name, file_key):
    s3 = boto3.resource('s3')

    # s3.meta.client.upload_file(Filename='./data/2024-05-13-Forecast-train.csv', Bucket=bucket, Key=prefix+'/train/2024-05-13-Forecast-Sample.csv')
    s3.meta.client.upload_file(Filename=source_file, Bucket=bucket_name, Key=file_key)


def main():
    parser = argparse.ArgumentParser(description='Perform a batch transform using a specified model.')
    parser.add_argument('model_name', type=str, help='The name of the model to use.')
    parser.add_argument('input_path', type=str, help='The S3 URI of the input data source.')
    parser.add_argument('--output_path', type=str, help='The S3 URI of the output data path (optional).')
    parser.add_argument('--download_output', action='store_true', help='Download the processed output locally.')

    args = parser.parse_args()
    timestamp_suffix = strftime("%Y%m%d-%H%M%S", gmtime())


    region = boto3.Session().region_name
    session = sagemaker.Session()

    role = get_execution_role()

    # Parse input data source
    if not args.input_path.startswith("s3://"):  # local filepath so we need to copy the file to s3
        file_name = args.input_path.split('/')[-1]
        input_bucket_name = session.default_bucket()
        input_object_key = f"korridor_batch_transform/{timestamp_suffix}"
        copy_file_to_s3(args.input_path, input_bucket_name, f"{input_object_key}/{file_name}")
        args.input_path = f"s3://{input_bucket_name}/{input_object_key}/"
        print(f"Setting input path to: {args.input_path}")
    else:
        input_bucket_name, input_object_key = parse_s3_uri(args.input_path)
    # print(f"Input bucket {input_bucket_name}, Input obj key {input_object_key}")

    # If output data path is not provided, set to a default path
    if not args.output_path:
        args.output_path = f"s3://{input_bucket_name}/{input_object_key}/output/"
        print(f"Setting output path to: {args.output_path}")

    output_bucket_name, output_object_key = parse_s3_uri(args.output_path)

    # This is the client we will use to interact with SageMaker Autopilot
    sm = boto3.Session().client(service_name="sagemaker", region_name=region)


    transform_job_name, response = initiate_batch_transform(sm, args.model_name, args.input_path, args.output_path,
                                                            timestamp_suffix)

    # transform_job_name = 'zambia-cash-forecast-batch-20240520-132831-20240524-100634'

    describe_response = poll_tranform_job_status(sm, transform_job_name)
    print(describe_response['DataProcessing'])

    if args.download_output:
        download_file(output_bucket_name, output_object_key)


if __name__ == "__main__":
    main()
    # python batch_transform_inference.py zambia-cash-forecast-20240725-131728 ./data/cash/ZM_forecast.csv --download_output
    # python batch_transform_inference.py rwanda-cash-forecast-20240725-115918 ./data/cash/RW_forecast.csv --download_output
