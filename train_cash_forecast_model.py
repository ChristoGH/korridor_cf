# %% 19 June 2024 - Cash forecasting model training script
# By Richard van der Wath
# Imports
# import argparse
# train_cash_forecast_model.py
import sagemaker
import boto3
from sagemaker import get_execution_role
from time import gmtime, strftime, sleep
import datetime
import pandas as pd

# %%

country_dict = {'BW': 'botswana',
                'CD': 'democratic-republic-of-congo',
                'NA': 'namibia',
                'RW': 'rwanda',
                'TZ': 'tanzania',
                'ZM': 'zambia',
                'ZW': 'zimbabwe',
                }


def copy_trainingfile_to_s3(country_code, bucket, prefix):
    print(f"{country_code} - copying training file to s3")
    filename = f"./data/cash/{country_code}.csv"
    # Clean potential issues in csv formatting, e.g. removing quotes
    # 25 Jul 2024 update - not needed anymore since double quotes removed
    # file_df = pd.read_csv(filename,na_filter=False) # Namibia 'NA' country code can be incorrectly interpreted as Nan
    # file_df.to_csv(filename,index=False)

    s3 = boto3.resource('s3')

    s3.meta.client.upload_file(Filename=filename, Bucket=bucket,
                               Key=f"{prefix}-{country_code}/train/{country_code}.csv")
    input_data_path = f"s3://{bucket}/{prefix}-{country_code}/train/"
    output_data_path = f"s3://{bucket}/{prefix}-{country_code}/train_output"

    return input_data_path, output_data_path


def train_model_for_country(sm, role, country_code, input_data_path, output_data_path):
    # Establish an AutoML training job name
    timestamp_suffix = strftime("%Y%m%d-%H%M%S", gmtime())
    auto_ml_job_name = f"{country_code}-ts-{timestamp_suffix}"
    print(f"{country_code} AutoMLJobName: {auto_ml_job_name}")

    # sm_client = boto3.client('sagemaker-runtime')

    # Define training job specification
    input_data_config = [
        {'ChannelType': 'training',
         'ContentType': 'text/csv;header=present',
         'CompressionType': 'None',
         'DataSource': {
             'S3DataSource': {
                 'S3DataType': 'S3Prefix',
                 'S3Uri': input_data_path,
             }
         }
         }
    ]

    output_data_config = {'S3OutputPath': output_data_path}

    optimizaton_metric_config = {'MetricName': 'AverageWeightedQuantileLoss'}

    automl_problem_type_config = {
        'TimeSeriesForecastingJobConfig': {
            'ForecastFrequency': '1D',
            'ForecastHorizon': 10,
            'ForecastQuantiles': ['p5', 'p50', 'p99'],
            'TimeSeriesConfig': {
                'TargetAttributeName': 'Demand',
                'TimestampAttributeName': 'EffectiveDate',
                'ItemIdentifierAttributeName': 'ProductId',
                'GroupingAttributeNames': [
                    'BranchId',
                    'CountryCode'
                ]
            },
            'HolidayConfig': [
                {
                    'CountryCode': country_code
                },
            ]
        }
    }

    # With parameters now defined, invoke the training job
    sm.create_auto_ml_job_v2(
        AutoMLJobName=auto_ml_job_name,
        AutoMLJobInputDataConfig=input_data_config,
        OutputDataConfig=output_data_config,
        AutoMLProblemTypeConfig=automl_problem_type_config,
        AutoMLJobObjective=optimizaton_metric_config,
        RoleArn=role
    )

    return timestamp_suffix, auto_ml_job_name


def monitor_job_status(sm, auto_ml_job_name):
    # a looping mechanism to query (monitor) job status
    describe_response = sm.describe_auto_ml_job_v2(AutoMLJobName=auto_ml_job_name)
    job_run_status = describe_response["AutoMLJobStatus"]

    while job_run_status not in ("Failed", "Completed", "Stopped"):
        describe_response = sm.describe_auto_ml_job_v2(AutoMLJobName=auto_ml_job_name)
        job_run_status = describe_response["AutoMLJobStatus"]

        print(
            datetime.datetime.now(),
            describe_response["AutoMLJobStatus"] + " - " + describe_response["AutoMLJobSecondaryStatus"]
        )
        sleep(180)
    return job_run_status


def save_best_model(sm, role, country_code, timestamp_suffix, auto_ml_job_name):
    metric_dict = {}
    best_candidate = sm.describe_auto_ml_job_v2(AutoMLJobName=auto_ml_job_name)['BestCandidate']
    best_candidate_containers = best_candidate['InferenceContainers']
    # best_candidate_name = best_candidate['CandidateName']
    model_name = f"{country_dict[country_code]}-cash-forecast-{timestamp_suffix}"
    metric_dict['model_name'] = model_name
    for metric in best_candidate['CandidateProperties']['CandidateMetrics']:
        metric_dict[metric['MetricName']] = metric['Value']
        if metric['MetricName'] == 'AverageWeightedQuantileLoss':
            awql = metric['Value']
        elif metric['MetricName'] == 'WAPE':
            wape = metric['Value']

    perf_metrics = f"AverageWeightedQuantileLoss:{awql} WAPE:{wape}"

    reponse = sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        Containers=best_candidate_containers,
        Tags=[
            {
                'Key': 'performance metrics',
                'Value': perf_metrics
            },
        ],
    )

    # print(f"{model_name} performance metrics: {best_candidate['CandidateProperties']['CandidateMetrics']}")
    print(f"{model_name} performance metrics: {metric_dict}")

    return metric_dict


# %%
def main():
    metrics_list = []
    # %% AWS conf settings
    region = boto3.Session().region_name
    session = sagemaker.Session()
    bucket = session.default_bucket()
    prefix = 'korridor-cash-train'

    role = get_execution_role()

    # This is the client we will use to interact with SageMaker Autopilot
    sm = boto3.Session().client(service_name="sagemaker", region_name=region)

    for country_code in country_dict.keys():
        print(f"Training model for {country_dict[country_code]}")
        input_data_path, output_data_path = copy_trainingfile_to_s3(country_code, bucket, prefix)
        timestamp_suffix, auto_ml_job_name = train_model_for_country(sm, role, country_code, input_data_path,
                                                                     output_data_path)
        job_run_status = monitor_job_status(sm, auto_ml_job_name)
        if job_run_status == 'Completed':
            metrics = save_best_model(sm, role, country_code, timestamp_suffix, auto_ml_job_name)
            metrics_list.append(metrics)
        timestamp = strftime("%Y%m%d-%H%M%S", gmtime())

    model_metrics = pd.DataFrame(metrics_list)
    model_metrics.to_csv(f'cash_forecasting_model_metrics_{timestamp}.csv', index=False)
    # Warning, first install openpyxl before using xlsx writer!
    model_metrics.to_excel(f'cash_forecasting_model_metrics_{timestamp}.xlsx', index=False)


if __name__ == "__main__":
    main()
