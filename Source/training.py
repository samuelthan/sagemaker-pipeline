import boto3
import re
import os
import wget
import time
from time import gmtime, strftime
import sys
import json

# Different algorithms have different registry and account parameters
# see: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/amazon/amazon_estimator.py#L272
def get_image_uri(region_name):
    """Return object detection algorithm image URI for the given AWS region"""
    account_id = {
        "us-east-1": "811284229777",
        "us-east-2": "825641698319",
        "us-west-2": "433757028032",
        "eu-west-1": "685385470294",
        "eu-central-1": "813361260812",
        "ap-northeast-1": "501404015308",
        "ap-northeast-2": "306986355934",
        "ap-southeast-2": "544295431143",
        "us-gov-west-1": "226302683700"
    }[region_name]
    return '{}.dkr.ecr.{}.amazonaws.com/object-detection:latest'.format(account_id, region_name)

start = time.time()

region_name = sys.argv[1]
role = sys.argv[2]
bucket = sys.argv[3]
stack_name = sys.argv[4]
commit_id = sys.argv[5]
commit_id = commit_id[0:7]

training_image = get_image_uri(region_name)
timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        wget.download(url, filename)


def upload_to_s3(channel, file):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)

# TODO: Load data and split in to train/test split
prefix = 'output'
train_size = 336

# print ("Downloadng Training Data")
# download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
# upload_to_s3('train', 'caltech-256-60-train.rec')
# print ("Finished Downloadng Training Data")
# print ("Downloadng Testing Data")
# download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')
# upload_to_s3('validation', 'caltech-256-60-val.rec')
# print ("Finished Downloadng Testing Data")

s3_train_data = 's3://ai-sagemaker-objdetect/rego-plate-detection/train'
s3_validation_data = 's3://ai-sagemaker-objdetect/rego-plate-detection/validation'
s3_train_annotation = 's3://ai-sagemaker-objdetect/rego-plate-detection/train_annotation'
s3_validation_annotation = 's3://ai-sagemaker-objdetect/rego-plate-detection/validation_annotation'

s3_output_location = 's3://ai-sagemaker-objdetect/rego-plate-detection/output'

sagemaker = boto3.client(service_name='sagemaker')

sess = sagemaker.Session()
training_image = get_image_uri(sess.boto_region_name)
print (training_image)


od_model = sagemaker.estimator.Estimator(training_image,
                                         role,
                                         train_instance_count=1,
                                         train_instance_type='ml.p2.16xlarge',
                                         train_volume_size = 50,
                                         train_max_run = 360000,
                                         input_mode = 'File',
                                         output_path=s3_output_location,
                                         sagemaker_session=sess)

od_model.set_hyperparameters(base_network='resnet-50',
                             use_pretrained_model=1,
                             num_classes=1,
                             mini_batch_size=16,
                             epochs=30,
                             learning_rate=0.001,
                             lr_scheduler_step='10',
                             lr_scheduler_factor=0.1,
                             optimizer='sgd',
                             momentum=0.9,
                             weight_decay=0.0005,
                             overlap_threshold=0.5,
                             nms_threshold=0.45,
                             image_shape=512,
                             label_width=600,
                             num_training_samples=train_size)

train_data = sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated',
                        content_type='image/jpeg', s3_data_type='S3Prefix')
validation_data = sagemaker.session.s3_input(s3_validation_data, distribution='FullyReplicated',
                             content_type='image/jpeg', s3_data_type='S3Prefix')
train_annotation = sagemaker.session.s3_input(s3_train_annotation, distribution='FullyReplicated',
                             content_type='image/jpeg', s3_data_type='S3Prefix')
validation_annotation = sagemaker.session.s3_input(s3_validation_annotation, distribution='FullyReplicated',
                             content_type='image/jpeg', s3_data_type='S3Prefix')

data_channels = {'train': train_data, 'validation': validation_data,
                 'train_annotation': train_annotation, 'validation_annotation':validation_annotation}

od_model.fit(inputs=data_channels, logs=True)


# create unique job name
job_name = stack_name + "-" + commit_id + "-" + timestamp

# creating configuration files so we can pass parameters to our sagemaker endpoint cloudformation

config_data_qa = {
  "Parameters":
    {
        "BucketName": s3_output_location,
        "CommitID": commit_id,
        "Environment": "qa",
        "ParentStackName": stack_name,
        "JobName": job_name,
        "SageMakerRole": role,
        "SageMakerImage": training_image,
        "Timestamp": timestamp
    }
}

config_data_prod = {
  "Parameters":
    {
        "BucketName": s3_output_location,
        "CommitID": commit_id,
        "Environment": "prod",
        "ParentStackName": stack_name,
        "JobName": job_name,
        "SageMakerRole": role,
        "SageMakerImage": training_image,
        "Timestamp": timestamp
    }
}


json_config_data_qa = json.dumps(config_data_qa)
json_config_data_prod = json.dumps(config_data_prod)

f = open( './CloudFormation/configuration_qa.json', 'w' )
f.write(json_config_data_qa)
f.close()

f = open( './CloudFormation/configuration_prod.json', 'w' )
f.write(json_config_data_prod)
f.close()

end = time.time()
print(end - start)
