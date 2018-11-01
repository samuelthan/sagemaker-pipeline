import boto3
import wget
import json
import numpy as np
import sys
import time

start = time.time()

endpoint_name = sys.argv[1]
configuration_file = sys.argv[2]

with open(configuration_file) as f:
    data = json.load(f)

commit_id = data["Parameters"]["CommitID"]
timestamp = data["Parameters"]["Timestamp"]


endpoint_name = endpoint_name + "-" + commit_id + "-" + timestamp

runtime = boto3.client('runtime.sagemaker') 

wget.download("http://www.vision.caltech.edu/Image_Datasets/Caltech256/images/008.bathtub/008_0007.jpg", "test.jpg")

with open("test.jpg", 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)
response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType='image/jpeg', 
                                   Body=payload)
result = response['Body'].read()
# result will be in json format and convert it to ndarray
result = json.loads(result.decode('utf-8'))
print(result)

end = time.time()
seconds = end - start
seconds = repr(seconds)
print ("Time: " + seconds)
