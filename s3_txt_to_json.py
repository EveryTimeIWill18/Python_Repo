import sys
import re
import json
from itertools import chain 
from io import BytesIO
from abc import ABCMeta, abstractmethod
import mmap, pickle
from datetime import datetime
import numpy as np
import pandas as pd
import pandas.io.parsers as pdp
import matplotlib.pyplot as plt
import boto3

s3 = boto3.resource('s3')
client = boto3.client('s3')
obj_ = client.list_objects(Bucket='bucket_name')
bucket_ = s3.Bucket('bkt')
unsorted = []
for file in bucket_.objects.filter(Prefix='prefix'):
    unsorted.append(file)

bkt = client.get_object(Bucket='bucket_name',Key=unsorted[0].key)
s3_array = bkt['Body'].read().decode('utf-8')
pos = 0
brkt_counter = 0
feed = []
json_recs = []
while pos < len(s3_array):
    if s3_array[pos] == '{':
        feed.append(s3_array[pos])
        brkt_counter += 1
        pos += 1
    if s3_array[pos] == '}' and brkt_counter > 1:
        feed.append(s3_array[pos])
        pos += 1
        brkt_counter -= 1
    if s3_array[pos] == '}' and brkt_counter == 1:
        feed.append(s3_array[pos])
        js_str = ''.join(feed)
        json_recs.append(json.loads(js_str))
        feed.clear()
        brkt_counter -= 1
        pos += 1
    else:
        feed.append(s3_array[pos])
        pos += 1
json_recs

import pandas.io.json as pdj
json_df = pdj.json_normalize(json_recs)
json_df
