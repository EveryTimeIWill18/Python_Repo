import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mathtext as mathtext
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

# --- make get request
url_ = 'https://assets.datacamp.com/production/course_3882/datasets/income_growth.csv'
response = requests.get(url_)
if response.status_code == 200:
    print("connection successful, status code: [{}]".format(response.status_code))
else:
    print("connection failed, status code: [{}]".format(response.status_code))
    
# --- process the stream 
raw_stream = response.content\
                     .decode('utf-8')\
                     .splitlines()

column_header = raw_stream[0].split(",")
data_ = raw_stream[1:]
data_dict = {str(h): [] for h in column_header}

# --- algorithm to build the dictionary that will be passded to pandas
for i, _ in enumerate(list(data_dict.keys())):
    for row in data_:
        current = row.split(",")
        for j, _ in enumerate(current):
            if j == i:
                data_dict.get(list(data_dict.keys())[i])\
                         .append(current[j])
                         
# --- build the data frame
df = pd.DataFrame(data_dict, index=pd.to_datetime(data_dict.get('DATE')))
df = df.drop(['DATE'], axis=1)
df.head()

# --- descriptive statistics
usa = df.USA.apply(lambda x: float(x))
china = df.China.apply(lambda x: float(x))
brazil = df.Brazil.apply(lambda x: float(x))
countries = [usa, china, brazil]
# mean 
print("----Expected Income Growth ----")
mean = list(map(lambda x: np.mean(np.array(x)), countries))
grand_mean = sum(mean)/3.0

print(mean)
print("\n")

# variance
print("----Variance Income Growth ----")
var = list(map(lambda x: np.var(np.array(x)), countries))
print(var)
print("\n")

# standard deviation
print("----Std Deviation Income Growth ----")
sdev = list(map(lambda x: np.std(np.array(x)), countries))
print(sdev)

df[['USA', 'China']]
usa = df.USA.apply(lambda x: float(x))
china = df.China.apply(lambda x: float(x))
brazil = df.Brazil.apply(lambda x: float(x))

# --- plotting setup
plt.style.use('bmh')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
usa.plot(legend=True); plt.tight_layout()
china.plot(legend=True); plt.tight_layout()
brazil.plot(title='Income Growth of United States, China, and Brazil', figsize=(16,6), grid=True, legend=True); plt.show()
