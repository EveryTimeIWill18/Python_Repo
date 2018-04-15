import os
import pandas as pd
from tabula import read_pdf
import itertools
import pprint

FILE = "C:\\Users\\wmurphy\\Downloads\\UserGroup_A_B"

if os.getcwd() is not FILE:
    os.chdir(FILE)

data = {str(d).split(".")[0]: pd.read_csv(d, header=0, index_col=None, engine='c') for d in list(os.listdir())}
keys = list(data.keys())

user_completion = [(keys[i], str((data.get(keys[i])['Completed']
                                  .isnull()
                                  .sum()/len(data.get(keys[i])))
                                  .round(decimals=2)*100)+"%")
                   for i,_ in enumerate(keys)]
user_df = {
            "User": [keys[i] for i,_ in enumerate(keys)],

            "Training % Completed": [(data.get(keys[i])['Completed']
                                        .isnull()
                                        .sum()/len(data.get(keys[i])))
                                         for i,_ in enumerate(keys)],

            "Avg Score": [(data.get(keys[i])['Completed']
                                  .sum()/len(data.get(keys[i])))
                                  .round(decimals=2) for i,_ in enumerate(keys)]

        }

usr_one = {keys[0]:"{}%".format(data.get(keys[0])['Completed'].isnull().sum()/len(data.get(keys[0])))}
