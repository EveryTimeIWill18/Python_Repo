import os
import pandas as pd
import numpy as np
import pandas_datareader as pdr
from pandas_datareader import data, wb
import datetime as dt
import pprint
from functools import wraps, reduce, partial
from itertools import chain




ovx_data = pd.read_csv("F:\\^OVX.csv", index_col=None, header=0, engine='c')

vix_data = pd.read_csv("F:\\^VIX.csv", index_col=None, header=0, engine='c')

sp500 = pd.read_csv("F:\\^SP500TR.csv", index_col=None, header=0, engine='c')

# --- weight average monthly
farma_french = pd.read_csv("F:\\48_Industry_Portfolios.CSV")


def load_datasets(path=None, *args: str) -> dict:
    """load data sets"""
    dirname = os.path.dirname(str(path))
    if os.getcwd() is not dirname:
        try:
            os.chdir(dirname)
            print(os.getcwd())
        except FileExistsError as e:
            print(str(e))
        except FileNotFoundError as e:
            print(str(e))
        except IOError as e:
            print(str(e))
        except Exception as e:
            print(str(e))

    # --- walk through path to find files
    dir_files = [f for _, _, f in os.walk(path) if len(f) > 0]
    flattened_dirs = list(chain(*dir_files))
    ticks = {str(a): pd.read_csv(str(path)+str(a), header=0, engine='c') for a in args if a in flattened_dirs}
    return ticks


pprint.pprint(load_datasets("F:\\", "^OVX.csv", "^VIX.csv", "^SP500TR.csv", "48_Industry_Portfolios.CSV"))
