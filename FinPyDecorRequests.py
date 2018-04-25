import numpy as np
import pandas as pd
from pprint import pprint
import requests
from functools import wraps, reduce
from datetime import date, datetime, timedelta, time
from bs4 import BeautifulSoup
import re

"""William Murphy
   4/25/2018
   use of decorators to
   limit the need of classes
"""


# --- decorator with variable input
def ticker(ticker: str):
    def decorate(f: callable):
        @wraps(f)
        def wrapper(url, period, t1, t2, *args, **kwargs):
            atom = datetime(1970,1,1)
            d = timedelta(days=1).total_seconds()
            t1_diff = int(((t1-atom).days)*d)+104400
            t2_diff = int(((t2-atom).days)*d)+104400

            url_ = (url + '/' + ticker + '/history?period1='
                    + str(t1_diff) + '&period2=' + str(t2_diff)
                    + '&interval=' + period
                    + '&filter=history&frequency=' + period)
            print(url_)
            F = f(url_, period, t1, t2, *args, **kwargs)
            return F
        return wrapper
    return decorate



@ticker('AAPL')
def makeWebRequest(url, period, t1, t2):
    r = requests.get(str(url))
    if r.status_code == 200:
        print(r.status_code)
        print("Connection successful")
        return r
    else:
        print("Failed to connect\n")


# --- test connection
yahoo = makeWebRequest(url="https://finance.yahoo.com/quote",
                       period='1d', t1=datetime(2018,4,22),
                       t2=datetime(2018,4,23))


#soup = BeautifulSoup(yahoo.content, 'html.parser')
#table = soup.find_all('table')

print(re.findall(r'\<span\s\bdata-reactid="\d{0,3}">\w+\<*', str(yahoo.content)))
