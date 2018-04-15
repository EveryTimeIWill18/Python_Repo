import requests
from bs4 import BeautifulSoup
from functools import reduce, partial
import re
import pandas as pd

"""
William Murphy
4/15/2018
    - first crack at an API for
    yahoo finance that we could use
    for future work.
    
    TODO:
        - generalize it(i.e. allow for any number of urls)
        - many things ...
        
        
"""


p_index = ['Date', 'Option', 'High', 'Low', 'Close', 'Adj Close']

# --- make a request
def make_request(url: str, search_str: str) -> list:
    try:
        r = requests.get(str(url))
        soup = BeautifulSoup(r.content, "html.parser")
        raw = soup.find_all(str(search_str), text=True)
        return list(raw)
    except Exception as e:
        print(str(e))

# --- being data cleaning
def clean_data(f: make_request, r_tag = None):
    try:
        cleaned = [re.sub(r'.*\>', '', str(f[i]).rstrip(str(r_tag))) for i,_ in enumerate(f)]
        cleaned = cleaned[20:719]
        return cleaned
    except Exception as e:
        print(str(e))

# --- create data dictionary
def data_dictionary(f: clean_data):
    names = list(p_index)
    Date = [str(f[i]).replace(',', '').strip().replace(' ', '-') for i,_ in enumerate(f) if i%7 == 0]
    Option = [str(f[i]).replace(',', '')  for i,_ in enumerate(f) if i%7 == 1]
    High = [str(f[i]).replace(',', '')  for i,_ in enumerate(f) if i%7 == 2]
    Low = [str(f[i]).replace(',', '')  for i,_ in enumerate(f) if i%7 == 3]
    Close = [str(f[i]).replace(',', '')  for i,_ in enumerate(f) if i%7 == 4]
    Adj_Close = [str(f[i]).replace(',', '')  for i,_ in enumerate(f) if i%7 == 5]

    DataDictionary = {
         'Date': Date,
         'Option': Option,
         'High': High,
         'Low': Low,
         'Close': Close,
         'Adj Close': Adj_Close
     }

    return DataDictionary

# --- create pandas DataFrame
def create_DataFrame(f: data_dictionary) -> pd.DataFrame:
    return pd.DataFrame(data=f, columns=list(f.keys()))


# --- run job
if __name__=='__main__':
    page = make_request(url='https://finance.yahoo.com/quote/^GSPC/history?period1=1199160000&period2=1523761200&interval=1d&filter=history&frequency=1d',
                        search_str='span')

    data_job = clean_data(page, r_tag=' </span>')

    munge = data_dictionary(f=data_job)

    df = create_DataFrame(f=munge)

    print(df.head())
    print(df.tail())