import os
import sys
import requests
import pandas_datareader as pdr
import pandas as pd
import numpy as np
from functools import wraps, reduce
import urllib
import requests
from bs4 import BeautifulSoup


#--- ticker search
def search_tickers(tick, start=0, end=10):
    """search for a specified ticker"""
    tickers = (pdr.get_iex_symbols()
               .drop(['date', 'iexId', 'isEnabled', 'type'], axis=1)
               .iloc[start:end, :]
               )
    str_tick = np.array(tickers[tickers.symbol == tick]).flatten()
    while len(str_tick) <= 0 and end < 10000:
        return search_tickers(tick, start=end, end=(1.5)*end)
    if end > 8999:
        try:
           if 1 == 0:
               pass
           else:
               raise IndexError('ticker: {} was not found, please refine search start position...\n'.format(tick))
        except IndexError as ie:
            print(ie)
        except Exception:
            print('uncaught exception was raise ...\n')
    else:
        return str_tick[0]


#--- decorator that takes a query string and creates a google search query
def url_builder(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        func_str = f(*args, **kwargs)
        base = 'https://www.google.com/search'
        query = '?q='
        return base+query+func_str
    return wrapper

#--- wrapped function
@url_builder
def create_query(search):

    key_words = ['{}+'.format(a) for a in search.split(" ")]
    query = ''.join(key_words)
    return query.rstrip('+')


def main():
    # set the stock ticker
    tkr = search_tickers(tick='AAME', start=0, end=1000)

    # create the query to be passed to google
    google_query = create_query(search=tkr)

    r = requests.get(google_query)

    data = r.text
    soup = BeautifulSoup(data, "lxml")

    # get all links returned from te google search
    links = [s.get_text() for s in soup.find_all('cite')]

    #--- search through the first link
    url = links[0]

    url_r = requests.get(url)
    company_data = url_r.text
    url_soup = BeautifulSoup(company_data, "lxml")
    company_page_links = {s.get_text():s.get('href') for s in url_soup.find_all('a')}
    print(company_page_links)

if __name__=='__main__':
    main()
