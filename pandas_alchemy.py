#!ve/bin/python3
import pyodbc
import pickle
from datetime import datetime, timedelta
from functools import reduce, wraps
from urllib.parse import quote_plus
from sqlalchemy import (create_engine, MetaData,Table,
                        Column, select, or_, and_, func, INT)
from sqlalchemy.types import DateTime
from sqlalchemy.sql import text
import pandas as pd

# --- get sql driver
def build_engine(un: str, pwd: str, srvr: str, db: str) -> 'engine.base.Engine':
    """get available sql drivers"""
    drivers_ = pyodbc.drivers()

    # add more sql drivers as needed
    driver_priority = {
                       "ODBC Driver 17 for SQL Server":   7,
                       "ODBC Driver 13.1 for SQL Server": 6,
                       "ODBC Driver 13 for SQL Server":   5,
                       "ODBC Driver 11 for SQL Server":   4,
                       "SQL Server Native Client 11.0":   3,
                       "SQL Server Native Client 10.0":   2,
                       "SQL Server":                      1
                      }
    effective_drivers = filter(lambda x: x in driver_priority, drivers_)
    try:
        best_driver = max(effective_drivers,
                          key=(lambda x: driver_priority[x]))
    except ValueError as e:
        print(str(e))
    try:
        engine = create_engine("mssql+pyodbc://"
                               + str(un) + ":"
                               + str(pwd)
                               + "@" + str(srvr)
                               + "/" + str(db)
                               + "?driver={}".format(best_driver))
    except Exception as e:
        print(str(e))

    return engine

def build_dataframe(table_name: str):
    """build a pandas DataFrame containing sql query data"""
    #TODO
