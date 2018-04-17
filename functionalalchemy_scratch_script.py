from sqlalchemy import create_engine, func, select, Table, MetaData
from SQL_Logins import db_servers
from itertools import chain
import urllib.parse
import pprint

# --- constants
DRIVER = 'driver={SQL Server}'

def get_datbase_creds(arg: str) -> str:
    return str(arg)

creds = lambda x: get_datbase_creds
mapped_creds = list(map(lambda x: creds, [i for i in range(4)]))
mapped_creds[0] = 'LL-SQL-PI01'
mapped_creds[1] = 'LifeLineV2'
mapped_creds[2] = db_servers.get(str(mapped_creds[0]))[0]
mapped_creds[3] = db_servers.get(str(mapped_creds[0]))[1]

conn_string = DRIVER + ";server={};DATABASE={};UID={};PWD={}".format(*list(map(lambda x: ''.join(x), mapped_creds)))
parse_con_string = urllib.parse.quote_plus(conn_string)
engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % parse_con_string)
conn = engine.connect()
metadata = MetaData()
calltraceattempts = Table('CallTraceAttempts', metadata, autoload=True, autoload_with=engine)

stmt = select([calltraceattempts])
results = conn.execute(stmt)

cntr = 0
for result in results:
    print(result)
    cntr += 1
    if cntr == 10:
        break
