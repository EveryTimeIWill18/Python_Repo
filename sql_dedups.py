import pickle
from datetime import datetime, timedelta
from functools import reduce
import pyodbc
import pandas as pd
from urllib.parse import quote_plus
from pprint import pprint
from sqlalchemy import (create_engine, MetaData,
                        Table, Column, select, or_, and_, func, INT)
from sqlalchemy.types import DateTime

# --- setup engine
dirver = 'driver={SQL Server}'
conn_string = reduce((lambda x, y: x+y),
                     ["driver={SQL Server}",';server=',
                      ';DATABASE=',';UID=',';PWD='])

# --- create engine
parse_ = quote_plus(conn_string)
engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % parse_)

# --- create connection
connection = engine.connect()

# --- get metadata
metadata = MetaData()
# --- get the CallTraceAttempts Table
callTraceAttempts = Table('CallTraceAttempts', metadata,
                          autoload=True, autoload_with=engine)
fields = callTraceAttempts.c.keys()
pprint(fields)
prev_day = datetime.today() - timedelta(1)
today = datetime.utcnow()

# --- create the query
stmt = select([callTraceAttempts.c.CallGuid,
               func.max(callTraceAttempts.c.CallTraceAttemptId)])\
            .where(and_(callTraceAttempts.c.EasternTime >= func.current_date()-int(1),
                        callTraceAttempts.c.EasternTime < func.current_date()))\
            .group_by(callTraceAttempts.c.CallGuid,
                      callTraceAttempts.c.AttemptStatus,
                      callTraceAttempts.c.TryNumber)\
            .having(func.count() > int(1))

# --- delete statement
delete_stmt = callTraceAttempts.delete().where(
    callTraceAttempts.c.CallTraceAttemptId==select([callTraceAttempts.c.CallGuid,
                   func.max(callTraceAttempts.c.CallTraceAttemptId)])\
                .where(and_(callTraceAttempts.c.EasternTime >= func.current_date()-int(1),
                            callTraceAttempts.c.EasternTime < func.current_date()))\
                .group_by(callTraceAttempts.c.CallGuid,
                          callTraceAttempts.c.AttemptStatus,
                          callTraceAttempts.c.TryNumber)\
                .having(func.count() > int(1))

)
# --- search for mare than 2
stmt_two = select([callTraceAttempts.c.CallGuid, callTraceAttempts.c.TryNumber,
                   func.count()])\
        .where(and_(callTraceAttempts.c.EasternTime >= func.current_date()-int(1),
               callTraceAttempts.c.EasternTime < func.current_date()))\
        .group_by(callTraceAttempts.c.CallGuid,
                  callTraceAttempts.c.AttemptStatus,
                  callTraceAttempts.c.TryNumber)\
        .having(func.count() > int(2))

# --- delete
delete_stmt_two =   callTraceAttempts.delete().where(
    callTraceAttempts.c.CallTraceAttemptId==select([callTraceAttempts.c.CallGuid, callTraceAttempts.c.TryNumber,
                       func.count()])\
            .where(and_(callTraceAttempts.c.EasternTime >= func.current_date()-int(1),
                   callTraceAttempts.c.EasternTime < func.current_date()))\
            .group_by(callTraceAttempts.c.CallGuid,
                      callTraceAttempts.c.AttemptStatus,
                      callTraceAttempts.c.TryNumber)\
            .having(func.count() > int(2))
)


# --- uncomment to check to errors
pprint(str(delete_stmt))

results = connection.execute(stmt).fetchall()

# --- export results via pickle file
today_ = str(datetime.today()).split(" ")[0]
results_path = "O:\\Information Technology\\Data Services\\CallTraceAttemptsPickleFiles\\SQLAlchemyOutput"
p_name = 'sql_alchemy_output_{}.pickle'.format(today_)
query_output = open(results_path + '\\' + p_name, 'wb')
pickle.dump(results, query_output)
query_output.close()

# --- export results via text file
f_name = 'results_query_text_out_{}.txt'.format(str(today_))
print(f_name)
output =  open(results_path + '//' + str(f_name) , 'w')
output.write(str(results))
output.close()
