import urllib.parse
from SQL_Logins import db_servers


# --- create a engine to connect to SQL

def databaseCredentials(make_eng: bool, *agrs: str):
    DRIVER = 'driver={SQL Server}'

    conn_string = (str(DRIVER) +
                   ";server={};DATABASE={};UID={};PWD={}"
                   .format(*list(map(lambda x: ''.join(x), [a for a in agrs]))))
    engine_str = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(conn_string)

    # --- return engine string or initialize the engine
    if make_eng:
        from sqlalchemy import create_engine
        engine = create_engine(engine_str)
        return engine
    else:
        return engine_str


def createQuery(p1: object, p2: object, p3: object, p4: object, tbl_name: object) -> object:
    """createQuery:
            acts as a function factory to generate sql queries
    """
    from sqlalchemy import MetaData, Table
    try:
        engine = databaseCredentials(True, p1, p2, p3, p4)
        metadata = MetaData()
        database_info = Table(str(tbl_name), metadata, autoload=True, autoload_with=engine)
        # print(engine.table_names())
        # --- attempt to connect to the database
        connection = engine.connect()
        return connection
    except Exception as e:
        print(str(e))

# query = createQuery("LL-SQL-PI01", 'LifeLineV2', db_servers.get("LL-SQL-PI01")[0], db_servers.get("LL-SQL-PI01")[1],
#           tbl_name="CallTraceAttempts")
