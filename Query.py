from CreateSQLEngineString import *
from datetime import date, timedelta


def queryTime(f):
    """decorator that sets and formats dates for sql queries"""
    def wrapper(date_diff, *agrs, **kwargs):
        today = date.today()
        today_formatted = today.strftime('%m/%d/%Y').lstrip("0").replace(" 0", " ")
        date_diff = today - timedelta(date_diff)
        date_diff_formatted = date_diff.strftime('%m/%d/%Y').lstrip("0").replace(" 0", " ")
        return f(today_formatted, date_diff_formatted)
    return wrapper

# TODO -- fix this decorator
def QueryDecorator(f):
    def wrapper(p1: str, p2: str, p3, p4, p5: str, *args, **kwargs):
        conn = f(createQuery(str(p1), str(p2), p3, p4, str(p5)))
        return conn
    return wrapper


# --- current queries that need to be run

@queryTime
def CallTraceAttemptsQuery(*args):
    """Fix dc to na issue in CallTraceAttempts"""

    conn = createQuery("LL-SQL-PI01", 'LifeLineV2',
                       db_servers.get("LL-SQL-PI01")[0],
                       db_servers.get("LL-SQL-PI01")[1],
                       tbl_name="CallTraceAttempts")

    stmt = """SELECT b.CallTraceAttemptId FROM(SELECT aa.CallGuid,aa.[COUNT CallGuid] 
    FROM(SELECT CallGuid,count(CallGuid) as [COUNT CallGuid] 
    FROM  [LifelineV2].[dbo].[CallTraceAttempts] 
    GROUP BY CallGuid)aa WHERE aa.[COUNT CallGuid] > 1)a 
    INNER JOIN ( SELECT [DateCreated],[CallGuid],[AttemptStatus],
    [CrisisCenterName],CallTraceAttemptId 
    FROM  [LifelineV2].[dbo].[CallTraceAttempts] 
    WHERE  EasternTime < '{}' AND EasternTime >= '{}' 
    AND AttemptStatus='dc' AND CrisisCenterKey IN ('NY212614') 
    AND TryNumber = 1 )b ON a.CallGuid = b.CallGuid
    """.format(*[''.join(str(a)) for a in args])

    #stmt2 = """SELECT TOP(100) * FROM [LifelineV2].[dbo].[CallTraceAttempts]"""
    # --- execute sql query
    result_proxy = conn.execute(stmt)
    # --- check results
    results = result_proxy.fetchall()
    conn.close()
    return results

@queryTime
def RemoveDupsCallTraceAttempts(*args):
    """remove duplicate data from CallTraceAttempts"""
    conn = createQuery("LL-SQL-PI01", 'LifeLineV2',
                       db_servers.get("LL-SQL-PI01")[0],
                       db_servers.get("LL-SQL-PI01")[1],
                       tbl_name="CallTraceAttempts")
    stmt = """SELECT CallGuid, MAX(CallTraceAttemptId) AS CallTraceAttemptId
    FROM [LifelineV2].[dbo].[CallTraceAttempts]
    WHERE EasternTime < '{}' AND EasternTime >= '{}'
    GROUP BY CallGuid, AttemptStatus, TryNumber
    HAVING COUNT(*) > 1""".format(*[''.join(str(a)) for a in args])

    # --- execute sql query
    result_proxy = conn.execute(stmt)
    # --- check results
    results = result_proxy.fetchall()
    conn.close()
    return results



#pprint.pprint(CallTraceAttemptsQuery(1, 10))
#pprint.pprint(RemoveDupsCallTraceAttempts(1, 1))
