

import MySQLdb as db

from DbTools.DbConnection import DbConnection

CONN_STR = "file:/reg/g/psdm/psdatmgr/etc/.taperestore-conn"

def __escape_string(str):

    conn = DbConnection(conn_string=CONN_STR)
    cursor = conn.cursor(dictcursor=True)
    return conn.escape_string(str)

def __do_select(statement):

    conn = DbConnection(conn_string=CONN_STR)
    cursor = conn.cursor(dictcursor=True)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute(statement)
    rows = cursor.fetchall()
    if not rows : return None

    return rows[0]

def __do_select_many(statement):

    conn = DbConnection(conn_string=CONN_STR)
    cursor = conn.cursor(dictcursor=True)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute(statement)
    return cursor.fetchall()

def __do_sql(statement):

    conn = DbConnection(conn_string=CONN_STR)
    cursor = conn.cursor(dictcursor=True)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute("BEGIN")
    cursor.execute(statement)
    cursor.execute("COMMIT")


DONE = 'DONE'
SUBMITTED = 'SUBMITTED'
FAILED = 'FAILED'
RECEIVED = 'RECEIVED'

def files_with_status(status=None):
    sel = ""
    if status:
        sel = "WHERE status = '%s'" % status

    files = __do_select_many("SELECT * FROM file_restore_requests %s " % sel)

    return files

    
def set_status(flag, irods_fname):
    query = "UPDATE file_restore_requests SET status='%s' " \
            "WHERE irods_filepath = '%s'" % (flag, irods_fname)
    __do_sql(query)

def set_submitted(irods_fname):
    query = "UPDATE file_restore_requests SET status='SUBMITTED' " \
        "WHERE irods_filepath = '%s'" % irods_fname
    __do_sql(query)
    

def file_by_name(name):
    sel = "WHERE irods_filepath like '%%%s%%'" % name
    query = "SELECT * FROM file_restore_requests %s " % sel
    print query

    files = __do_select_many("SELECT * FROM file_restore_requests %s " % sel)
    for restore in files:
        yield restore
