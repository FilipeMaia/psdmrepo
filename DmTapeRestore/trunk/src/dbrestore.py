
from __future__ import print_function

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

def __do_select_many(statement, param=None):

    conn = DbConnection(conn_string=CONN_STR)
    cursor = conn.cursor(dictcursor=True)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    if param:
        cursor.execute(statement, param)
    else:
        cursor.execute(statement)
    return cursor.fetchall()

def __do_sql(statement, param=None):

    conn = DbConnection(conn_string=CONN_STR)
    cursor = conn.cursor(dictcursor=True)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute("BEGIN")
    #print(statement)
    #print(param)
    cursor.execute(statement, param)
    cursor.execute("COMMIT")


DONE = 'DONE'
SUBMITTED = 'SUBMITTED'
FAILED = 'FAILED'
RECEIVED = 'RECEIVED'


def __make_sel(selection):
    """ Given a dict with selection criteria create the WHERE clause for
    the SQL query. 
    Returns a tuple (where_clause, param) with the WHERE clause and the 
    parameters used by it, e.g.: 
      ("WHERE fn like %s and runnum = %s", ("/file/name",162))
    """
    sel = []
    param = []
    for key, value in selection.iteritems():        
        if key == "fn":
            if value.find('%') >= 0:
                sel.append("irods_filepath like %s")
            else:
                sel.append("irods_filepath = %s")
        elif key == "expid":
            sel.append("exper_id = %s".format(value))
        elif key == 'runnum':
            sel.append("runnum = %s".format(value))
        elif key == 'status' and value:
            sel.append("status = %s")
        else:
            continue
        param.append(value)

    q = "WHERE {}".format(" AND ".join(sel)) if sel else ""
    return q, param

def query(selection):
    """ find restore requests """

    q,param = __make_sel(selection)
    files = __do_select_many("SELECT * FROM file_restore_requests %s" % q, param)
    return files

def files_with_status(status=None):
    return query({'status' : status})


def set_status(flag, irods_fname):
    """ set status for a request selected by the irods file name"""

    if flag:
        where, param = __make_sel({'fn' : irods_fname})
        if where:
            query = "UPDATE file_restore_requests SET status=%s {}".format(where)
            param.insert(0, flag)
            #print(query, param)
            __do_sql(query, param)
            return

    print("No value for status, update failed")

def set_submitted(irods_fname):
    query = "UPDATE file_restore_requests SET status='SUBMITTED' " \
        "WHERE irods_filepath = '%s'" % irods_fname
    __do_sql(query)
    

def file_by_name(name):
    sel = "WHERE irods_filepath like '%%%s%%'" % name
    query = "SELECT * FROM file_restore_requests %s " % sel
    print(query)

    files = __do_select_many("SELECT * FROM file_restore_requests %s " % sel)
    for restore in files:
        yield restore
