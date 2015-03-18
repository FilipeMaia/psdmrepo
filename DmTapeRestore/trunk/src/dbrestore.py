

import MySQLdb as db


__host   = 'psdb.slac.stanford.edu'
__user   = 'data_migration'
__passwd = 'pcds'
__db     = 'webportal'


def __escape_string(str):

    conn = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
    return conn.escape_string(str)

def __do_select(statement):

    conn = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
    cursor = conn.cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute(statement)
    rows = cursor.fetchall()
    if not rows : return None

    return rows[0]

def __do_select_many(statement):

    conn = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
    cursor = conn.cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute(statement)
    return cursor.fetchall()

def __do_sql(statement):

    conn = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
    cursor = conn.cursor(db.cursors.SSDictCursor)
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

    #print query    
    __do_sql(query)
    


def set_submitted(irods_fname):
    query = "UPDATE file_restore_requests SET status='SUBMITTED' " \
        "WHERE irods_filepath = '%s'" % irods_fname

    #print query    
    __do_sql(query)
    

def file_by_name(name):

    sel = "WHERE irods_filepath like '%%%s%%'" % name
    query = "SELECT * FROM file_restore_requests %s " % sel
    print query

    files = __do_select_many("SELECT * FROM file_restore_requests %s " % sel)
    print files
    
