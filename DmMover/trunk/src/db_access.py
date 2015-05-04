#!/usr/bin/python

# ------------------------------------------------------------------
# Find below utilities needed by the ONLINE-to-Lustre data migration
# service.
# ------------------------------------------------------------------

import sys
import time
import MySQLdb as db

__host   = None
__user   = None
__passwd = None
__db     = None

__db_conn = "/reg/g/psdm/psdatmgr/datamigr/.mvrdb-conn"
__dmtable = "data_migration"

# ------------------------------------------------------------------------
# Connect to MySQL server and execute the specified SELECT statement which
# is supposed to return a single row (if it return more then simply ignore
# anything before the first one). Return result as a dictionary. Otherwise return None.
#
#   NOTE: this method won't catch MySQL exceptions. It's up to
#         the caller to do so. See the example below:
#
#           try:
#               result = __do_select('SELECT...')
#           except db.Error, e:
#               print 'MySQL connection failed: '.str(e)
#               ...
#
# ------------------------------------------------------------------------------

# database connection and table selection

def __connect_db():
    """ Connect to database. Use default if no connection param were given"""
    if not __db:
        select_db(__db_conn)
    return db.connect(host=__host, user=__user, passwd=__passwd, db=__db)

def select_db(conn_file):
    """ Set database connection parameter"""
    global __host, __user, __passwd, __db
    line = open(conn_file).readline().rstrip()
    key_value = dict([x.split('=', 1) for x in line.split(';') if x])

    __host = key_value['Server']
    __user = key_value['Uid']
    __passwd = key_value.get('Pwd', "") 
    __db = key_value['Database']
    

def table_dm():
    """ Select the data_migration table for ffb datamovers """
    global __dmtable
    __dmtable = "data_migration"

def table_dm_ana():
    """ Select the data_migration table for ffb datamovers """
    global __dmtable
    __dmtable = "data_migration_ana"

def table_dm_ffb():
    """ Select the data_migration table for ffb datamovers """
    global __dmtable
    __dmtable = "data_migration_ffb"

# ----------------
# query functions
# ----------------

def __escape_string(str):
    conn = __connect_db()
    return conn.escape_string(str)

def __do_select(statement):
    conn = __connect_db()
    cursor = conn.cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute(statement)
    rows = cursor.fetchall()
    if not rows : return None
    return rows[0]

def __do_select_many(statement):
    conn = __connect_db()
    cursor = conn.cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute(statement)
    return cursor.fetchall()

# ------------------------------------------------------------------------------------------
# Execute any SQL statement which doesn't return a result set
#
# Notes:
# - exceptions are thrown exactly as explained for the previously defined method __do_select
# - the statement will be surrounded by BEGIN and COMMIT transaction statements
# ------------------------------------------------------------------------------------------

def __do_sql(statement):
    conn = __connect_db()
    cursor = conn.cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute("BEGIN")
    cursor.execute(statement)
    cursor.execute("COMMIT")

# ------------------------------------------------------------
# Return the current time expressed in nanoseconds. The result
# will be packed into a 64-bit number.
# ------------------------------------------------------------

def __now_64():
    t = time.time()
    sec = int(t)
    nsec = int(( t - sec ) * 1e9 )
    return sec*1000000000L + nsec

# ---------------------------------------------------------------------
# Look for an experiment with specified identifier and obtain its name.
# Return None if no such experiment exists in the database.
# ------------------------------------------------------------------------------

def id2name(id):
    row = __do_select("SELECT name FROM experiment WHERE id=%s" % id)
    if not row : return None
    return row['name']

# ---------------------------------------------------------------------
# Look for an experiment with specified identifier.
# Return None if no such experiment exists in the database.
# ------------------------------------------------------------------------------

def getexp(id):
    row = __do_select("SELECT * FROM experiment WHERE id=%s" % id)
    return row

# -----------------------------------------------------------------------------
# Look for an experiment with specified name and obtain its numeric identifier.
# Return None if no such experiment exists in the database.
# ------------------------------------------------------------------------------

def name2id(name):
    row = __do_select("SELECT id FROM experiment WHERE name='%s'" % name)
    if not row : return None
    return int(row['id'])

def instr4id(id):
    row = __do_select("SELECT i.name FROM instrument `i`, experiment `e` "
                      "WHERE e.id=%d AND e.instr_id=i.id" % id)
    if row:
        return row['name']
    else:
        return None

# --------------------------------------------------------------------
# Get data path for an experiment. Use a numeric identifier to specify
# the experiment.
# Return None if no data path is configured for the experiment.
# --------------------------------------------------------------------

def getexp_datapath(id):
    row = __do_select("SELECT val FROM experiment_param WHERE exper_id=%s AND param='DATA_PATH'" % id)
    if not row : return None
    return row['val']

def getexp_datapath_all():
    """ return name, id and exper-path for all experiments """

    rows = __do_select_many("select e.name, e.id, p.val from experiment `e`, experiment_param `p`"
                            "where p.exper_id=e.id and p.param='DATA_PATH'")
    return rows

# -------------------------------------
# Report the file migration start event
# -------------------------------------

def file_migration_start(exper_id, fn):
    now = __now_64()
    __do_sql("UPDATE %s SET start_time=%d, stop_time=NULL, error_msg=NULL "
             "WHERE exper_id=%d AND file='%s'" % (__dmtable, now, exper_id, fn))
    
# --------------------------------------------------
# Report the file migration stop event and error_msg
#  if error_msg='' mark file as FAIL
# --------------------------------------------------

def file_migration_stop(exper_id, fn, error_msg=None):
    now = __now_64()
    if error_msg is None:
        error_msg_sql = ", error_msg=NULL, status='DONE'"
    elif error_msg == "":
        error_msg_sql = ", error_msg='', status='FAIL'"
    else:
        error_msg_sql = ", error_msg='%s'" % __escape_string(error_msg)
            
    __do_sql("UPDATE %s SET stop_time=%d %s WHERE exper_id=%d AND file='%s'" %
             (__dmtable, now, error_msg_sql, exper_id, fn))
    return

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def files2migrate(instr=None,host=None,filetype=None):
    """ Report files which are yet to be migrated from the specified (if any provided)
    host belonging to the specified instrument (if any provided).
    Select files from instrument(s), host the file originates or filetype.
    <instr> is either a instrument name or a list of instr names.
    """
    
    exper_id_select_sql = ""
    if instr:
        if isinstance(instr,str):
            instr_q = "i.name='%s'" % instr
        elif isinstance(instr,list):
            if len(instr) == 1:
                instr_q = "i.name='%s'" % instr[0]
            else:
                instr_q = "i.name IN (%s)" % ",".join(["'%s'" % x for x in instr])
        else:
            instr_q=None
        if instr_q:
            exper_id_select_sql = "AND exper_id in (SELECT e.id FROM experiment `e`, " \
                                  "instrument `i` WHERE %s AND e.instr_id=i.id)" % instr_q
    
    host_sql = ""
    if host is not None: 
        host_sql = "AND host='%s'" % host
    ftype_sql = ""
    if filetype:
        ftype_sql = "AND file_type='%s'" % filetype
    else:
        ftype_sql = "AND file_type != '%s'" % "smd.xtc"

    select_condition = "status = 'WAIT'"
    
    files = __do_select_many("SELECT dm.exper_id,dm.file,dm.file_type,dm.host,dm.dirpath FROM %s `dm` "
                             "WHERE %s %s %s %s" % 
                             (__dmtable, select_condition, exper_id_select_sql, host_sql, ftype_sql))

    for f in files: 
        f['instrument'] = __do_select("SELECT i.name FROM instrument `i`, experiment `e` "
                                      "WHERE e.id=%d AND e.instr_id=i.id" % f['exper_id'])
    return files

# -------------------------------------
# Find index files that failed transfer
# -------------------------------------


def failed_idx_files(age=0):
    
    select = ""
    if age > 0:
        select = "AND start_time > %d" % age
        
    query = "SELECT exper_id,file,start_time FROM data_migration " \
        "WHERE file_type = 'xtc.idx' and status = 'FAIL' %s " % select

    files = __do_select_many(query)                             

    return files

# ------------------------------------
#  Get info for all experiments
# ------------------------------------

def expr_info():
    data = {}
    exprinfo = __do_select_many("SELECT name,posix_gid,instr_id from experiment")
    for info in exprinfo:
        data[info['name']] = info

    instr = {}
    instrinfo = __do_select_many("SELECT id,name from instrument")
    for info in instrinfo:
        instr[info['id']] = info['name']


    return data, instr

# ----------------------------------------------------------------
#  add new row to data_migration_ana for the ffb offline migration
# ----------------------------------------------------------------

def file4offline(exp_id, host, filetype, fname, dirpath):
    __do_sql("""INSERT INTO data_migration_ana (exper_id,file,file_type,host,dirpath)
             VALUES(%d, '%s','%s','%s','%s')""" % (exp_id, fname, filetype, host, dirpath))

# --------------------------------------------------
# Functions to keep track of file migration to NERSC
# --------------------------------------------------

def file_migration2nersc_start ( exper_id, file_name, file_type) :
    exper_id  = int (exper_id)
    file_name = __escape_string (file_name)
    file_type = __escape_string (file_type)
    now       = __now_64 ()
    file = __do_select ("SELECT * FROM data_migration_nersc WHERE exper_id=%d AND file='%s' AND file_type='%s'" % (exper_id, file_name, file_type))
    if file is None :
        __do_sql ("INSERT INTO data_migration_nersc VALUES(%d,'%s','%s',%d,NULL,NULL)" % (exper_id, file_name, file_type, now))
    else :
        __do_sql ("UPDATE data_migration_nersc SET start_time=%d, stop_time=NULL, error_msg=NULL WHERE exper_id=%d AND file='%s' AND file_type='%s'" % (now, exper_id, file_name, file_type))

def file_migration2nersc_stop ( exper_id, file_name, file_type, error_msg=None ) :
    now       = __now_64 ()
    exper_id  = int (exper_id)
    file_name = __escape_string (file_name)
    file_type = __escape_string (file_type)
    error_msg_sql = ", error_msg=NULL"
    if error_msg is not None : 
        error_msg_sql = ", error_msg='%s'" % __escape_string (error_msg)
    __do_sql ("UPDATE data_migration_nersc SET stop_time=%d %s WHERE exper_id=%d AND file='%s' AND file_type='%s'" % (now, error_msg_sql, exper_id, file_name, file_type))

def files2migrate2nersc(exper_id) :
    select_condition = "exper_id=%d AND file_type='xtc' AND stop_time IS NOT NULL AND (error_msg IS NULL OR error_msg='0' OR error_msg='')" % int(exper_id)

    # Get a list of files which have already been migrated from OFFLINE
    # to NERSC and turn them into a dictionary.

    migrated2nersc_dict = dict()
    for file in __do_select_many ("SELECT file,file_type FROM data_migration_nersc WHERE %s" % select_condition) :
        migrated2nersc_dict[file['file']] = file

    # Get a list of files which have already been migrated from DAQ to OFFLINE
    # and produce a list with subset of those which haven't been migrated to NERSC.

    files = []
    for file in __do_select_many ("SELECT file, file_type FROM data_migration WHERE %s ORDER BY file, file_type" % select_condition) :
        if file['file'] not in migrated2nersc_dict :
            files.append(file)

    return files


# -------------------------------
# Here folow a couple of examples
# -------------------------------

if __name__ == "__main__" :

    try:
        print 'experiment id 47 translates into %s' % id2name(47)
        print 'experiment sxrcom10 translates into id %d' % name2id('sxrcom10')
        print 'data path for experiment id 116 set to %s' % getexp_datapath(116)
        print 'current time is %d nanoseconds' % __now_64()

        # Note that experiment id=18 corresponds to a test experiment 'amodaq09'
        #
        #file = 'test_file_%d.txt' % __now_64()
        #file_migration_start(18,file)
        #file_migration_start(18,file)
        #time.sleep(1.0)
        #file_migration_stop(18,file)
        #time.sleep(5.0)
        #file_migration_stop(18,file,"Failed 'cause of unknown reason")
        #time.sleep(5.0)
        #file_migration_stop(18,file)
        #time.sleep(5.0)
        #file_migration_stop(18,file,"Failed 'cause of unknown reason")
        #time.sleep(5.0)
        #file_migration_start(18,file)
        #time.sleep(5.0)
        #file_migration_stop(18,file)

        print 'Files to be migrated from all instruments and hosts:'
        for f in files2migrate():
            print '  ',f

        print '...and for CXI only and all hosts:'
        for f in files2migrate('CXI'):
            print '  ',f

        print '...and for AMO only and host pslogin02:'
        for f in files2migrate('AMO','pslogin02'):
            print '  ',f


    except db.Error, e:
         print 'MySQL operation failed because of:', e
         sys.exit(1)

    sys.exit(0)

