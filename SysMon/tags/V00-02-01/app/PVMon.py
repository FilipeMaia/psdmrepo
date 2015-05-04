#!/usr/bin/python

# --------------------------------------------------------------
# Find below utilities needed for populating MySQL database with
# values of select EPICS PVs.
# --------------------------------------------------------------

import sys
import time
import MySQLdb as db

__host   = 'psdb.slac.stanford.edu'
__user   = 'sysmon'
__passwd = 'pcds'
__db     = 'sysmon'

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

# ------------------------------------------------------------------------------------------
# Execute any SQL statement which doesn't return a result set
#
# Notes:
# - exceptions are thrown exactly as explained for the previously defined method __do_select
# - the statement will be surrounded by BEGIN and COMMIT transaction statements
# ------------------------------------------------------------------------------------------

def __do_sql(statement):

    conn = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
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

# -------------------------------------------------------------------
# Check if the specified PV is already known in a given scope. Create
# it if not found. Return its identifier.
# ---------------------------------------------------------------------

def get_pv_id(pv,scope):
    row = __do_select("SELECT id FROM pv WHERE name='%s' AND scope='%s'" % (pv,scope,))
    if row : return int(row['id'])
    __do_sql("INSERT INTO pv VALUES(NULL,'%s','%s')" % (pv,scope,))
    return get_pv_id(pv,scope)

# --------------------------------------------------------------------
# Get the latest value of the PV. If the one exists then return a list
# representing the finding:
#
#   (pv,scope,timestamp,value)
#
# Otherwise return None
# --------------------------------------------------------------------

def get_pv_value_last(pv,scope):
   id = get_pv_id(pv,scope)
   row = __do_select("SELECT timestamp,value FROM pv_val WHERE pv_id=%d AND timestamp=(SELECT MAX(timestamp) FROM pv_val WHERE pv_id=%d)" % (id,id,))
   if not row: return None
   return (pv,scope,int(row['timestamp']),row['value'],)

# --------------------------------------------------------------------
# Store a value of the PV if it's changed from the previous invocation
# of this function. Always return a list representing the last value
# regardeless if this was done before or by this invocation of the
# the function. The list will have the same members as explained
# for function get_pv_value_last().
# --------------------------------------------------------------------

def store_pv(pv,scope,value):
    last_entry = get_pv_value_last(pv,scope)
    value_as_str = "%s" % value
    if (last_entry is None) or last_entry[3] != value_as_str:
        id = get_pv_id(pv,scope)
        now = __now_64()
        __do_sql("INSERT INTO pv_val VALUES(%d,%d,'%s')" % (id,now,value_as_str))
        last_entry = get_pv_value_last(pv,scope)
    return last_entry

# -------------------------------
# Here folow a couple of examples
# -------------------------------

if __name__ == "__main__" :

    import random

    value = int(10 * random.random())  # a value in the range of [0,10)

    pv = 'xyz'
    scope = 'Test'

    try:

        print "pv '%s' at scope '%s' has id=%d" % (pv,scope,get_pv_id(pv,scope),)
        last_value = get_pv_value_last(pv,scope)
        if last_value is None:
            print "pv '%s' at scope '%s' never has never been used" % (pv,scope,)
        else:
            print "pv '%s' at scope '%s' has last value='%s' stored on %d" % (last_value[0],last_value[1],last_value[3],last_value[2],)

        store_pv(pv,scope,value)

    except db.Error, e:
         print 'MySQL operation failed because of:', e
         sys.exit(1)

    sys.exit(0)

