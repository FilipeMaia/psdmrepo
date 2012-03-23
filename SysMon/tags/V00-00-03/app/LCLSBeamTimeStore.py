#!/usr/bin/env python

"""
This script will read a name and a value of an EPICS PV from
input arguments and store th evalue in the database.

In addition, the script provides a number of (in principle) general
purpose methods to simplify operations with MySQL database:

    __escape_string(str)
    __do_select(statement)
    __do_select_many(statement)  
    __do_sql(statement)

Each of those methods will establish a new connection with the database
server. Please, make sure the calling code is properly handling MySQL
exceptions as shown below:

    try:
        result = __do_select('SELECT...')
        except db.Error, e:
            print 'MySQL connection failed: '.str(e)
            ...

In addition, the module provides one function for generating a high-resolution
timestamp for the current time:

    __now_64()

And three functions for retreiving and storing values of EPICS PV
in the monitoring database:

    get_pv_id(pv,scope)
    get_pv_value_last(pv,scope)
    store_pv(pv,scope,value,force)

See further details in the code of the script.
"""

import sys
import time

import MySQLdb as db

# ---------------------------------------
# Default connection parameters for MySQL
# ---------------------------------------

__host   = 'psdb.slac.stanford.edu'
__user   = 'sysmon'
__passwd = 'pcds'
__db     = 'sysmon'

# ------------------------------------------------------------
# Utility functions for operations with MySQL and manipulating
# high resolution timestamps.
# ------------------------------------------------------------

def __escape_string(str):

    conn = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
    return conn.escape_string(str)

def __do_select(statement):

    """
    Allows to return only one value from a result set of
    the specified query statement.
    """ 

    conn = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
    cursor = conn.cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute(statement)
    rows = cursor.fetchall()
    if not rows : return None

    return rows[0]

def __do_select_many(statement):

    """
    Allows to retun many values from a result set of the specified
    query statement.
    """

    conn = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
    cursor = conn.cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute(statement)
    return cursor.fetchall()

def __do_sql(statement):

    """
    Execute any SQL statement which doesn't return a result set
    
    Notes:
    - exceptions are thrown exactly as explained for the previously defined method __do_select
    - the statement will be surrounded by BEGIN and COMMIT transaction statements
    """

    conn = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
    cursor = conn.cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute("BEGIN")
    cursor.execute(statement)
    cursor.execute("COMMIT")

def __now_64():

    """
    Return the current time expressed in nanoseconds. The result
    will be packed into a 64-bit number.
    """

    t = time.time()
    sec = int(t)
    nsec = int(( t - sec ) * 1e9 )
    return sec*1000000000L + nsec

def get_pv_id(pv,scope):

    """
    Check if the specified PV is already known in a given scope. Create
    it if not found. Return its identifier.
    """

    row = __do_select("SELECT id FROM pv WHERE name='%s' AND scope='%s'" % (pv,scope,))
    if row : return int(row['id'])
    __do_sql("INSERT INTO pv VALUES(NULL,'%s','%s')" % (pv,scope,))
    return get_pv_id(pv,scope)

def get_pv_value_last(pv,scope):

   """
   Get the latest value of the PV. If the one exists then return a list
   representing the finding:

   (pv,scope,timestamp,value)

   Otherwise return None
   """

   id = get_pv_id(pv,scope)
   row = __do_select("SELECT timestamp,value FROM pv_val WHERE pv_id=%d AND timestamp=(SELECT MAX(timestamp) FROM pv_val WHERE pv_id=%d)" % (id,id,))
   if not row: return None
   return (pv,scope,int(row['timestamp']),row['value'],)

def store_pv(pv,scope,value,force=False,last_records2keep=None):

    """
    Store a value of the PV if either of the following is true:

    1. the method is invoked in the 'force' mode
    2. no value for the PV has been stored in the database
    3. the last stored value differs from the new one

    The method will always return the latest value stored in the database,
    regardeless if it was stored by the current invocation of the method
    or it's been loaded from the database. The result list returned by
    the method is the same as explained in function get_pv_value_last().

    Optionally, if the last parameters is provided the function will
    all so clean all previous entries of the PV from the database to keep
    no more than the number specified in the parameter. 

    """

    last_entry = get_pv_value_last(pv,scope)
    value_as_str = "%s" % value
    if force or (last_entry is None) or last_entry[3] != value_as_str:
        id = get_pv_id(pv,scope)
        now = __now_64()
        __do_sql("INSERT INTO pv_val VALUES(%d,%d,'%s')" % (id,now,__escape_string(value_as_str)))
        last_entry = get_pv_value_last(pv,scope)

    if last_records2keep:
        id = get_pv_id(pv,scope)
        rows = __do_select_many("SELECT timestamp FROM pv_val WHERE pv_id=%d ORDER BY timestamp DESC" % (id,))
        if len(rows) > last_records2keep:
            first_timestamp_2keep = rows[last_records2keep]['timestamp']
            __do_sql("DELETE FROM pv_val WHERE pv_id=%d AND timestamp <= %d" % (id,first_timestamp_2keep,))

    return last_entry

# -------------------------
# Here is the actual script
# -------------------------

def usage_and_exit(msg=None):
    if msg is not None: print msg
    print "usage: %s <pvname> <value> [-force][-keep <last_records>]" % sys.argv[0]
    sys.exit(1)

if __name__ == '__main__':

    if len(sys.argv) < 3:
        usage_and_exit()

    pvname,value = sys.argv[1:3]

    force = False
    last_records2keep = None  ## keep them all

    if len(sys.argv) > 3:
        numArgs = len(sys.argv) - 3
        nextArg = 3
        while numArgs:

            optName = sys.argv[nextArg]
            numArgs = numArgs - 1
            nextArg = nextArg + 1

            if optName == '-force':
                force = True

            elif optName == '-keep':

                if not numArgs:
                    usage_and_exit("missing value for option %s" % optName)
                optValue = int(sys.argv[nextArg])

                last_records2keep = int(sys.argv[nextArg])
                numArgs = numArgs - 1
                nextArg = nextArg + 1

                if not last_records2keep:
                    usage_and_exit("option %s should have a non-zero value" % optName)

            else:
                usage_and_exit("unknown option %s" % optName)

    scope = 'BeamTime'

    try:
    	store_pv(pvname,scope,value,force,last_records2keep)

    except db.Error, e:
        print 'MySQL operation failed because of:', e
        sys.exit(1)
    except Exception, e:
        print 'Exception:', e
        sys.exit(1)

    sys.exit(0)
