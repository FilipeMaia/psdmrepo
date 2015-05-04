#--------------------------------------------------------------------------
# File and Version Information:
#  $Id:$
#
# Description:
#  The API for publishing psanamon monitoring plots in the 'sysmon' database
#  for further viewing through the Web.
#
#------------------------------------------------------------------------

"""
The API for publishing psanamon monitoring plots in the 'sysmon' database
for further viewing through the Web.
@version $Id:$

@author Igor Gaponenko
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision:$"
# $Source:$

#--------------------------------
#  Imports of standard modules --
#--------------------------------

import sys
import time

#-----------------------------
# Imports for other modules --
#-----------------------------

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

__connection = None

def __get_connection():
    global __connection
    if __connection is None: __connection = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
    return __connection

def __escape_string(str):

    return __get_connection().escape_string(str)

def __do_select_many(statement):

    cursor = __get_connection().cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute("BEGIN")
    cursor.execute(statement)
    return cursor.fetchall()

def __do_select(statement):

    rows = __do_select_many(statement)
    if not rows : return None

    return rows[0]

# ------------------------------------------------------------------------------------------
# Execute any SQL statement which doesn't return a result set
#
# Notes:
# - exceptions are thrown exactly as explained for the previously defined method __do_select
# - the statement will be surrounded by BEGIN and COMMIT transaction statements
# ------------------------------------------------------------------------------------------

def __do_sql(statement):

    cursor = __get_connection().cursor(db.cursors.SSDictCursor)
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

# --------------
# Publish a plot
# --------------

def publish(exper_name, name, type, descr, data):

    import pwd, os
    uid = pwd.getpwuid(os.geteuid())[0] # assume current logged user

    from RegDB import experiment_info
    exper_id = experiment_info.name2id(exper_name)
    if not exper_id :
        print "No such experiment: %s" % exper_name
        return False

    table = 'psanamon_plot_m'

    name_esc  = __escape_string(name)
    type_esc  = __escape_string(type)
    descr_esc = __escape_string(descr)
    data_esc  = __escape_string(data)
    now       = __now_64()
    uid_esc   = __escape_string(uid)

    row = __do_select("SELECT update_time FROM %s WHERE exper_id=%d AND name='%s'" % (table, exper_id, name_esc,))
    if not row :
        __do_sql("INSERT INTO %s VALUES (NULL,%d,'%s','%s','%s','%s',%d,'%s')" %
                  (table, exper_id, name_esc, type_esc, descr_esc, data_esc, now, uid_esc,))
    else : 
        __do_sql("UPDATE %s SET type='%s', descr='%s', data='%s', update_time=%d, update_uid='%s' WHERE exper_id=%s AND name='%s'" %
                 (table, type_esc, descr_esc, data_esc, now, uid_esc, exper_id, name_esc))

# -------------------------------
# Here folow a couple of examples
# -------------------------------

if __name__ == "__main__" :

    if len(sys.argv) != 8:
        print "Usage: <expname> <plotname> <descr> <file-1> <mime-type-1> <file-2> <mime-type-2>"
        sys.exit(1)

    (expname, plotname, descr, filename1, mimetype1, filename2, mimetype2) = sys.argv[1:]
    
    f1 = open(filename1, 'rb')
    data1 = f1.read()

    f2 = open(filename2, 'rb')
    data2 = f2.read()

    for i in range(1,1000):
        publish(expname, plotname,mimetype1,descr,data1)
        #time.sleep(1.)
        publish(expname, plotname,mimetype2,descr,data2)

    sys.exit(0)

