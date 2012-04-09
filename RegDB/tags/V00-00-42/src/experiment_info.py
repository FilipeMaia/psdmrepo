#--------------------------------------------------------------------------
# File and Version Information:
#  $Id:$
#
# Description:
#  The API for inquering various information about instruments and
#  experiments registered at PCDS.
#
#------------------------------------------------------------------------

"""
The API for inquering various information about instruments and
experiments registered at PCDS.

This software was developed for the LCLS project.  If you use all or
part of it, please give an appropriate acknowledgment.

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
__user   = 'regdb_reader'
__passwd = ''
__db     = 'regdb'

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

# --------------------------------------------------------------------
# Get data path for an experiment. Use a numeric identifier to specify
# the experiment.
# Return None if no data path is configured for the experiment.
# --------------------------------------------------------------------

def getexp_datapath(id):
    row = __do_select("SELECT val FROM experiment_param WHERE exper_id=%s AND param='DATA_PATH'" % id)
    if not row : return None
    return row['val']


def active_experiment(instr):
    """
    Get a record for the latest experiment activated for the given instrument.
    The function will return a tuple of:

      (experiment, id, activated, user)

    Where:

      instrument - the name of the instrument

      experiment - the name of the experiment

      id - a numeric identifier of the experiment

      activated - a 64-bit integer timestamp representing a time when the experiment
                  was activated as the 'current' experiment of the instrument. The value
                  of the timestamp is calculated as: nsec + sec * 10^9

      activated_local_str - a human-readable repreentation of the activation time
                            in the local timezone

      user - a UID of a user who's requested the activation

    The function wil return None if no record database was found for the requested
    instrument name. This may also be an indication that the instrument name is not valid.
    """

    row = __do_select(
        """
        SELECT e.name AS `name`, e.id AS `id`, sw.switch_time AS `switch_time`, sw.requestor_uid AS `requestor_uid` FROM expswitch sw, experiment e, instrument i
        WHERE sw.exper_id = e.id AND e.instr_id = i.id AND i.name='%s' ORDER BY sw.switch_time DESC LIMIT 1
        """ % instr)

    if not row : return None

    timestamp = int(row['switch_time'])
    return (instr, row['name'], int(row['id']), timestamp, time.ctime(timestamp / 1e9), row['requestor_uid'])

# -------------------------------
# Here folow a couple of examples
# -------------------------------

if __name__ == "__main__" :

    try:
        print 'experiment id 47 translates into %s' % id2name(47)
        print 'experiment sxrcom10 translates into id %d' % name2id('sxrcom10')
        print 'data path for experiment id 116 set to %s' % getexp_datapath(116)
        print 'current time is %d nanoseconds' % __now_64()

        print """
 -------+------------+------+------------------------------------------------+----------
        |            |      |            activation time                     |
  instr | experiment |   id +---------------------+--------------------------+ by user
        |            |      |         nanoseconds | local timezone           |
 -------+------------+------+---------------------+--------------------------+----------"""

        for instr in ('AMO','SXR','XPP','XCS','CXI','MEC','XYZ'):
            exp_info = active_experiment(instr)
            if exp_info is None:
                print "  %3s   | *** no experiment found in the database for this instrument ***" % instr
            else:
                print "  %3s   | %-10s | %4d | %19d | %20s | %-8s" % exp_info

        print ""

    except db.Error, e:
         print 'MySQL operation failed because of:', e
         sys.exit(1)

    sys.exit(0)

