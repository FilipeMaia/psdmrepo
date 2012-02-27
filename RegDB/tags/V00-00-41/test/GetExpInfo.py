# ------------------------------------------------------------------
# Find below utilities fetching various information from Experiments
# Registration database.
# ------------------------------------------------------------------

import sys
import MySQLdb as db

__host = 'psdb.slac.stanford.edu'
__user = 'regdb_reader'
__db   = 'regdb'

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

def __do_select(statement):

    # connect to the server and set up the cursor which would return
    # each result row as a dictionary rather than a list.

    conn = db.connect(host=__host, user=__user, db=__db)
    cursor = conn.cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")

    # Proceed with the request

    cursor.execute(statement)
    rows = cursor.fetchall()
    if not rows : return None

    return rows[0]

# ---------------------------------------------------------------------
# Look for an experiment with specified identifier and obtain its name.
# Return None if no such experiment exists in the database.
# ------------------------------------------------------------------------------

def id2name(id):

    row = __do_select("SELECT name FROM experiment WHERE id=%s" % id)
    if not row : return None
    return row['name']

# -----------------------------------------------------------------------------
# Look for an experiment with specified name and obtain its numeric identifier.
# Return None if no such experiment exists in the database.
# ------------------------------------------------------------------------------

def name2id(name):

    row = __do_select("SELECT id FROM experiment WHERE name='%s'" % name)
    if not row : return None
    return int(row['id'])


# -------------------------------
# Here folow a couple of examples
# -------------------------------

if __name__ == "__main__" :

    try:
        print 'experiment id 47 translates into %s' % id2name(47)
        print 'experiment sxrcom10 translates into id %d' % name2id('sxrcom10')

    except db.Error, e:
         print 'MySQL operation failed because of:', e
         sys.exit(1)

    sys.exit(0)

