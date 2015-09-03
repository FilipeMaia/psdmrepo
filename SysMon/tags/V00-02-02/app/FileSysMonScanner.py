#!/usr/bin/python

# ------------------------------------------------------------------
# This script will populate the database with a full snapshot of the
# file system's catalog.
# ------------------------------------------------------------------

import sys, os, subprocess, time

from stat import *

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
#               __begin()
#               result = __do_select('SELECT...')
#               __commit()
#           except db.Error, e:
#               print 'MySQL connection failed: '.str(e)
#               ...
#
# ------------------------------------------------------------------------------

__conn = None
__cursor = None


def __connect () :

    global __conn, __cursor

    if __conn   is None : __conn   = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
    if __cursor is None : __cursor = __conn.cursor(db.cursors.SSDictCursor)

    __cursor.execute("SET SESSION SQL_MODE='ANSI'")

def __escape_string (str) :

    __connect()

    global __conn

    return __conn.escape_string(str)

def __begin () :

    __connect()

    global __cursor

    __cursor.execute("BEGIN")

def __commit () :

    __connect()

    global __cursor

    __cursor.execute("COMMIT")

# -----------------------------------------------------------
# Execute any SQL statement which doesn't return a result set
# -----------------------------------------------------------

def __do_sql(statement):

    __connect()

    global __cursor

    __cursor.execute(statement)

# -------------------------------------------------------
# Execute any SQL statement which returns one row or None
# -------------------------------------------------------

def __do_select(statement) :

    __connect()

    global __cursor

    __cursor.execute(statement)
    rows = __cursor.fetchall()
    if not rows : return None
    else :        return rows[0]

# ----------------------------------------------------------
# Execute any SQL statement which returns one number or None
# ----------------------------------------------------------

def __do_select_number(statement, colname) :

    row =  __do_select(statement)
    if not row : return None

    return int(row[colname])


# ------------------------------------------------------------
# Return the current time expressed in nanoseconds. The result
# will be packed into a 64-bit number.
# ------------------------------------------------------------

def __now_64() :
    t = time.time()
    sec = int(t)
    nsec = int(( t - sec ) * 1e9 )
    return sec*1000000000L + nsec

def __now_sec() :
    t = time.time()
    return int(t)


# ------------------------------------------------------
# Functions to store catalog information in the database
# ------------------------------------------------------

__verbose = False
__file_type2id = dict()  # this dictionary is used to avoid duplicates in file types
__all_files    = dict()  # this dictionary has full path names of all files to avoid duplicated
                         # files found in GlusterFS file systems.

def store_dir (pathname, name, stat, file_system_id, parent_id=None) :

    global __verbose
    if __verbose :
        print 'dir:     %s' % pathname

    parent_id_opt = 'NULL'
    if parent_id is not None : parent_id_opt = parent_id

    __do_sql("INSERT INTO file_catalog VALUES(NULL,%s,%s,'%s','DIR',NULL,NULL,%d,%d,%d,%d,%d,%d)" % (parent_id_opt, file_system_id, __escape_string(name), stat.st_uid, stat.st_gid, stat.st_size, stat.st_atime, stat.st_ctime, stat.st_mtime))

    id =  __do_select_number('SELECT LAST_INSERT_ID() AS id', 'id')
    if id is None :
        print 'store_dir:  failed to retreive a numeric identifier of the directory'
        sys.exit(1)
    return id

def store_link (pathname, name, stat, file_system_id, parent_id=None) :

    global __verbose
    if __verbose :
        print 'link:    %s' % pathname

    parent_id_opt = 'NULL'
    if parent_id is not None : parent_id_opt = parent_id

    __do_sql("INSERT INTO file_catalog VALUES(NULL,%s,%s,'%s','LINK',NULL,NULL,%d,%d,%d,%d,%d,%d)" % (parent_id_opt, file_system_id, __escape_string(name), stat.st_uid, stat.st_gid, stat.st_size, stat.st_atime, stat.st_ctime, stat.st_mtime))

def store_file (pathname, name, stat, file_system_id, parent_id=None) :

    global __verbose
    if __verbose :
        print 'file:    %s' % pathname

    parent_id_opt = 'NULL'
    if parent_id is not None : parent_id_opt = parent_id

    extension = ''
    if len(name) > 2 :
      idx = name.rfind('.', 1)
      if (idx != -1) and (len(name) - 1 - idx > 0) : extension = name[idx+1:]

    file_type_id = None
    try :
        file_type_tester = subprocess.Popen(['/usr/bin/file','-b',pathname],stdout=subprocess.PIPE)
        for line in file_type_tester.stdout :
            file_type = line[:-1]
            file_type_id = None
            global __file_type2id
            if file_type in __file_type2id : file_type_id = __file_type2id[file_type]
            else :
                __do_sql("INSERT INTO file_type VALUES(NULL,%d,'%s')" % (file_system_id,__escape_string(file_type)))
                file_type_id =  __do_select_number('SELECT LAST_INSERT_ID() AS id', 'id')
                __file_type2id[file_type] = file_type_id
    except OSError, e :
        pass
    file_type_id_opt = 'NULL'
    if file_type_id is not None : file_type_id_opt = file_type_id

    __do_sql("INSERT INTO file_catalog VALUES(NULL,%s,%s,'%s','FILE','%s',%d,%d,%d,%d,%d,%d,%d)" % (parent_id_opt, file_system_id, __escape_string(name), __escape_string(extension), file_type_id_opt, stat.st_uid, stat.st_gid, stat.st_size, stat.st_atime, stat.st_ctime, stat.st_mtime))

def store_other (pathname, name, stat, file_system_id, parent_id=None) :

    global __verbose
    if __verbose :
        print 'other:   %s' % pathname

    parent_id_opt = 'NULL'
    if parent_id is not None : parent_id_opt = parent_id

    __do_sql("INSERT INTO file_catalog VALUES(NULL,%s,%s,'%s','OTHER',NULL,NULL,%d,%d,%d,%d,%d,%d)" % (parent_id_opt, file_system_id, __escape_string(name), stat.st_uid, stat.st_gid, stat.st_size, stat.st_atime, stat.st_ctime, stat.st_mtime))

def store_error (pathname, name, file_system_id, parent_id=None) :

    global __verbose
    if __verbose :
        print 'OSError: %s' % pathname

    parent_id_opt = 'NULL'
    if parent_id is not None : parent_id_opt = parent_id

    __do_sql("INSERT INTO file_catalog VALUES(NULL,%s,%s,'%s','ERROR',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)" % (parent_id_opt, file_system_id, __escape_string(name)))

def store_file_system (pathname):
    global __verbose
    if __verbose :
        print 'file_system: %s' % pathname
    now = __now_sec()
    __do_sql("INSERT INTO file_system VALUES(NULL,'%s',%d)" % (__escape_string(pathname), now))
    file_system_id = __do_select_number('SELECT LAST_INSERT_ID() AS id', 'id')
    if file_system_id is None :
        print 'store_file_system: failed to retreive a numeric identifier of the file system'
        sys.exit(1)
    return file_system_id

# --------------------------------------------------------------
# This function will recurively walk a file system tree and call
# corresponding handlers for the file system entries found.
#
# NOTE: The algorithm will NOT propagate its search along
#       symbolic links.
# -------------------------------------------------------------

def walktree (dir, file_system_id, parent_id=None) :

    for f in os.listdir(dir) :

        pathname = os.path.join(dir, f)
        if pathname in __all_files : continue  ## skip duplicate file entries allowed by some file systems
        __all_files[pathname] = True

        stat = None
        try :
            stat = os.lstat(pathname)
        except OSError, e :
            store_error(pathname, f, file_system_id, parent_id)
        if stat is not None :
            mode = stat.st_mode
            if   S_ISDIR(mode) : walktree(pathname, file_system_id, store_dir  (pathname, f, stat, file_system_id, parent_id))
            elif S_ISLNK(mode) :                                    store_link (pathname, f, stat, file_system_id, parent_id)
            elif S_ISREG(mode) :                                    store_file (pathname, f, stat, file_system_id, parent_id)
            else :                                                  store_other(pathname, f, stat, file_system_id, parent_id)

if __name__ == '__main__' :

    if len(sys.argv) < 2 :
        print  'usage: <base_path>i [-v]'
        sys.exit(1)

    base_path = os.path.abspath(sys.argv[1])

    __verbose = (len(sys.argv) > 2) and (sys.argv[2] == '-v') 

    try :
        __begin()
        file_system_id = store_file_system(base_path)
        walktree(base_path, file_system_id)
        __commit()
    except db.Error, e :
        print 'MySQL connection failed: ',e
        sys.exit(1)

    sys.exit(0)

