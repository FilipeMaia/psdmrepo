#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module InterfaceDb...
#
#------------------------------------------------------------------------

""" Interface class for InterfaceDb.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import logging
import resource
import threading
from pprint import *

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from LusiTime.Time import Time

#----------------------------------
# Local non-exported definitions --
#----------------------------------

class _DatabaseOperatonFailed(Exception):
    def __init__ (self, message):
        Exception.__init__( self, message )

class _RAIILock(object):
    def __init__(self,lock):
        self._lock = lock
        self._lock.acquire()
        
    def __del__(self):
        self._lock.release()

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class InterfaceDb ( object ) :

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, conn, log ) :
        """Constructor.

        @param conn      database connection object
        """

        self._conn = conn

        self._log = log

        self.__lock = threading.Lock()

    def getlock(self) :
        return _RAIILock(self.__lock)
    
    # ===========================================================
    # Return database connection, attempt to reconnect if dropped
    # ===========================================================


    #
    # Make a cursor object
    #
    def cursor (self, dict=False):
        return self._conn.cursor(dict)


    # ==============================================
    # Define new controller instance in the database
    # ==============================================

    def new_controller(self, host):
        """
        Define new controller in the database, return new controller object
        which is a dictionary with these keys:
            - id
            - translate_uri
            - log_uri
        """

        # critical sections begins
        lock = self.getlock()
        
        cursor = self.cursor()
        
        cursor.execute("START TRANSACTION")
        # find node ID
        cursor.execute( """SELECT id, translate_uri,log_uri FROM translator_node WHERE 
                        (node_uri = %s AND active = 1)""", (host,) )
        rows = cursor.fetchall()
        if not rows :
            raise _DatabaseOperatonFailed("Failed to find node %s in translator_node table." % host)
        if len(rows) > 1:
            raise _DatabaseOperatonFailed("Too many rows in translator_node table for node %s" % host)

        row = rows[0]
        xlatenode_id = row[0]
        translate_uri = row[1]
        log_uri = row[2]
        proc_id= os.getpid()
        start  = Time.now()

        # define new controller instance
        cursor.execute("""INSERT INTO interface_controller 
            (id, fk_translator_node, process_id, kill_ic, started)
            VALUES(NULL, %s, %s, False, %s)""", 
            ( xlatenode_id, proc_id, start.toString("%F %T") ) )
        cursor.execute("SELECT LAST_INSERT_ID()")
        rows = cursor.fetchall()
        controller_id = rows[0][0]
        cursor.execute("COMMIT")

        return dict ( id=controller_id, translate_uri=translate_uri, log_uri=log_uri)

    # ===========================================
    # get configuration information from database
    # ===========================================

    def get_config(self, section):
        
        # critical sections begins
        lock = self.getlock()
        
        cursor = self.cursor()
        
        q = "SELECT param, value, type FROM config_def WHERE section=%s"
        cursor.execute( q, ( section, ) )
        res = cursor.fetchall()
        cursor.execute("COMMIT")

        parsers = { 'Integer':   int,
                    'Float':     float,
                    'Date/Time': Time.parse }

        config = {}
        for row in res :
            param = row[0]
            value = parsers.get(row[2], lambda x: x)(row[1])
            if param.startswith('list:') :
                config.setdefault(param,[]).append(value)
            else :
                config[param] = value

        return config

    # =========================================
    # read complete configuration from database
    # =========================================

    def read_config (self, sections, verbose=False):

        # throw away all we got
        fullconfig = {}

        # get configuration from database
        for section in sections :
            
            # read section from database
            config = self.get_config(section)
            if verbose :
                self._log.trace ( 'config[%s] = %s', section, pformat(config) )
                
            # merge configurations
            for k,v in config.iteritems() :
                if k.startswith('list:') :
                    fullconfig.setdefault(k, []).extend(v)
                else :
                    fullconfig[k] = v
        
        return fullconfig

    # ====================================
    # Get fileset with requested status id
    # ====================================

    def get_fileset ( self, *status ) :

        """return a fileset id with the specified status or None if
        no fileset exists with that status"""
        
        fs = None

        # critical sections begins
        lock = self.getlock()

        cursor = self.cursor(dict=True)
        
        cursor.execute("START TRANSACTION")
        
        # find a matching fileset and lock it for update
        fmt = ','.join(['%s']*len(status))
        cursor.execute("""SELECT fs.id AS id, fs.experiment, fs.instrument, run_type, run_number, stat.name as status
                    FROM fileset AS fs, fileset_status_def AS stat, active_exp AS act
                    WHERE stat.name IN (""" + fmt + """) AND fs.fk_fileset_status = stat.id AND fs.locked = FALSE
                    AND fs.instrument = act.instrument AND fs.experiment = act.experiment
                    ORDER BY fs.created ASC LIMIT 1 FOR UPDATE""", tuple(status) )
        rows = cursor.fetchall()
        
        if rows :
            # set lock flag 
            fs = rows[0]
            try :
                cursor.execute("""UPDATE fileset SET locked = TRUE
                     WHERE fileset.id = %s""", ( fs['id'], ) )
            except db.Error, ex :
                self.warning( "Failed to lock fileset record, retry later, set #%d" % fs['id'] )
                fs = None

            if fs :
                # get the list of files in fileset
                cursor = self.cursor()
                rows = cursor.execute("SELECT name FROM files WHERE fk_fileset_id = %s and type='XTC'", (fs['id'],) )
                rows = cursor.fetchall()
                fs['xtc_files'] = [ r[0] for r in rows ]

        cursor.execute("COMMIT")
        return fs

    # ======================
    # Add files to a fileset
    # ======================
    
    def add_files (self, fs_id, type, files):
    
        # critical sections begins
        lock = self.getlock()
        
        cursor = self.cursor()
        
        q = """INSERT INTO files (fk_fileset_id,name,type) VALUES (%s,%s,%s)"""
        cursor.executemany ( q, [(fs_id,name,type) for name in files] )
        
        cursor.execute("COMMIT")


    # ================================
    # Note where the file was archived
    # ================================
    
    def archive_file (self, fs_id, name, archive_dir) :
        
        # critical sections begins
        lock = self.getlock()
        
        cursor = self.cursor()
        
        q = """UPDATE files SET archive_dir = %s WHERE fk_fileset_id = %s AND name = %s"""
        cursor.execute ( q, (archive_dir, fs_id, name) )

        cursor.execute("COMMIT")
        
    # ===================================
    # Test if this Controller should exit
    # ===================================

    def test_exit_controller ( self, controller_id ) :
        """Check the kill field for this controller"""

        # critical sections begins
        lock = self.getlock()
        
        cursor = self.cursor()
        
        cursor.execute( "SELECT kill_ic FROM interface_controller WHERE id=%s", (controller_id,) )
        rows = cursor.fetchall()
        cursor.execute("COMMIT")
        if rows:
            return rows[0][0]
        else:
            raise _DatabaseOperatonFailed("Failed to obtain kill field for controller id: %d" % controller_id)


    # ===================================
    # Test if this Translator should exit
    # ===================================

    def test_exit_translator ( self, translator_id ) :
        """Check the kill field for this translator"""

        # critical sections begins
        lock = self.getlock()

        cursor = self.cursor()
        
        cursor.execute("SELECT kill_tp FROM translator_process WHERE id=%s", (translator_id,) )
        rows = cursor.fetchall()
        cursor.execute("COMMIT")
        if rows:
            return rows[0][0]
        else:
            raise _DatabaseOperatonFailed("Failed to obtain kill field for translator id: %d" % translator_id)
        return


    # ========================================================
    # Change the status of a fileset and all files it contains
    # ========================================================

    def change_fileset_status ( self, fileset_id, status ) :

        """ change the fileset to the requested status"""

        # critical sections begins
        lock = self.getlock()

        cursor = self.cursor()
        
        cursor.execute("START TRANSACTION")
        cursor.execute("""UPDATE fileset SET fk_fileset_status = (SELECT id FROM fileset_status_def WHERE name=%s), locked = FALSE 
            WHERE fileset.id = %s""", (status, fileset_id) )
        cursor.execute("COMMIT")

    # =====================================
    # Insert row for new translator process
    # =====================================

    def new_translator ( self, ctlr_id,  fs_id ) :

        """Add a row for this translator process. Update statistics after run is complete.
        Return the id of the new row for later update"""

        # critical sections begins
        lock = self.getlock()

        cursor = self.cursor()

        cursor.execute("START TRANSACTION")
        cursor.execute("""INSERT INTO translator_process 
                (id, fk_interface_controller, fk_fileset, kill_tp, started)
                VALUES(NULL, %s, %s, False, %s)""", 
                ( ctlr_id, fs_id, Time.now().toString("%F %T") ) )
        cursor.execute("SELECT LAST_INSERT_ID()")
        rows = cursor.fetchall()
        cursor.execute("COMMIT")

        if not rows:
            raise _DatabaseOperatonFailed( "Failed to retrieve LAST_INSERT_ID in new_translator" )

        return rows[0][0]

    # ======================================
    # Update row info for translator process
    # ======================================

    def update_translator( self, translator_id, proc_code, perf_prev, ofilesize) :

        """Update the row for this translator process. Update run statistics and process return
        code. We store all of the resource usage even though some are 0 for a given OS.
        Take usage - perf_prev to get usage for the last child process. """

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        diff = tuple([ usage[x]-perf_prev[x] for x in range(0,16) ])
        
        # critical sections begins
        lock = self.getlock()

        cursor = self.cursor()
        
        cursor.execute("START TRANSACTION")
        cursor.execute("""UPDATE translator_process SET 
            stopped = %s, filesize_bytes = %s, tstatus_code = %s, 
           tru_utime = %s,  tru_stime = %s,    tru_maxrss = %s,   tru_ixrss = %s,
           tru_idrss = %s,  tru_isrss = %s,    tru_minflt = %s,   tru_majflt = %s,
           tru_nswap = %s,  tru_inblock = %s,  tru_outblock = %s, tru_msgsnd = %s,
           tru_msgrcv = %s, tru_nsignals = %s, tru_nvcsw = %s,    tru_nivcsw = %s
           WHERE id = %s """,
           ( Time.now().toString("%F %T"), ofilesize, proc_code, ) + diff + 
           (translator_id,) )
        cursor.execute("COMMIT")

    # ======================================
    # Update row info for translator process
    # ======================================

    def update_irods_status ( self, translator_id, status_code) :
        """Update the irods status in the row for this translator process."""

        # critical sections begins
        lock = self.getlock()
        
        cursor = self.cursor()
        
        cursor.execute("START TRANSACTION")
        cursor.execute("""UPDATE translator_process SET istatus_code = %s
               WHERE id = %s """, (status_code, translator_id) )
        cursor.execute("COMMIT")


    # ===================
    # Exit the Controller
    # ===================

    def exit_controller ( self, controller_id ) :
        """Update the stop time for the controller and exit"""

        # critical sections begins
        lock = self.getlock()
        
        cursor = self.cursor()

        endtime = Time.now().toString("%F %T")
        cursor.execute("START TRANSACTION")
        cursor.execute("""UPDATE interface_controller SET stopped = %s WHERE id = %s """, 
                            ( endtime, controller_id ) )
        cursor.execute("COMMIT")

    # =================================
    # Check if the experiment is active
    # =================================

    def is_exp_active ( self, instr, exp ) :
        """Returns true when experiment is active"""

        # critical sections begins
        lock = self.getlock()
        
        cursor = self.cursor()

        cursor.execute("START TRANSACTION")
        cursor.execute("SELECT 1 FROM active_exp WHERE instrument = %s AND experiment = %s", 
                            ( instr, exp ) )
        rows = cursor.fetchall()
        cursor.execute("COMMIT")

        # non-empty means we found something
        return bool(rows)



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
