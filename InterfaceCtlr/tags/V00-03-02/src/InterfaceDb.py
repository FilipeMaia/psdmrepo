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
import MySQLdb as db
import logging
import time
import resource

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
    def __init__ ( self, host, port, database, user, passwd ) :
        """Constructor.

        Explanation of what it does. So it does that and that, and also 
        that, but only if x is equal to that and y is not None.

        @param x   first parameter
        @param y   second parameter
        """

        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._passwd = passwd
        
        self._conn = None

        self._log = logging.getLogger('iface_db')
    
    # ===========================================================
    # Return database connection, attempt to reconnect if dropped
    # ===========================================================

    def connection(self):
        """ Method that returns connection to a database """
        
        if self._conn :
            try :
                self._conn.ping()
            except db.OperationalError, message:
                # we lost database connection
                self._conn = None
        
        if not self._conn :
            using_password = "NO"
            if self._passwd : using_password = "YES"
            connstr = "'%s'@'%s':%s (using password: %s)" % (
                self._user,self._host, self._port, using_password)
            self._log.info( 'connecting to database: '+connstr )
        
        t0 = time.time()
        while self._conn is None :
    
            # retry several times
            try :
                self._conn = db.connect( host=self._host, port=self._port, db=self._database,
                                         user=self._user, passwd=self._passwd )
            except db.Error, ex :
                self._log.error( 'database connection failed: '+str(ex) )
                # wait max 15 seconds
                if time.time() - t0 > 15 :
                    self._log.warning( 'retry limit exceeded, abort' )
                    raise
                else :
                    os.sleep(3)
            
            if self._conn :
                
                # initialize session
                cursor = self._conn.cursor()
                cursor.execute("SET SESSION SQL_MODE='ANSI'")
            
            
        return self._conn

    #
    # Make a cursor object
    #
    def cursor (self, dict=False):
        if dict: 
            return self.connection().cursor(db.cursors.SSDictCursor)
        else:
            return self.connection().cursor()


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
            config[row[0]] = parsers.get(row[2], lambda x: x)(row[1])

        return config

    # ==============================================
    # Get the internal identifier of a fileset status
    # ==============================================

    def fstat_id(self):

        cursor = self.cursor()
        
        cursor.execute( "SELECT name, id FROM fileset_status_def" )
        rows = cursor.fetchall()
        cursor.execute( "COMMIT" )
        if rows:
            return dict( rows )
        else:
            raise _DatabaseOperatonFailed( "failed to obtain fileset status ids" )

    # ====================================
    # Get fileset with requested status id
    # ====================================

    def get_fileset ( self, stat_id ) :

        """return a fileset id with the specified status or 0 if
        no fileset exists with that status"""
        
        fs = None

        cursor = self.cursor(dict=True)
        
        cursor.execute("START TRANSACTION")
        
        # find a matching fileset and lock it for update
        cursor.execute("""SELECT id, experiment, instrument, run_type, run_number 
                    FROM fileset AS fs
                    WHERE  fs.fk_fileset_status = %s AND fs.locked = FALSE
                    ORDER BY fs.created ASC LIMIT 1 FOR UPDATE""", (stat_id,) )
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
                rows = cursor.execute("SELECT name FROM files WHERE fk_fileset_id = %s", (fs['id'],) )
                rows = cursor.fetchall()
                if not rows:
                    raise _DatabaseOperatonFailed("No files in fileset #%d" % fs['id'])
                fs['xtc_files'] = [ r[0] for r in rows ]

        cursor.execute("COMMIT")
        return fs

    # ===================================
    # Test if this Controller should exit
    # ===================================

    def test_exit_controller ( self, controller_id ) :
        """Check the kill field for this controller"""

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

    def change_fileset_status ( self, fileset_id, status_id ) :

        """ change the fileset to the requested status"""

        cursor = self.cursor()
        
        cursor.execute("START TRANSACTION")
        cursor.execute("""UPDATE fileset SET fk_fileset_status = %s, locked = FALSE 
            WHERE fileset.id = %s""", (status_id, fileset_id) )
        cursor.execute("""UPDATE files SET fk_fileset_status = %s 
            WHERE files.fk_fileset_id = %s""", (status_id, fileset_id) )
        cursor.execute("COMMIT")

    # =====================================
    # Insert row for new translator process
    # =====================================

    def new_translator ( self, ctlr_id,  fs_id) :

        """Add a row for this translator process. Update statistics after run is complete.
        Return the id of the new row for later update"""

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

        cursor = self.cursor()

        endtime = Time.now().toString("%F %T")
        cursor.execute("START TRANSACTION")
        cursor.execute("""UPDATE interface_controller SET stopped = %s WHERE id = %s """, 
                            ( endtime, controller_id ) )
        cursor.execute("COMMIT")



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
