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
from InterfaceCtlr.Config import Config

#----------------------------------
# Local non-exported definitions --
#----------------------------------

class _DatabaseOperatonFailed(Exception):
    def __init__ (self, message):
        Exception.__init__( self, message )

# decorator for locking
def _synchronized(fun):
    
    def wrp( *args, **kwargs ) :
        
        _self = args[0]
        _self._lock.acquire()
        try :
            return fun( *args, **kwargs )
        finally :
            _self._lock.release()
        
    return wrp

# decorator for transaction
def _transaction(fun):
    
    def wrp( *args, **kwargs ) :
        
        _self = args[0]
        cursor = _self._conn.cursor()
        commit_or_abort = "ROLLBACK"
        try :
            cursor.execute("START TRANSACTION")
            newkw = kwargs.copy()
            newkw['cursor'] = cursor
            res = fun( *args, **newkw )
            commit_or_abort = "COMMIT"
            return res                
        finally :
            cursor.execute(commit_or_abort)
        
    return wrp

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
    def __init__ ( self, conn, log=None ) :
        """Constructor.

        @param conn      database connection object
        """

        self._conn = conn
        self._log = log or logging.getLogger()
        self._lock = threading.Lock()

    # ==============================================
    # Define new controller instance in the database
    # ==============================================

    @_synchronized
    @_transaction
    def new_controller(self, host, log, cursor=None):
        """
        Define new controller in the database, return new controller object
        which is a dictionary with these keys:
            - id
            - translate_uri
            - log_uri
        """

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
        proc_id = os.getpid()
        start  = Time.now()

        # get allowed instruments
        cursor.execute( """SELECT instrument FROM node2instr WHERE translator_node_id = %s""", (xlatenode_id,) )
        instruments = [row[0] for row in cursor.fetchall()]

        # define new controller instance
        cursor.execute("""INSERT INTO interface_controller 
            (id, fk_translator_node, process_id, kill_ic, started, log)
            VALUES(NULL, %s, %s, False, %s, %s)""", 
            ( xlatenode_id, proc_id, start.toString("%F %T"), log ) )
        cursor.execute("SELECT LAST_INSERT_ID()")
        rows = cursor.fetchall()
        controller_id = rows[0][0]
        
        return dict ( id=controller_id, 
                      instruments=instruments,
                      translate_uri=translate_uri, 
                      log_uri=log_uri)

    @_synchronized
    @_transaction
    def controller_status(self, id=None, cursor=None):
        """
        Returns list describing controller status. Every list item describes separate host
        which is in active state.
        """

        if id is None :
            # get the latest controller for every active node
            q = """SELECT c.id, n.node_uri, c.started, c.stopped, c.process_id, c.log
                FROM interface_controller c, translator_node n, 
                    (SELECT fk_translator_node nid, max(started) started 
                     FROM interface_controller 
                     GROUP BY nid) cmax 
                WHERE n.id = c.fk_translator_node AND c.fk_translator_node = cmax.nid 
                     AND c.started = cmax.started AND n.active"""
            vars = ()
        else :
            q = """SELECT c.id, n.node_uri, c.started, c.stopped, c.process_id, c.log
                FROM interface_controller c, translator_node n
                WHERE n.id = c.fk_translator_node AND c.id = %s"""
            vars = (id,)
        
        cursor.execute(q, vars)
        
        res = []
        for row in cursor.fetchall():
            cid, host, started, stopped, pid, log = row
            status = 'Running'
            if stopped : 
                stopped = str(stopped)
                status = 'Stopped'
            res.append( dict ( id=cid, host=host, pid=pid, started=str(started), stopped=stopped, status=status, log=log) )
            
        return res


    # ===========================================
    # get configuration information from database
    # ===========================================

    def __get_config(self, section, cursor):
        """ read single configuration section from database """
        
        q = "SELECT param, value, type, instrument, experiment FROM config_def WHERE section=%s"
        cursor.execute( q, ( section, ) )
        res = cursor.fetchall()
        cursor.execute("COMMIT")

        parsers = { 'Integer':   int,
                    'Float':     float,
                    'Date/Time': Time.parse }

        config = Config()
        for row in res :
            param = row[0]
            value = parsers.get(row[2], lambda x: x)(row[1])
            instr = row[3]
            exper = row[4]
            config.add(param, value, instr, exper)

        return config

    # =========================================
    # read complete configuration from database
    # =========================================

    @_synchronized
    @_transaction
    def read_config (self, sections, verbose=0, cursor=None):
        """ read all configuration sections from database """

        # throw away all we got
        fullconfig = Config()

        # get configuration from database
        for section in sections :
            
            # read section from database
            config = self.__get_config(section,cursor)
            if verbose :
                self._log.info ( 'config[%s] = %s', section, config )

            # merge configurations
            fullconfig.merge(config)
        
        return fullconfig

    # ====================================
    # Get fileset with requested status id
    # ====================================

    @_synchronized
    @_transaction
    def get_fileset ( self, status, instruments=None, cursor=None ) :

        """return a fileset id with the specified status or None if
        no fileset exists with that status"""
        
        fs = None
        
        # find a matching fileset and lock it for update
        q = """SELECT fs.id AS id, fs.experiment, fs.instrument, run_type, run_number, stat.name as status
                    FROM fileset AS fs, fileset_status_def AS stat, active_exp AS act
                    WHERE stat.name = %s AND fs.fk_fileset_status = stat.id AND fs.locked = FALSE
                    AND fs.instrument = act.instrument AND fs.experiment = act.experiment"""
        vars = (status,)
        if instruments :
            fmt = ','.join(['%s']*len(instruments))
            q += " AND fs.instrument IN (" + fmt + ')'
            vars += tuple(instruments)
        q += " ORDER BY fs.priority DESC, fs.created ASC LIMIT 1 FOR UPDATE"

        cursor.execute(q, vars)
        rows = cursor.fetchall()
        
        if rows :
            # set lock flag 
            fs = rows[0]
            fs = dict( id=fs[0], experiment=fs[1], instrument=fs[2], 
                       run_type=fs[3], run_number=fs[4], status=fs[5] )
            try :
                cursor.execute("""UPDATE fileset SET locked = TRUE
                     WHERE fileset.id = %s""", ( fs['id'], ) )
            except Exception, ex :
                self._log.warning( "Failed to lock fileset record, retry later, set #%d" % fs['id'] )
                fs = None

            if fs :
                # get the list of files in fileset
                rows = cursor.execute("SELECT name FROM files WHERE fk_fileset_id = %s and type='XTC'", (fs['id'],) )
                rows = cursor.fetchall()
                fs['xtc_files'] = [ r[0] for r in rows ]

        return fs

    # ======================
    # Add files to a fileset
    # ======================
    
    @_synchronized
    @_transaction
    def add_files (self, fs_id, ftype, files, cursor=None):
        """ Add new files to a fileset """
    
        q = """INSERT INTO files (fk_fileset_id,name,type) VALUES (%s,%s,%s)"""
        cursor.executemany ( q, [(fs_id, name, ftype) for name in files] )


    # ================================
    # Note where the file was archived
    # ================================
    
    @_synchronized
    @_transaction
    def archive_file (self, fs_id, name, archive_dir, cursor=None) :
        
        q = """UPDATE files SET archive_dir = %s WHERE fk_fileset_id = %s AND name = %s"""
        cursor.execute ( q, (archive_dir, fs_id, name) )


    # ===================================
    # Test if this Controller should exit
    # ===================================

    @_synchronized
    @_transaction
    def test_exit_controller ( self, controller_id, cursor=None ) :
        """Check the kill field for this controller"""

        cursor.execute( "SELECT kill_ic FROM interface_controller WHERE id=%s", (controller_id,) )
        rows = cursor.fetchall()
        if rows:
            return rows[0][0]
        else:
            raise _DatabaseOperatonFailed("Failed to obtain kill field for controller id: %d" % controller_id)


    # ===================================
    # Test if this Translator should exit
    # ===================================

    @_synchronized
    @_transaction
    def test_exit_translator ( self, translator_id, cursor=None ) :
        """Check the kill field for this translator"""

        cursor.execute("SELECT kill_tp FROM translator_process WHERE id=%s", (translator_id,) )
        rows = cursor.fetchall()
        if rows:
            return rows[0][0]
        else:
            raise _DatabaseOperatonFailed("Failed to obtain kill field for translator id: %d" % translator_id)


    # ========================================================
    # Change the status of a fileset and all files it contains
    # ========================================================

    @_synchronized
    @_transaction
    def change_fileset_status ( self, fileset_id, status, cursor=None ) :

        """ change the fileset to the requested status"""

        cursor.execute("""UPDATE fileset SET fk_fileset_status = (SELECT id FROM fileset_status_def WHERE name=%s), locked = FALSE 
            WHERE fileset.id = %s""", (status, fileset_id) )

    # ================================
    # Change the priority of a request
    # ================================

    @_synchronized
    @_transaction
    def change_fileset_priority ( self, fileset_id, priority, cursor=None ) :

        """ change the fileset to the requested status"""

        cursor.execute("""UPDATE fileset SET priority = %s WHERE fileset.id = %s""", (priority, fileset_id) )


    # =====================================
    # Insert row for new translator process
    # =====================================

    @_synchronized
    @_transaction
    def new_translator ( self, ctlr_id,  fs_id, log, cursor=None ) :

        """Add a row for this translator process. Update statistics after run is complete.
        Return the id of the new row for later update"""

        cursor.execute("""INSERT INTO translator_process 
                (id, fk_interface_controller, fk_fileset, kill_tp, started, log)
                VALUES(NULL, %s, %s, False, %s, %s)""", 
                ( ctlr_id, fs_id, Time.now().toString("%F %T"), log ) )
        cursor.execute("SELECT LAST_INSERT_ID()")
        rows = cursor.fetchall()
        if not rows:
            raise _DatabaseOperatonFailed( "Failed to retrieve LAST_INSERT_ID in new_translator" )
        return rows[0][0]


    # ======================================
    # Update row info for translator process
    # ======================================

    @_synchronized
    @_transaction
    def update_translator( self, translator_id, proc_code, perf_prev, ofilesize, cursor=None) :

        """Update the row for this translator process. Update run statistics and process return
        code. We store all of the resource usage even though some are 0 for a given OS.
        Take usage - perf_prev to get usage for the last child process. """

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        diff = tuple([ usage[x]-perf_prev[x] for x in range(0, 16) ])
        
        cursor.execute("""UPDATE translator_process SET 
           stopped = %s, filesize_bytes = %s, tstatus_code = %s, 
           tru_utime = %s,  tru_stime = %s,    tru_maxrss = %s,   tru_ixrss = %s,
           tru_idrss = %s,  tru_isrss = %s,    tru_minflt = %s,   tru_majflt = %s,
           tru_nswap = %s,  tru_inblock = %s,  tru_outblock = %s, tru_msgsnd = %s,
           tru_msgrcv = %s, tru_nsignals = %s, tru_nvcsw = %s,    tru_nivcsw = %s
           WHERE id = %s """,
           ( Time.now().toString("%F %T"), ofilesize, proc_code, ) + diff + 
           (translator_id,) )
            

    # ======================================
    # Update row info for translator process
    # ======================================

    @_synchronized
    @_transaction
    def update_irods_status ( self, translator_id, status_code, cursor=None) :
        """Update the irods status in the row for this translator process."""

        cursor.execute("""UPDATE translator_process SET istatus_code = %s
               WHERE id = %s """, (status_code, translator_id) )


    # ===================
    # Exit the Controller
    # ===================

    @_synchronized
    @_transaction
    def exit_controller ( self, controller_id, cursor=None ) :
        """Update the stop time for the controller and exit"""

        endtime = Time.now().toString("%F %T")
        cursor.execute("""UPDATE interface_controller SET stopped = %s WHERE id = %s """, 
                            ( endtime, controller_id ) )


    # =================================
    # Check if the experiment is active
    # =================================

    @_synchronized
    @_transaction
    def is_exp_active ( self, instr, exp, cursor ) :
        """Returns true when experiment is active"""

        cursor.execute("SELECT 1 FROM active_exp WHERE instrument = %s AND experiment = %s", 
                            ( instr, exp ) )
        rows = cursor.fetchall()

        # non-empty means we found something
        return bool(rows)

    # =======================
    # List active experiments
    # =======================

    @_synchronized
    @_transaction
    def active_experiments ( self, cursor ) :
        """Returns list of tuples (instr,exp) for all active experiments"""

        cursor.execute("SELECT instrument, experiment, since FROM active_exp ORDER BY instrument, experiment")
        rows = cursor.fetchall()

        # non-empty means we found something
        return rows

    # ===================
    # Activate experiment
    # ===================

    @_synchronized
    @_transaction
    def activate_experiment ( self, instr, exp, cursor ) :
        """Make experiments active"""

        cursor.execute("INSERT INTO active_exp (instrument, experiment) VALUES (%s,%s)",
                       (instr, exp) )

    # =====================
    # Deactivate experiment
    # =====================

    @_synchronized
    @_transaction
    def deactivate_experiment ( self, instr, exp, cursor ) :
        """Remove experiments from active list"""

        cursor.execute("DELETE FROM active_exp WHERE instrument=%s AND experiment=%s",
                       (instr, exp) )


    # ==================
    # Create new fileset
    # ==================

    @_synchronized
    @_transaction
    def new_fileset (self, instr, exper, runnum, runtype, xtcfiles, duplicate=False, status = 'Waiting_Translation', priority=0, cursor=None ) :
        """ Register new fileset.         
            @param instr        instrument name
            @param exper        experiment name
            @param runum        run number
            @param runtype      run type, one of 'DATA' or 'CALIB'
            @param xtcfiles     list of path names
            @param duplicate    disable/enable duplicate filesets
            @param status       fileset status
            @param priority     request priority
            
            Returns ID of the new fileset.
        """
        

        # check if there is an entry already for the run
        if not duplicate :
            cursor.execute("SELECT id FROM fileset WHERE instrument = %s AND experiment = %s AND run_number = %s", 
                           (instr, exper, runnum) )
            if cursor.fetchall() :
                # there is something there already
                raise _DatabaseOperatonFailed( "fileset already exists: instr=%s exper=%s run=%d" % (instr, exper, runnum) )
           
        cursor.execute("SELECT id FROM fileset_status_def WHERE name = %s", ('Initial_Entry',) )
        row = cursor.fetchone()
        if not row :
            raise _DatabaseOperatonFailed( "No rows found in fileset_status_def with Initial_Entry" )
        stat = row[0]
           
        cursor.execute( """INSERT INTO fileset 
            (fk_fileset_status,experiment,instrument,run_type,run_number,created,locked,priority)
            VALUES (%s,%s,%s,%s,%s,NOW(),1,%s)""", ( stat, exper, instr, runtype, runnum, priority ) )
        cursor.execute('SELECT LAST_INSERT_ID()')
        row = cursor.fetchone()
        newset = row[0]
    
        # add all XTC files to the fileset
        for filename in xtcfiles:
            cursor.execute( "INSERT INTO files (fk_fileset_id,name,type) VALUES (%s,%s,%s)",
                (newset, filename, 'XTC') )
    
        cursor.execute("SELECT id FROM fileset_status_def WHERE name = %s", (status,) )
        row = cursor.fetchone()
        if not row :
            raise _DatabaseOperatonFailed( "No rows found in fileset_status_def with "+status )
        wtstat = row[0]
        
        # Set fileset status to "Waiting_Translation" and we're done
        cursor.execute( "UPDATE fileset SET fk_fileset_status=%s, locked=0 WHERE fileset.id = %s", (wtstat, newset) )

        return newset

    # ==============
    # Remove fileset
    # ==============

    @_synchronized
    @_transaction
    def remove_fileset_id (self, id, cursor=None ) :
        """ Delete existing fileset based on its ID.
            @param id        fileset id
        """
        
        # delete all translator processes for those filesets
        q = "DELETE FROM translator_process WHERE fk_fileset = %s"
        cursor.execute ( q, (id,) )
    
        # delete all files for those filesets
        q = "DELETE FROM files WHERE fk_fileset_id = %s"
        cursor.execute ( q, (id,) )

        # delete filesets
        q = "DELETE FROM fileset WHERE id = %s"
        cursor.execute ( q, (id,) )

    @_synchronized
    @_transaction
    def remove_fileset (self, instr, exper, runnum=None, cursor=None ) :
        """ Delete existing fileset.
            @param instr        instrument name
            @param exper        experiment name
            @param runum        run number, if None remove all run numbers
        """
        
        qvars = ( instr, exper )
        sel = "instrument=%s AND experiment=%s"
        if runnum is not None :
            qvars += ( runnum, )
            sel += " AND run_number=%s"
        
        # delete all translator processes for those filesets
        q = "DELETE FROM translator_process WHERE fk_fileset IN (SELECT id FROM fileset WHERE %s)" % sel
        cursor.execute ( q, qvars )
    
        # delete all files for those filesets
        q = "DELETE FROM files WHERE fk_fileset_id IN (SELECT id FROM fileset WHERE %s)" % sel
        cursor.execute ( q, qvars )

        # delete filesets
        q = "DELETE FROM fileset WHERE %s" % sel
        cursor.execute ( q, qvars )

        
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
