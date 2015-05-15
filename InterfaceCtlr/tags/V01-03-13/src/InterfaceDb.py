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
import errno
import logging
import resource
import threading
import types
from pprint import *
import MySQLdb

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from LusiTime.Time import Time
from InterfaceCtlr.Config import Config
from ExpNameDb.ExpNameDatabase import ExpNameDatabase

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_expNameDb = ExpNameDatabase()

class DBConnectionError(StandardError):
    """This exception is thrown when there is a problem with database connection.
    Caller may retry operation at later time"""
    def __init__ (self, ex):
        msg = "Database connection failed: " + str(ex)
        Exception.__init__(self, msg)

class _DatabaseOperatonFailed(StandardError):
    def __init__ (self, message):
        Exception.__init__( self, message )

class ControllerInstanceError(StandardError):
    def __init__ (self, message):
        Exception.__init__( self, message )

class DatasetIdError(StandardError):
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
    """This decorator will start a transaction before calling the method and 
    will commit it after successful return. If method throws an exception 
    the transaction will be aborted.
    This wrapper will throw an exception DBConnectionError if it fails to connect
    to database, start transaction; or if the method throws MySQLdb.InternalError or
    MySQLdb.OperationalError. For other MySQLdb exception it will throw 
    _DatabaseOperatonFailed. All other exceptions will be passed upstream.
    """
    
    def wrp( *args, **kwargs ) :
        
        _self = args[0]
        
        # connect to database, this may fail
        try:
            cursor = _self._conn.cursor()
            cursor.execute("START TRANSACTION")
        except StandardError, ex:
            raise DBConnectionError(ex)
        
        try:
            try :
                newkw = kwargs.copy()
                newkw['cursor'] = cursor
                res = fun( *args, **newkw )
                cursor.execute("COMMIT")
                return res                
            except MySQLdb.OperationalError, ex:
                raise DBConnectionError(ex)
            except MySQLdb.InternalError, ex:
                raise DBConnectionError(ex)
            except MySQLdb.Error, ex:
                raise _DatabaseOperatonFailed(str(ex))
        except:
            # statement may fail if exception was already generated, catch and ignore
            try:
                cursor.execute("ROLLBACK")
            except:
                pass
            raise

    return wrp

def _checkProcess(pid):
    """Returns true if process with given PID still running"""
    try:
        os.kill(pid, 0)
    except OSError, ex:
        if ex.errno == errno.ESRCH:
            return False
    return True

#------------------------
# Exported definitions --
#------------------------

class Controller(object):
    def __init__(self, id, host, instruments):
        self.id = id
        self.host = host
        self.instruments = instruments[:]
    def __str__(self):
        return "Controller(id=%s, host=%s, instr=%s)" % (self.id, self.host, self.instruments)


class Translator(object):
    def __init__(self, id, jobid, outputDir):
        self.id = id
        self.jobid = jobid
        self.outputDir = outputDir
    def __str__(self):
        return "Translator(id=%s, jobid=%s)" % (self.id, self.jobid)

class Fileset(object):
    def __init__(self, id, experiment, instrument, run_type, run_number, status, xtc_files, translator, priority):
        self.id = id
        self.experiment = experiment
        self.instrument = instrument
        self.instrument_lower = instrument.lower()
        try:
            self.experimentId = _expNameDb.getID(instrument, experiment)
        except:
            self.experimentId = 0            
        self.run_type = run_type
        self.run_number = run_number
        self.status = status
        self.xtc_files = xtc_files[:]
        self.translator = translator
        self.priority = priority
    def __str__(self):
        jobid = None
        if self.translator: jobid = self.translator.jobid
        return "FileSet(id=%s, instr=%s, exp=%s, run=%s, status=%s, jobid=%s, files=%s)" % \
            (self.id, self.instrument, self.experiment, self.run_number, self.status, jobid, self.xtc_files)

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
        which is an instance of Controller class.
            
        @param host   Controller host name
        @param log    Log file name
            
        If there is already a controller running exception ControllerInstanceError
        will be raised.
        """

        # check if there is already an instance running
        cursor.execute("SELECT controller_id FROM active_controller FOR UPDATE")
        rows = cursor.fetchall()
        if not rows :
            raise _DatabaseOperatonFailed("Failed to fetch rows from active_controller table.")
        if len(rows) > 1:
            raise _DatabaseOperatonFailed("Too many rows in active_controller table: %d" % len(rows))
        
        active_id = rows[0][0]
        if active_id is not None:
            # if there is a running controller, if it runs on the same host then check if 
            # the process is still alive
            
            q = """SELECT n.node_uri, ic.process_id FROM translator_node n, interface_controller ic
                WHERE ic.fk_translator_node = n.id AND ic.id = %s"""
            cursor.execute(q, active_id)
            rows = cursor.fetchall()
            if not rows :
                raise _DatabaseOperatonFailed("Failed to fetch rows for active_controller instance.")
            if len(rows) > 1:
                raise _DatabaseOperatonFailed("Too many rows for active_controller instance: %d" % len(rows))

            active_host, active_pid = rows[0]
            if active_host != host or _checkProcess(active_pid):
                # stop here
                raise ControllerInstanceError("There is a controller instance running already with id=%s (host=%s, pid=%s)" % \
                                              (active_id, active_host, active_pid))
            else:
                self._log.trace("Previous active controller instance is not running any more: host=%s, pid=%s", active_host, active_pid)

        # find node ID
        cursor.execute( """SELECT id FROM translator_node WHERE (node_uri = %s AND active = 1)""", (host,) )
        rows = cursor.fetchall()
        if not rows :
            raise _DatabaseOperatonFailed("Failed to find node %s in translator_node table." % host)
        if len(rows) > 1:
            raise _DatabaseOperatonFailed("Too many rows in translator_node table for node %s" % host)

        xlatenode_id = rows[0][0]
        proc_id = os.getpid()
        start  = Time.now()

        # get allowed instruments
        cursor.execute( "SELECT instrument FROM node2instr WHERE translator_node_id = %s", (xlatenode_id,) )
        instruments = [row[0] for row in cursor.fetchall()]

        # define new controller instance
        cursor.execute("""INSERT INTO interface_controller 
            (id, fk_translator_node, process_id, kill_ic, started, log)
            VALUES(NULL, %s, %s, False, %s, %s)""", 
            ( xlatenode_id, proc_id, start.toString("%F %T"), log ) )
        cursor.execute("SELECT LAST_INSERT_ID()")
        rows = cursor.fetchall()
        controller_id = rows[0][0]
        
        # update active table
        cursor.execute("UPDATE active_controller SET controller_id = %s", (controller_id, ))
        
        return Controller(controller_id, host, instruments)

    @_synchronized
    @_transaction
    def deactivate_controller(self, controller_id, cursor=None):
        """Unlocks database by removing controller from active table so that
        another controller can start.
        
        @param[in]  controller_id  If non-None then it is checked against active ID in database.
        
        Exception is raised if controller_id does not match current active ID.
        """

        # if controller ID is provider then active ID must be the same
        if controller_id is not None:
            cursor.execute("SELECT controller_id FROM active_controller FOR UPDATE")
            rows = cursor.fetchall()
            if not rows :
                raise _DatabaseOperatonFailed("Failed to fetch rows from active_controller table.")
            if len(rows) > 1:
                raise _DatabaseOperatonFailed("Too many rows in active_controller table: %d" % len(rows))
            active_id = rows[0][0]
            if controller_id != active_id:
                raise _DatabaseOperatonFailed("stop_controller: controller IDs do not match, active ID = %s, requested ID = " % (active_id, controller_id))

        # update active table
        cursor.execute("UPDATE active_controller SET controller_id = NULL")
        

    @_synchronized
    @_transaction
    def controller_status(self, id=None, cursor=None):
        """
        Returns list describing controller status. Every list item describes separate host
        which is in active state. If id is None the only the currently active controller
        is returned.
        """

        if id is None :
            # get active controller for every active node
            cursor.execute("SELECT controller_id FROM active_controller FOR UPDATE")
            rows = cursor.fetchall()
            if rows: id = rows[0][0]

        q = """SELECT c.id, n.node_uri, c.started, c.stopped, c.process_id, c.log, n.id
            FROM interface_controller c, translator_node n
            WHERE n.id = c.fk_translator_node AND c.id = %s"""
        vars = (id,)
        
        cursor.execute(q, vars)
        
        res = []
        for row in cursor.fetchall():
            cid, host, started, stopped, pid, log, nodeid = row
            status = 'Running'
            if stopped : 
                stopped = str(stopped)
                status = 'Stopped'

            res.append( dict ( id=cid, host=host, pid=pid, started=str(started), stopped=stopped, status=status, log=log, nodeid=nodeid) )

        for ctrl in res :
            
            # get the list of instruments
            cursor.execute("SELECT instrument FROM node2instr WHERE translator_node_id = %s", (ctrl['nodeid'],))
            instruments = [row[0] for row in cursor.fetchall()]

            # update controller info
            del ctrl['nodeid']
            ctrl['instruments'] = instruments
            
        return res

    @_synchronized
    @_transaction
    def controller_stop(self, id, cursor=None):
        """Stop given controller instance"""
        
        # stop controller process
        q = """UPDATE interface_controller SET kill_ic=1 WHERE id = %s"""
        cursor.execute(q, (id,))

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
        """ read configuration sections from database """

        fullconfig = Config()

        # get configuration from database, include empty section
        for section in [""] + sections:
            
            # read section from database
            config = self.__get_config(section,cursor)
            if verbose : self._log.info ( 'config[%s] = %s', section, config )

            # merge configurations
            fullconfig.merge(config)
        
        return fullconfig

    # ====================================
    # Get fileset with requested status id
    # ====================================

    @_synchronized
    @_transaction
    def get_filesets ( self, status, instruments=None, activeOnly=True, cursor=None ) :
        """Finds and returns a list of filesets with a given status. 
        Filesets will be ordered by their priority. If activeOnly is true then
        filesets from active experiments will be returned, otherwise from all
        experiments.
        """
        
        if type(status) in [types.TupleType, types.ListType]:
            statfmt = ",".join(['%s']*len(status))
            vars = tuple(status)
        else:
            statfmt = "%s"
            vars = (status,)
        
        # find a matching fileset and lock it for update
        if activeOnly:
            q = """SELECT fs.id id, fs.experiment, fs.instrument, run_type, run_number, 
                          stat.name status, tr.id tr_id, tr.jobid, tr.output_dir, fs.priority
                        FROM fileset fs LEFT OUTER JOIN translator_process tr ON tr.id = fs.translator_id, 
                        fileset_status_def stat, active_exp act
                        WHERE stat.name IN (%s) AND fs.fk_fileset_status = stat.id 
                        AND fs.instrument = act.instrument AND fs.experiment = act.experiment""" % statfmt
        else:
            q = """SELECT fs.id id, fs.experiment, fs.instrument, run_type, run_number, 
                          stat.name status, tr.id tr_id, tr.jobid, tr.output_dir, fs.priority
                        FROM fileset fs LEFT OUTER JOIN translator_process tr ON tr.id = fs.translator_id, 
                        fileset_status_def stat
                        WHERE stat.name IN (%s) AND fs.fk_fileset_status = stat.id""" % statfmt
        if instruments :
            fmt = ','.join(['%s']*len(instruments))
            q += ' AND fs.instrument IN (%s)' % fmt
            vars += tuple(instruments)
        q += " ORDER BY fs.priority DESC, fs.created ASC FOR UPDATE"

        res = []
        cursor.execute(q, vars)
        for row in cursor.fetchall():
            
            id, experiment, instrument, run_type, run_number, fstatus, tr_id, jobid, output_dir, priority = row

            # set lock flag 
            try :
                cursor.execute("""UPDATE fileset SET locked = TRUE WHERE fileset.id = %s""", ( id, ) )
            except Exception, ex :
                self._log.warning( "Failed to lock fileset record, retry later, set #%d" % id )
                return None

            # get the list of files in fileset
            cursor.execute("SELECT name FROM files WHERE fk_fileset_id = %s and type='XTC'", (id,) )
            rows = cursor.fetchall()
            xtc_files = [ r[0] for r in rows ]

            translator = None
            if tr_id: translator = Translator(tr_id, jobid, output_dir)
            res.append(Fileset(id, experiment, instrument, run_type, run_number, fstatus, xtc_files, translator, priority))
        
        self._log.debug('get_filesets: found %d filesets with status %s', len(res), status)        
        
        return res

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
        """ change the priority value for fileset"""

        cursor.execute("""UPDATE fileset SET priority = %s WHERE fileset.id = %s""", (priority, fileset_id) )


    # =====================================
    # Insert row for new translator process
    # =====================================

    @_synchronized
    @_transaction
    def new_translator ( self, fs_id, log, jobId, output_dir, cursor=None ) :

        """Add a row for this translator process. Update statistics after run is complete.
        Return the id of the new row for later update"""

        cursor.execute("""INSERT INTO translator_process 
                (id, fk_fileset, kill_tp, started, log, jobid, output_dir)
                VALUES(NULL, %s, False, %s, %s, %s, %s)""",
                ( fs_id, Time.now().toString("%F %T"), log, jobId, output_dir ) )
        cursor.execute("SELECT LAST_INSERT_ID()")
        rows = cursor.fetchall()
        if not rows:
            raise _DatabaseOperatonFailed( "Failed to retrieve LAST_INSERT_ID in new_translator" )
        translator_id = rows[0][0]

        # remember working translator ID for this fileset
        cursor.execute("UPDATE fileset SET translator_id = %s WHERE id = %s", (translator_id, fs_id))
        
        return translator_id


    # ======================================
    # Update row info for translator process
    # ======================================

    @_synchronized
    @_transaction
    def update_translator( self, translator_id, proc_code, ofilesize, cursor=None) :

        """Update the row for this translator process. Update run statistics and process return
        code. We store all of the resource usage even though some are 0 for a given OS.
        Take usage - perf_prev to get usage for the last child process. """

        cursor.execute("""UPDATE translator_process SET 
           stopped = %s, filesize_bytes = %s, tstatus_code = %s
           WHERE id = %s """,
           (Time.now().toString("%F %T"), ofilesize, proc_code, translator_id,) )
            

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
    def new_fileset (self, instr, exper, runnum, runtype, xtcfiles, duplicate=False, status = 'WAIT', priority=0, cursor=None ) :
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
           
        cursor.execute("SELECT id FROM fileset_status_def WHERE name = %s", ('WAIT_FILES',) )
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
    def stop_fileset_id (self, id, cursor=None ) :
        """ Either kill hte job or remove existing fileset based on its ID.
            @param id        fileset id
        """

        # need to know the status
        q = "SELECT fk_fileset_status FROM fileset WHERE id = %s FOR UPDATE"
        cursor.execute ( q, (id,) )
        row = cursor.fetchone()
        if row is None: raise DatasetIdError("Dataset with id {} does not exist".format(id))
        status = row[0]
        
        q = "SELECT name, job_state FROM fileset_status_def WHERE id = %s"
        cursor.execute ( q, (status,) )
        row = cursor.fetchone()
        sname, state = row
        
        if state == 'QUEUE' and sname != 'PENDING':

            # request has not been sent to LSF yet, simply remove it from database
        
            # break foreign key
            q = "UPDATE fileset SET translator_id = NULL WHERE id = %s"
            cursor.execute ( q, (id,) )
            
            # delete all translator processes for those filesets
            q = "DELETE FROM translator_process WHERE fk_fileset = %s"
            cursor.execute ( q, (id,) )
        
            # delete all files for those filesets
            q = "DELETE FROM files WHERE fk_fileset_id = %s"
            cursor.execute ( q, (id,) )
    
            # delete filesets
            q = "DELETE FROM fileset WHERE id = %s"
            cursor.execute ( q, (id,) )

        elif state == 'RUN' or sname == 'PENDING':
            
            # kill the job, just set the flag in database
            q = "UPDATE translator_process SET kill_tp = 1 WHERE fk_fileset = %s"
            cursor.execute ( q, (id,) )

        else:

            # means finished already, won't do anything
            raise _DatabaseOperatonFailed("Dataset processing already finished for dataset {}".format(id))

    @_synchronized
    @_transaction
    def remove_fileset_id (self, id, cursor=None ) :
        """ Delete existing fileset based on its ID.
            @param id        fileset id
        """
        
        # break foreign key
        q = "UPDATE fileset SET translator_id = NULL WHERE id = %s"
        cursor.execute ( q, (id,) )
        
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


    @_synchronized
    @_transaction
    def translated_runs(self, instr, exper, cursor=None):
        """
        Get the set of runs which are in translator already (translated or 
        being translated). Returns set of integer numbers
        """ 
        q = """select run_number from fileset where instrument = %s and experiment = %s"""
        cursor.execute( q, (instr, exper) )
        return set(row[0] for row in cursor.fetchall())
        
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
