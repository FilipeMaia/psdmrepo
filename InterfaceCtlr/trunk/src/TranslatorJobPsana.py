#!@PYTHON@
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: TranslatorJob.py 6414 2013-06-13 16:31:06Z salnikov@SLAC.STANFORD.EDU $
#
# Description:
#  Class TranslatorJob.
#
#------------------------------------------------------------------------

"""Class which represents a single translator job.

This software was developed for the LUSI project.  If you use all or
part of it, please give an appropriate acknowledgement.

@version $Id: TranslatorJob.py 6414 2013-06-13 16:31:06Z salnikov@SLAC.STANFORD.EDU $

@author Andy Salnikov
"""


#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 6414 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import time
import shutil

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from DbTools.DbConnection import DbConnection
from InterfaceCtlr.FileMgrIrods import FileMgrIrods
from InterfaceCtlr.InterfaceDb import InterfaceDb
from InterfaceCtlr import LSF
from LusiTime.Time import Time
from RegDB.RegDb import RegDb

#---------------------
# Local definitions --
#---------------------

_defOutputDir = "/reg/d/psdm/%(instrument_lower)s/%(experiment)s/hdf5"
_defHdf5FileName = "%(experiment)s-r%(run_number)04d-c{seq2}.h5"
_defOutputDirTmp = "/reg/d/psdm/%(instrument_lower)s/%(experiment)s/hdf5/%(run_number)04d-%(current_time)s"
_defLogDir = "/reg/g/psdm/psdatmgr/ic/log/psana/translate-%(experiment)s"
_defLogName = "translate-%(experiment)s-r%(run_number)04d-%(current_time)s.log"
_defReleasePath = "/reg/g/psdm/sw/releases/ana-current"
_defCfgFileName = "data/Translator/automatic_translation.cfg"

# shutil.Error has rather weird content that needs special formatting
def _shutilErrFmt(ex):
    
    def _fmt(item):
        if isinstance(item, tuple):
            return '(' + ', '.join([_fmt(x) for x in item]) + ')'
        elif isinstance(item, list):
            return '[' + ', '.join([_fmt(x) for x in item]) + ']'
        else:
            return str(item)
    
    return _fmt(ex.args)


#-------------------
# Local functions --
#-------------------

#--------------------------
# Specialized exceptions --
#--------------------------

#--------------------
# Class definition --
#--------------------

class TranslatorJob(object) :
    """
    Class representing a translator job running in a batch system.
    """

    # ===========
    # constructor
    # ===========

    def __init__ ( self, fs, db, config, logger, translator = None) :
        """
        @param[in]  fs        Fileset dictionary
        @param[in]  db        Databse instance
        @param[in]  config    Configurtaion object
        @param[in]  logger    Logging instance
        @param[in]  jobid     Batch job ID
        
        If jobid is not None then there is an existing job running already,
        and this instance will correspond to that existing job. Otherwise
        it will send new translator job to LSF.
        """

        # instance variables
        self._fs = fs
        self._db = db
        self._config = config
        self._log = logger
        # name will be LSF job name, try to limit it to 10 chars
        self._name = "%s-%03d" % (fs.experiment[:6], fs.run_number)

        if translator is None:
            self._id = None
            self._outputDir = self._get_config('output-dir', _defOutputDir, True)
            self._outputDirTmp = self._get_config('output-dir-tmp', _defOutputDirTmp, True)
            self._job = self._startJob()
        else:
            self._id = translator.id
            self._outputDir = self._get_config('output-dir', _defOutputDir, True)
            self._outputDirTmp = translator.outputDir
            self._job = LSF.Job(translator.jobid)

        self._status = None  # None means running, 0 - finished OK; non-zero - error


    # ======================
    # Process single fileset
    # ======================

    def _startJob( self ) :
        """Start new job"""

        # output directory must be empty or non-existent
        try:
            self.__make_hdf5_dir(self._outputDirTmp)
        except Exception, exc:
            self.error("[%s] Failed to make temporary directory %s: %s", self._name, self._outputDirTmp, exc )
            self._update_fs_status('FAIL_MKDIR')
            return None

        # build command line for running translator
        cmd = self.__build_translate_cmd(self._fs, self._outputDir, self._outputDirTmp)

        # start translator
        logdir = self._get_config('log-dir', _defLogDir, True)
        logname = self._get_config('log-name', _defLogName, True)
        logname = os.path.join(logdir, logname)
        job = self._start ( cmd, logname )
        if not job:
            self._update_fs_status('FAIL')
            return None            
        
        self._id = self._db.new_translator(self._fs.id, logname, job.jobid(), self._outputDirTmp)
        
        # tell everybody we are taking care of this fileset
        self._update_fs_status('PENDING')
        
        self.info ("[%s] Started translator #%d (%s) with cmd %s", self._name, self._id, job, ' '.join(cmd) )
        self.info ("[%s] output directory %s", self._name, self._outputDirTmp )
        self.info ("[%s] Log file is in %s", self._name, logname )
        
        return job


    def check(self):
        """Use this method to periodically check the status. It performs 
        all necessary completion tasks once job is finished. Returns
        None if job is still running; zero if job has finished and 
        all post-processing completed OK; non-zero number on errors. 
        """

        # do not re-check it again
        if self._status is not None : return self._status

        # update job status from LSF
        self._job.update()
        
        lsf_status = self._job.status()
        exitCode = self._job.exitStatus()
        if lsf_status is None:
            # something bad like LSF is down or job has disappeared
            self.warning("[%s] Failed to get job status for job %s", self._name, self._job)
            
            # say that we are still running though, next iteration could fix it
            return self._status

        # update name from LSF
        self._name = self._job.name()

        self.debug("[%s] %s status=%#x exitCode=%s", self._name, self._job, lsf_status, exitCode)
        if lsf_status & LSF.JOB_STAT_PEND:
            
            # still pending
            if self._fs.status != 'PENDING': self._update_fs_status('PENDING')
            
            # may decide to change priority still
            self._checkDbPriority()
            if self._fs.priority != self._job.priority():
                self.trace("[%s] Job %s priority change %s to %s", self._name, self._job, self._job.priority(), self._fs.priority)
                self._job.setPriority(self._fs.priority)
            
        elif lsf_status & LSF.JOB_STAT_RUN:
            
            # still running
            self._update_fs_status('RUN')
        
        elif lsf_status & (LSF.JOB_STAT_PSUSP|LSF.JOB_STAT_SSUSP|LSF.JOB_STAT_USUSP):
            
            # still running
            self._update_fs_status('SUSPENDED')
        
        elif lsf_status & LSF.JOB_STAT_UNKWN:
            
            # do not know how to handle this thing
            self.warning("[%s] Job %s has unknown status, wait until next iteration", self._name, self._job)
        
        elif lsf_status & LSF.JOB_STAT_DONE:

            # job finished, one could also check post-processing status, but
            # in LSF 6.2 that we have now it's not easy to do
            self._status = 0
        
        elif lsf_status & LSF.JOB_STAT_EXIT:

            # Exit generally means failure, check the code too
            if exitCode == 0:
                self.warning("[%s] job %s has <EXIT> status, but zero exit code, will return -1", self._name, self._job)
                exitCode = -1
            self._status = exitCode

        # if still not finished then return
        if self._status is None: 
            
            # but check if anybody wants it killed
            if self._db.test_exit_translator(self._id):
                self.info("[%s] killing job %s", self._name, self._job)
                self._job.kill()
            
            return None

        self.info ("[%s] translator #%d finished (%s) retcode=%s", self._name, self._id, self._job, exitCode)
        
        # get the size of resulting files
        output_size = 0
        if not self._get_config('self-move', 0):
            output_size = self.__dir_size(self._outputDirTmp)
            self.info ("[%s] translator #%d produced %d bytes of data", self._name, self._id, output_size)

        # store statistics
        self._db.update_translator(self._id, exitCode, output_size)

        if exitCode != 0 :
            
            self.warning ("[%s] translator #%d failed", self._name, self._id)
            self._update_fs_status('FAIL')
            
        else:
            
            # copy resulting files to final destination
            returncode = self.__store_hdf5(self._fs, self._outputDirTmp, self._outputDir)
            self._db.update_irods_status (self._id, returncode)
            if returncode != 0:
                self._update_fs_status('FAIL_COPY')
            else:
                self.info ("[%s] moved data to final directory", self._name)
                self._update_fs_status('DONE')

    # ===========================================================
    # Build a list that has the command to execute the translator
    # ===========================================================

    def __build_translate_cmd ( self, fs, outputDir, outputDirTmp ) :

        #  -m Translator.H5Output -o Translator.H5Output.output_file=myoutput.h5 \
        #    -o Translator.H5Output.overwrite=True  exp=amoc23:run=023


        """Build the arg list to pass to the translator from the files in fileset
        and the translate_uri destination for the translator output"""

        cmd_list = []
        cmd_list.append("psana")

        # configuration file

        cfg_file = self._get_config('config',  _defCfgFileName, True)
        if cfg_file.startswith('/'): 
            cfgFilePath = cfg_file
        else:
            cfgFilePath = os.path.join(self._get_config('release', _defReleasePath, True), cfg_file)

        cmd_list.append("-c")
        cmd_list.append(cfgFilePath)
        cmd_list.append("-o")
        cmd_list.append("psana.dump_config_file")

        ## modules  
        #if self._get_config('calib-enable', 0) != 0:
        #    modules = "cspad_mod.CsPadCalib,Translator.H5Output"
        #else:
        #    modules = "Translator.H5Output"
        #cmd_list.append("-m")
        #cmd_list.append(modules)
        
        # translator module options 
        cmd_list.extend("-o Translator.H5Output.overwrite=True".split())
        h5name = os.path.join(outputDirTmp, 
                              self._get_config('hdf5-file-name', _defHdf5FileName, True))
        cmd_list.append("-o")
        cmd_list.append("Translator.H5Output.output_file=" + h5name)


        live_timeout = self._get_config('live-timeout', 0)
        if live_timeout > 0:
            cmd_list.append("-o")
            cmd_list.append("PSXtcInput.XtcInputModule.liveTimeout=%d" % live_timeout)

        calibdir = self._get_config('calib-data-dir', None, True)
        if calibdir:
            cmd_list.append("--calib-dir")
            cmd_list.append(calibdir)

        # any extra options 
        #for opt in self._get_config('list:psana-extra-options',[]) :
        #    cmd_list.extend( opt.split() )
            
        # there might be no xtc files as a dataset is used
        for xtc in fs.xtc_files:
            cmd_list.append(xtc)
        #cmd_list.append("exp=%s:run=%s" % (fs.experiment, str(fs.run_number)))

        return cmd_list


    # ===============================
    # make directory for output files
    # ===============================

    def __make_hdf5_dir(self, dirname) :

        # output directory must be empty or non-existent
        if os.path.exists(dirname) :
            if not os.path.isdir(dirname) :
                msg = '[%s] output directory exist but is not a directory: %s' % ( self._name, dirname )
                self.warning ( msg )
                raise IOError( msg )
            elif os.listdir(dirname) :
                msg = '[%s] output directory exist but is not empty: %s' % ( self._name, dirname )
                self.warning ( msg )
                raise IOError( msg )
        else :
            # create output directory
            self.trace ( '[%s] create directory for output files: %s', self._name, dirname )
            os.makedirs(dirname)

    # =================================================
    # Store HDF5 files in both dataset and file manager
    # =================================================

    def __store_hdf5 (self, fs, tmpdirname, dirname):

        # generator for all file paths under given directory, 
        # returns list of tuples (subdir,filename)
        def _all_files( root, subdir = "" ) :
            for e in os.listdir( os.path.join(root,subdir) ) :
                path = os.path.join(root,subdir,e)
                if os.path.isdir(path) :
                    for x in _all_files ( root, os.path.join(subdir,e) ) :
                        yield x
                else :
                    yield ( subdir, e )

        def _basename(path):
            path = os.path.split(path)
            if not path[1]: path = os.path.split(path[0])
            return path[1]
        
        if not os.path.exists(tmpdirname):
            self.error("store_hdf5: temporary directory %s does not exist", tmpdirname)
            return 2

        # in self-move mode files should have been moved already, so only try to remove temporary directory 
        if self._get_config('self-move', 0):
            try:
                self.debug("removing temp dir %s", tmpdirname)
                os.rmdir( tmpdirname )
                return 0
            except Exception, e :
                # if it fails means that translator failed to move some files, report this as an error
                self.error("store_hdf5: failed to remove directory %s: %s", tmpdirname, str(e) )
                return 2
        
        # first move tmpdir as a whole to the final directory
        tmpdirfinal = os.path.join(dirname, _basename(tmpdirname))
        if not os.path.exists(tmpdirfinal) or not os.path.samefile(tmpdirfinal, tmpdirname):
            try:
                self.trace("moving directory %s -> %s", tmpdirname, tmpdirfinal)
                shutil.move(tmpdirname, tmpdirfinal)
            except Exception, e:
                self.error("store_hdf5: failed to move output files to directory %s", tmpdirfinal)
                self.error("store_hdf5: exception raised: %s", _shutilErrFmt(e) )
                return 2

        try:
            # build a list of files to store
            files = [ x for x in _all_files( tmpdirfinal ) ]
        except Exception, e:
            self.error("store_hdf5: failed to find output files in directory %s", tmpdirfinal)
            self.error("store_hdf5: exception raised: %s", str(e) )
            return 2

        backup_suffix = self._get_config("hdf-backup-suffix", None, True)
        if backup_suffix:
            # if destination file already there rename it
            for f in files :
                dst = os.path.join( dirname, f[0], f[1] )
                if os.path.exists(dst) :
                    dstbck = dst + backup_suffix
                    self.trace("store_hdf5: backup existing file %s -> %s", dst, dstbck)
                    try:
                        shutil.move(dst, dstbck)
                    except Exception, e :
                        self.error("store_hdf5: failed to backup file %s", dst)
                        self.error("store_hdf5: exception raised: %s", _shutilErrFmt(e) )
                        return 2
        else:
            # check that final destination does not have these files
            for f in files :
                dst = os.path.join( dirname, f[0], f[1] )
                if os.path.exists(dst) :
                    self.error("store_hdf5: destination files already exists: %s",dst)
                    return 2

            
        # move all files to final destination
        for f in files :
            src = os.path.join( tmpdirfinal, f[0], f[1] )
            dst = os.path.join( dirname, f[0], f[1] )
            try:
                self.trace("moving file %s -> %s", src,dst)
                shutil.move(src,dst)
            except Exception, e :
                self.error("store_hdf5: failed to move file: %s -> %s", src, dst)
                self.error("store_hdf5: exception raised: %s", _shutilErrFmt(e) )
                return 2
        
        # remove temporary directory
        try :
            self.debug("removing temp dir %s", tmpdirfinal)
            os.rmdir( tmpdirfinal )
        except Exception, e :
            # non-fatal, means it may have some subdirectories
            self.error("store_hdf5: failed to remove directory %s: %s", tmpdirfinal, str(e) )

        # add all files to fileset
        self._db.add_files( fs.id, 'HDF5', [os.path.join(dirname,f[0],f[1]) for f in files ] )


        # register them with iRODS, do not generate errors even if registration fails
        irods_cmd = self._get_config('filemanager-irods-command', None, True)
        dst_coll = self._get_config('filemanager-hdf5-dir', None, True)
        if irods_cmd and dst_coll:
            
            file_mgr = FileMgrIrods(irods_cmd, self._config, self)
            
            for f in files:
                
                src = os.path.join( dirname, f[0], f[1] )
                dst_dir = os.path.join(dst_coll, f[0])
                
                try:
                    # check that file is not there yet
                    if f[1] in file_mgr.listdir(dst_dir):
                        self.trace("file %s is already registered", f[1] )
                    else:
                        dst = os.path.join( dst_dir, f[1] )
                        file_mgr.storeFile(src, dst)
                        self.trace("registered file in irods as %s", dst )
                except Exception, e:
                    self.warning("store_hdf5: failed to register file in irods %s: %s", os.path.join( dst_dir, f[1] ), str(e) )

        return 0
                
    # ========================
    # Start the translator job
    # ========================
    def _start ( self, cmd, logname ) :

        def shellquote(s):
            return "'" + s.replace("'", "'\\''") + "'"

        # create log directory if needed
        logdir = os.path.dirname(logname)
        if logdir and not os.path.isdir(logdir):
            try:
                os.makedirs(logdir)
            except OSError, e:
                # complain but proceed, we may have race condition here
                self.warning("Failed to create log directory: %s", str(e))


        # build full command name
        cmd = ' '.join([shellquote(c) for c in cmd])

        # add LD_PRELOAD before command
        ld_preload = self._get_config('ld-preload')
        if ld_preload: 
            cmd = 'LD_PRELOAD=' + shellquote(ld_preload) + ' ' + cmd

        # add PAZLIB_MAX_THREADS before command
        max_threads = self._get_config('pazlib-max-threads')
        if max_threads: 
            cmd = 'PAZLIB_MAX_THREADS=' + shellquote(str(max_threads)) + ' ' + cmd

        use_calib = self._get_config('calib-enable')
        if use_calib:
            cmd = 'MSGLOGCONFIG="CalibDataProxy=trace"' + ' ' + cmd

        # go to specific release before running a job, release can be a 
        # release name or directory 
        rel = self._get_config('release')
        if rel:
            if os.path.isabs(rel):
                # absolute path must be directory name
                if not os.path.isdir(rel):
                    self.error("Release directory %s does not exist", rel)
                    return None
                self.debug("found release directory: %s", rel)
                cmd = "cd " + rel + "; . $SIT_ROOT/bin/sit_setup.sh; " + cmd
            else:
                # treat it as release name
                try:
                    reldir = os.path.join(os.environ['SIT_RELDIR'], rel)
                    if not os.path.isdir(reldir): reldir = None
                except KeyError:
                    reldir = None
                if reldir and os.path.isdir(reldir):
                    self.debug("found standard release: %s", rel)
                    cmd = ". $SIT_ROOT/bin/sit_setup.sh " + rel + "; " + cmd
                else:
                    self.error("Release directory %s does not exist", rel)
                    return None
                    
        # get queue name
        queue = self._get_queue_name()

        # get LSF resources
        resource = self._get_config('lsf-resource', None, True)

        # get number of CPUS
        numproc = self._get_config('lsf-numproc', 1)

        # submit a job
        self.info('submitting new job, command: %s\n  queue=%s, jobName=%s, log=%s, numProc=%s, resource=%s', \
                  cmd, queue, self._name, logname, numproc, resource)
        self._checkDbPriority()
        job = LSF.submit(cmd, queue=queue, jobName=self._name, log=logname, numProc=numproc, 
                         priority=self._fs.priority, resource=resource)
        self.trace('new job: %s', job)
        
        return job

    # ============================================
    # Calculate the size of all files in directory
    # ============================================

    def __dir_size( self, dirname ) :

        def safe_stat(f):
            # version of stat which does not throw
            try:
                return os.stat(f).st_size
            except:
                return 0

        # generator for all file paths under given directory
        def _all_files( dirname ) :
            for root, dirs, files in os.walk( dirname ) :
                for f in files :
                    yield os.path.join( root, f )

        return sum( [ safe_stat(f) for f in _all_files(dirname) ] )


    def _get_config(self, option, default = None, interpolate = False):
        """Read configuration parameter value"""

        subs = None        
        if interpolate : 
            subs = {"experiment": self._fs.experiment, 
                    "experimentId": self._fs.experimentId, 
                    "instrument": self._fs.instrument,
                    "instrument_lower": self._fs.instrument.lower(),
                    "run_type": self._fs.run_type, 
                    "run_number": self._fs.run_number,
                    "current_time": Time.now().toString("%Y%m%dT%H%M%S")}
        val = self._config.get(option, self._fs.instrument, self._fs.experiment, default=default, subs=subs)
        #self.debug("_get_config: option=%s instrument=%s experiment=%s default=%s -> %s" % (option, self._fs.instrument, self._fs.experiment, default, val))
            
        return val
            
            
    def _get_queue_name(self):        
        """Determine queue name for the job. If experiment is current according to
        regdb use high-priority queue."""
        
        queue_param = 'lsf-queue'
        conn_str = self._get_config("regdb-conn")
        if conn_str:
            
            try:
                regdb = RegDb(DbConnection(conn_string=conn_str))
                exp = regdb.last_experiment_switch(self._fs.instrument)
                if exp and exp[0][0] == self._fs.experiment:
                    self.trace("use high-priority queue for active experiment %s", self._fs.experiment)
                    queue_param = 'lsf-queue-active'
            except Exception, ex:
                self.warning("exception while accessing regdb: %s", str(ex))
            
        else:
            self.warning("cannot locate regdb connection string in configuration")

        return self._get_config(queue_param, 'lclsq', True)
    
    
    def _update_fs_status(self, status):
        if self._fs.status != status: self._db.change_fileset_status (self._fs.id, status)

    def _checkDbPriority(self):
        """Check the value of dataset priority, update it if it's out of range"""

        try:
            maxPriority = LSF.maxUserPriority()
        except LSF.LSBError, ex:
            # LSF may not be available temporarily, no reason to stop here now
            self.warning("failed to get max priority number, LSF may be unavailable: %s", ex)
            return
            
        if self._fs.priority is None or self._fs.priority < 1 :
            
            self.debug("Change database priority from %s to %s", self._fs.priority,  maxPriority / 2)
            # 0 or negative means use default priority, LSF defines default as MAX/2
            self._fs.priority = maxPriority / 2
            # update priority in database
            self._db.change_fileset_priority(self._fs.id, self._fs.priority)
            
        elif self._fs.priority > maxPriority:
            
            self.debug("Change database priority from %s to %s", self._fs.priority,  maxPriority)
            # limit the priority value
            self._fs.priority = maxPriority
            # update priority in database
            self._db.change_fileset_priority(self._fs.id, self._fs.priority)
    
    #                                                                                                                                                 
    #  Logging methods                                                                                                                                
    #                                                                                                                                                 
    def debug ( self, *args, **kwargs ) : return self._log.debug ( *args, **kwargs )
    def trace ( self, *args, **kwargs ) : return self._log.trace ( *args, **kwargs )
    def info ( self, *args, **kwargs ) : return self._log.info ( *args, **kwargs )
    def warning ( self, *args, **kwargs ) : return self._log.warning ( *args, **kwargs )
    def error ( self, *args, **kwargs ) : return self._log.error ( *args, **kwargs )
    def critical ( self, *args, **kwargs ) : return self._log.critical ( *args, **kwargs )

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
