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

@author Wilko Kroeger
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

pjoin = os.path.join
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

def _exper_dir(fs):
    return "/reg/d/psdm/%s/%s" % (fs.instrument.lower(), fs.experiment)

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

        self._fs = fs
        self._db = db
        self._config = config
        self._log = logger
        # name will be LSF job name, try to limit it to 10 chars
        self._name = "%s-%03d" % (fs.experiment[:6], fs.run_number)
        self._read_from_ffb = False

        self._outputDir = self._get_config('output-dir', _defOutputDir, True)
        if not os.path.isabs(self._outputDir):
            self._outputDir = pjoin(_exper_dir(self._fs), self._outputDir)
            
        if translator is None:
            self._id = None
            self._outputDirTmp = pjoin(self._outputDir, Time.now().toString("%Y%m%dT%H%M%S"))
            self._job = self._startJob()
        else:
            self._id = translator.id
            self._outputDirTmp = translator.outputDir
            self._job = LSF.Job(translator.jobid)
        
        self._status = None  # None means running, 0 - finished OK; non-zero - error
        

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


    # ======================
    # Process single fileset
    # ======================

    def _startJob(self) :
        """Start new job"""
    
        try:
            self.__make_hdf5_dir(self._outputDirTmp)
        except Exception, exc:
            self.error("[%s] Failed to make temporary directory %s: %s", self._name, self._outputDirTmp, exc )
            #    self._update_fs_status('FAIL_MKDIR')
            return None

        # build command line for running translator
        cmd = self.__build_translate_cmd(self._fs, self._outputDir, self._outputDirTmp)

        # start translator
        logname = os.path.join(self._get_config('log-dir', _defLogDir, True),
                               self._get_config('log-name', _defLogName, True))

        job = self._start(cmd, logname)
        #return # test-only
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


    # ===========================================================
    # Build a list that has the command to execute the translator
    # ===========================================================

    def __build_translate_cmd(self, fs, outputDir, outputDirTmp ) :
        """Build the arg list to pass to the translator from the files in fileset
        and the translate_uri destination for the translator output"""
        
        cmd = []
        # configuration file
        cfg_file = self._get_config('config',  _defCfgFileName, True)
        if not cfg_file.startswith('/'): 
            cfg_file = pjoin(self._get_config('release', _defReleasePath, True), cfg_file)

        cmd.extend(("-c", cfg_file))
        # cmd.extend(("-o", "psana.dump_config_file"))
        
        h5name = pjoin(outputDirTmp, self._get_config('hdf5-file-name', _defHdf5FileName, True))
        cmd.extend(("-o", "Translator.H5Output.output_file=" + h5name))
        
        self.__link_hdf5_canonical(h5name, outputDir)
        
        to_subdir = self._get_config('output-cc-subdir', False)
        cmd.extend(("-o", "Translator.H5Output.split_cc_in_subdir=" + str(to_subdir)))

        live_timeout = self._get_config('live-timeout', 0)
        if live_timeout > 0:
            cmd.append("-o")
            cmd.append("PSXtcInput.XtcInputModule.liveTimeout=%d" % live_timeout)
            
        for xtc in fs.xtc_files:
            if xtc.find('/reg/d/ffb/') >= 0:
                self._read_from_ffb = True
            cmd.append(xtc)
        self.trace("ffb status %s", self._read_from_ffb)

        return cmd

    # ========================
    # Start the translator job
    # ========================

    def _start(self, cmd, logname) :

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

        rel = self._get_config('release')
        if not rel:
            self.error("No Release directory specified")
            return None
            
        if not os.path.isabs(rel):
            # assume release is a name (e.g.: ana-0.13.1)
            #rel = os.path.join(os.environ['SIT_RELDIR'], rel)
            rel = os.path.join("/reg/g/psdm/sw/releases", rel)


        if not os.path.isdir(rel):
            self.error("Release directory %s does not exist", rel)
            return None
        
        self.debug("found release directory: %s", rel)
        cmd = "ic-mon-mpi-runner" + " " + rel + " " + cmd
            
        # LSF parameter
        queue = self._get_queue_name()
        numproc = self._get_config('lsf-numproc', 1)
        resource = extra_opt = None 
        if self._read_from_ffb:
            ptile = self._get_config('lsf-ptile', 0)
            resource="span[ptile=%d]" % ptile if ptile > 0 else None
            extra_opt = "-x" if self._get_config('lsf-exclusive', 0) == 1 else None

        # submit a job
        self.info('submitting new job, command: %s\n  queue=%s, jobName=%s, log=%s, numProc=%s, resource=%s', \
                  cmd, queue, self._name, logname, numproc, resource)
        self._checkDbPriority()
        job = LSF.submit_bsub(cmd, queue=queue, jobName=self._name, log=logname, numProc=numproc,\
                              resource=resource, extraOpt=extra_opt)
        self.trace('new job: %s', job)
        
        return job


    # ===============================
    # make directory for output files
    # ===============================

    def __make_hdf5_dir(self, dirname) :

        # output directory must be empty or non-existent
        if os.path.exists(dirname) :
            if not os.path.isdir(dirname) :
                msg = '[%s] output directory exist but is not a directory: %s' % ( self._name, dirname )
                self.warning(msg)
                raise IOError(msg)
        else :
            # create output directory
            self.trace ( '[%s] create directory for output files: %s', self._name, dirname )
            os.makedirs(dirname)

    def __link_hdf5_canonical(self,h5name, outputDir):
        link_name = pjoin(outputDir, os.path.basename(h5name))
        rel_target = os.path.relpath(h5name, os.path.commonprefix((link_name,h5name)))
        try:
            if os.path.islink(link_name):
                os.remove(link_name)            
            os.symlink(rel_target, link_name)
        except Exception as msg:
            self.warning("Failed to create link %s", link_name)

    # =================================================
    # Store HDF5 files in both dataset and file manager
    # =================================================

    def __store_hdf5 (self, fs, tmpdirname, dirname):
        return 0
                
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
        if self._read_from_ffb:
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

        return self._get_config(queue_param, 'psanaq', True)
    
    
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
