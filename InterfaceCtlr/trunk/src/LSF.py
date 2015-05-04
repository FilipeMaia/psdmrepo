#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module LSF...
#
#------------------------------------------------------------------------

"""Brief one-line description of the module.

Following paragraphs provide detailed description of the module, its
contents and usage. This is a template module (or module template:)
which will be used by programmers to create new Python modules.
This is the "library module" as opposed to executable module. Library
modules provide class definitions or function definitions, but these
scripts cannot be run by themselves.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andy Salnikov
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
import logging
import types
import time
import signal
import subprocess

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from pylsf import lsf

#------------------------
# Exported definitions --
#------------------------

# job status values (bitmasks)
JOB_STAT_PEND      = lsf.JOB_STAT_PEND      # job is pending
JOB_STAT_PSUSP     = lsf.JOB_STAT_PSUSP     # job is held
JOB_STAT_RUN       = lsf.JOB_STAT_RUN       # job is running
JOB_STAT_SSUSP     = lsf.JOB_STAT_SSUSP     # job is suspended by LSF batch system
JOB_STAT_USUSP     = lsf.JOB_STAT_USUSP     # job is suspended by user
JOB_STAT_EXIT      = lsf.JOB_STAT_EXIT      # job exited
JOB_STAT_DONE      = lsf.JOB_STAT_DONE      # job is completed successfully
JOB_STAT_PDONE     = lsf.JOB_STAT_PDONE     # post job process done successfully
JOB_STAT_WAIT      = lsf.JOB_STAT_WAIT      # chunk job waiting its execution turn
JOB_STAT_UNKWN     = lsf.JOB_STAT_UNKWN     # unknown status

# options for bjobs() call
ALL_JOB    = lsf.ALL_JOB
DONE_JOB   = lsf.DONE_JOB
PEND_JOB   = lsf.PEND_JOB
SUSP_JOB   = lsf.SUSP_JOB
CUR_JOB    = lsf.CUR_JOB
LAST_JOB   = lsf.LAST_JOB


class LSBError(Exception):

    def __init__(self):
        Exception.__init__(self, lsf.lsb_sysmsg())

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_parameterinfo = None # result of lsb_parameterinfo()

def _statStr(status):
    if status is None: return "NONE"
    res = []
    if status & JOB_STAT_PEND : res.append("PEND")
    if status & JOB_STAT_PSUSP : res.append("PSUSP")
    if status & JOB_STAT_RUN : res.append("RUN")
    if status & JOB_STAT_SSUSP : res.append("SSUSP")
    if status & JOB_STAT_USUSP : res.append("USUSP")
    if status & JOB_STAT_EXIT : res.append("EXIT")
    if status & JOB_STAT_DONE : res.append("DONE")
    if status & JOB_STAT_PDONE : res.append("PDONE")
    if status & JOB_STAT_WAIT : res.append("WAIT")
    if status & JOB_STAT_UNKWN : res.append("UNKWN")
    return "|".join(res)
        
#---------------------
#  Class definition --
#---------------------
class Job ( object ) :
    """Class representing a job in LSF."""

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, arg) :
        """ Constructor takes single argument which could be 
        an integer specifying job ID or a data structure 
        returned by lsb_readjobinfo. If job ID is specified
        the all other data are available only after call to 
        update() method"""
        
        if type(arg) in (types.IntType, types.LongType):
            self._jobid = arg
            self.__reset()
        else:
            # object is a data from lsb_readjobinfo
            self._jobid = arg.jobId
            self.__set(arg)
            
    #-------------------
    #  Public methods --
    #-------------------

    def jobid(self) :
        return self._jobid

    def name(self):
        return self._name

    def status(self):
        return self._status

    def statusStr(self):
        return _statStr(self._status)

    def exitStatus(self):
        return self._exitStatus

    def priority(self):
        return self._priority

    def __str__(self):
        return "Job(id=%s,status=%s)" % (lsf.lsb_jobid2str(self._jobid), _statStr(self._status))
    
    def update(self):
        """ Retrieves job status information from LSF and updates internal state."""

        data = self.__update()
        if data:
            # update status
            self.__set(data)

    def setPriority(self, priority):
        """Change priority of the existing job"""
        
        data = self.__update()
        if data is None:
            logging.warning('Job.setPriority() failed to get current status for %s', self)
            return

        req = lsf.submit()
        options, options2 = 0, 0
        req.command = str(self._jobid)
        if priority is not None:
            options2 = options2 | lsf.SUB2_JOB_PRIORITY
            req.userPriority = priority
            
        req.options = options
        req.options2 = options2
    
        req.rLimits = data.submit.rLimits
    
        req.beginTime = data.submit.beginTime
        req.termTime = 0
        req.numProcessors = data.submit.numProcessors
        req.maxNumProcessors = data.submit.maxNumProcessors
        
        reply = lsf.submitReply()
    
        job_id = lsf.lsb_modify(req, reply, self._jobid)
        if job_id < 0:
            raise LSBError()

        self._priority = priority

    def kill(self, sig=None):
        """Send signal to a job"""
        
        if sig is None:
            # this follows bkill algorithm
            lsf.lsb_signaljob(self._jobid, signal.SIGTERM)
            lsf.lsb_signaljob(self._jobid, signal.SIGINT)
            # give it 5 seconds to cleanup
            time.sleep(5)
            lsf.lsb_signaljob(self._jobid, signal.SIGKILL)
        else:
            lsf.lsb_signaljob(self._jobid, sig)
        
    def __update(self):
        """ Retrieves job status information from LSF and updates internal state."""

        count = lsf.lsb_openjobinfo(self._jobid, None, "all", None, None, lsf.ALL_JOB)
        if count < 1:
            logging.warning('lsb_openjobinfo() failed for %s (job may be finished long ago or has not started)', self)
            self.__reset()
            data = None
        else:
            if count > 1:
                logging.warning('lsb_openjobinfo() returned more than one match for %s', self)
            jobp = lsf.new_intp()
            lsf.intp_assign(jobp, 0)
            data = lsf.lsb_readjobinfo(jobp)

        lsf.lsb_closejobinfo()
        return data

    def __reset(self):
        self._status = None
        self._exitStatus = None
        self._priority = None
        self._user = None
        self._durationMinutes = None
        self._cpuTime = None
        self._name = None

    def __set(self, data):
        self._status = data.status
        self._exitStatus = data.exitStatus
        self._priority = data.jobPriority
        self._user = data.user
        self._durationMinutes = data.duration
        self._cpuTime = data.cpuTime
        self._name = data.jName

def submit_bsub(command, queue="psanaq", jobName=None, log=None, numProc=1, \
                resource=None, extraOpt=None):
    bsub_opt = ["-q %s" % queue]
    bsub_opt.append("-n %d" % numProc)
    if jobName:
        bsub_opt.append("-J %s" % jobName)
    if log:
        bsub_opt.append("-o %s" % log)
    if resource:
        bsub_opt.append("-R \"%s\"" % resource)
    if extraOpt:
        bsub_opt.append(extraOpt)
        
    bsub_opt.append("-a mympi")
    bsub_cmd = "bsub " + " ".join(bsub_opt)
                        
    cmd = bsub_cmd.split()
    cmd.extend(command.split())
    
    res = subprocess.check_output(cmd)
    for line in res.split('\n'):
        if line.startswith('Job <'):
            jobid = int(line[5:].split('>')[0])
    
    time.sleep(2)
    maxLoops = 10
    for i in xrange(maxLoops):
        job = Job(jobid)
        job.update()
        stat = job.status()
        if stat == None:
            print "Could not get job id, wait 4s count", i
            time.sleep(4)
        else:
            return Job(jobid)

    # Could not get job id
    raise LSBError()


def submit(command, queue = None, jobName = None, 
           log = None, priority = None, 
           numProc = 1, resource = None, beginTime = 0):
    """Submit a job to LSF.
    
    @param[in] command    Command to execute
    @param[in] queue      Queue name
    @param[in] jobName    Name of the job
    @param[in] log        Log file path
    @param[in] priority   User priority, number between 1 and LSF.maxUserPriority()
    @param[in] resource   Resource requirements string
    """
    
    req = lsf.submit()
    options, options2 = 0, 0
    req.command = command
    if queue:
        options = options | lsf.SUB_QUEUE
        req.queue = queue
    if jobName:
        options = options | lsf.SUB_JOB_NAME
        req.jobName = jobName
    if resource:
        options = options | lsf.SUB_RES_REQ
        req.resReq = resource
    if log:
        options = options | lsf.SUB_OUT_FILE
        options2 = options2 | lsf.SUB2_OVERWRITE_OUT_FILE
        req.outFile = log
    if priority is not None:
        options2 = options2 | lsf.SUB2_JOB_PRIORITY
        req.userPriority = priority
        
    req.options = options
    req.options2 = options2

    req.rLimits = [lsf.DEFAULT_RLIMIT] * lsf.LSF_RLIM_NLIMITS

    req.beginTime = beginTime
    req.termTime = 0
    req.numProcessors = numProc
    req.maxNumProcessors = numProc

    reply = lsf.submitReply()

    job_id = lsf.lsb_submit(req, reply)
    if job_id < 0: 
        raise LSBError()
    
    return Job(job_id)


def bjobs(queue = None, user = "all", options = CUR_JOB):

    count = lsf.lsb_openjobinfo(0, None, user, queue, None, options)
    if count < 1: return []
    
    result = []
    
    jobp = lsf.new_intp()
    lsf.intp_assign(jobp, count)
    while lsf.intp_value(jobp):
        data = lsf.lsb_readjobinfo(jobp)
        result.append(Job(data))

    lsf.lsb_closejobinfo()
    
    return result

def maxUserPriority():
    """Return max user priority value"""
    
    global _parameterinfo
    if not _parameterinfo: 
        _parameterinfo = lsf.lsb_parameterinfo([], None, 0)
        if not _parameterinfo:
            raise LSBError()
    return _parameterinfo.maxUserPriority

#
# One-time initialization needed for LSB
#    
lsf.lsb_init("LSF.py")

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
