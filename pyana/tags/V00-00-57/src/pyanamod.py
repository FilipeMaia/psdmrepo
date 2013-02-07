#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module pyana...
#
#------------------------------------------------------------------------

"""Module that encapsulates analysis job.

This software was developed for the SIT project.  If you use all or 
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
import multiprocessing as mp
from resource import *
import gc
import traceback

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import xtc
import pyana as pyana_
from pyana import event
from pyana.input import dgramGen, threadedDgramGen
from pyana.userana import mod_import, evt_dispatch
from pyana.mp_proto import *
from pyana.config import config
from pyana.merger import merger
from pyana.expname import *

#----------------------------------
# Local non-exported definitions --
#----------------------------------

def _vmsize():
    fname = '/proc/%d/stat' % os.getpid() 
    return int(file(fname).read().split()[22])

def _gc(gc_debug, gc_threshod, vmsize):
    # explicitly run garbage collector if memory grows too much
    if gc_debug: logging.debug("VM size: %.2fMB", _vmsize()/1048576.)
    if gc_threshod > 0:
        newvmsize = _vmsize()
        if newvmsize - vmsize > gc_threshod*1048576:
            logging.debug("running garbage collector, virtual size: %.2fMB", newvmsize/1048576.)
            gc.collect()
            vmsize = _vmsize()
            logging.debug("virtual size after GC: %.2fMB", vmsize/1048576.)
    return vmsize

def _rusage(msg, ru1, ru2):
    
    print "%s: %.1f user %.1f sys" % ( msg, ru2.ru_utime-ru1.ru_utime, ru2.ru_stime-ru1.ru_stime )

def _epicsOnly(xtcObj):
    """ Returns true if container has EPICS data only """
    for x in xtcObj:
        if x.contains.id() == xtc.TypeId.Type.Id_Xtc:
            if not _epicsOnly(x):
                return False
        elif x.contains.id() != xtc.TypeId.Type.Id_Epics: 
            return False
    return True

def _proc(jobname, id, pipes, userObjects, jobConfig, expNameProvider):
    """ method which is running in child process when we do multi-processing """

    for i in range(len(pipes)):
        pipes[i][0].close()
        if i != id : pipes[i][1].close()
    pipe = pipes[id][1]

    # gc options
    gc_threshod = jobConfig.getJobConfig("gc-threshod", 10)
    gc_debug = jobConfig.getJobConfig("gc-debug", False)

    dg_ref = jobConfig.getJobConfig('dg-ref', False)

    ru1 = getrusage(RUSAGE_SELF)
    
    # create own environment, make unique job name
    calibDir = jobConfig.getJobConfig("calib-dir", "/reg/d/psdm/$instrument/$experiment/calib")
    env = event.Env(jobname, subproc=id, calibDir=calibDir, expNameProvider=expNameProvider)

    logging.info("proc-%s: starting job %s", id, jobname)

    dispatch = evt_dispatch(userObjects)
    proto = mp_proto(pipe, dg_ref, 'prot-%d' % id)
    nevent = 0
    vmsize = _vmsize()

    # event loop
    for req in proto.getRequest():

        # request is a tuple with first element being opcode
        if req[0] == OP_EVENT :

            epics_data, dg, run, expNum = req[1:]

            evt = event.Event(dg, run=run, expNum=expNum, env=env)
            
            # update configuration objects
            env.updateConfig(evt)
            
            # on configure build list of EPICS objects
            if evt.seq().service() == xtc.TransitionId.Configure: env.updateEpics(evt)
    
            # update epics data
            env.m_epics.m_name2epics = epics_data
    
            # process all data
            dispatch.dispatch(evt, env)
            if evt.status() != pyana_.Normal:
                logging.warning("event loop termination is not supported in multi-process mode")

            if evt.seq().service() == xtc.TransitionId.L1Accept : nevent += 1

        elif req[0] == OP_FINISH :
            
            logging.info("proc-%s: received FIN", id)
            evt = event.Event(None, run=run, expNum=expNum, env=env)
            dispatch.finish(evt, env)
            
        elif req[0] == OP_RESULT :
            
            logging.info("proc-%s: result requested, tag=%s", id, req[1])
            proto.sendData( env.result() )

        elif req[0] == OP_END :
            
            logging.info("proc-%s: received END", id)
            # stop here
            break

        # explicitly run garbage collector if memory grows too much
        vmsize = _gc(gc_debug, gc_threshod, vmsize)
            
    # done with the environment
    env.finish()

    logging.info("proc-%s: processed %d events", id, nevent)

    proto.close()

    ru2 = getrusage(RUSAGE_SELF)
    _rusage( "proc-%s time" % id, ru1, ru2 )

#
# Main() for pyana, takes the standard command-line arguments list
#
def _pyana ( argv ) :

    ru1 = getrusage(RUSAGE_SELF), getrusage(RUSAGE_CHILDREN)

    # get all options from command line and config file
    jobConfig = config( argv[1:] )

    # gc options
    gc_threshod = jobConfig.getJobConfig("gc-threshod", 10)
    gc_debug = jobConfig.getJobConfig("gc-debug", False)

    # get file names
    names = jobConfig.files()

    # get job name
    jobname = jobConfig.getJobConfig("job-name")
    if not jobname :
        # if it was not specified then build it from the input file names
        file_list = jobConfig.getJobConfig("file-list")
        if file_list :
            jobname = os.path.basename(file_list)
            jobname = os.path.splitext(jobname)[0]
        else :
            jobname = os.path.basename(names[0])
            jobname = os.path.splitext(jobname)[0]

    # get all names of user modules
    modules = jobConfig.getJobConfig('modules')
    if isinstance(modules, (str,unicode)): modules = modules.split()
    if not modules :
        logging.error("no user modules defined")
        return 2
    
    # import all user modules, mod_import does not throw but returns None instead
    modConfig = map(jobConfig.getModuleConfig, modules)
    userObjects = map(mod_import, modules, modConfig)
    if None in userObjects :
        # not all modules imported 
        return 2

    # instantiate experiment name provider
    if jobConfig.getJobConfig("experiment"):
        expNameProvider = ExpNameFromConfig(jobConfig.getJobConfig("experiment"))
    else:
        expNameProvider = ExpNameFromXtc(names)
    calibDir = jobConfig.getJobConfig("calib-dir", "/reg/d/psdm/$instrument/$experiment/calib")

    # start worker process if requested
    num_cpu = jobConfig.getJobConfig('num-cpu', 1)
    logging.info("num-cpu: %s", num_cpu)
    nsub = max(num_cpu-1,0)
    dg_ref = jobConfig.getJobConfig('dg-ref', False)
    logging.info("dg-ref: %s", dg_ref)

    conns = [ mp.Pipe() for i in range(nsub)]
    procs = [mp.Process(target=_proc, args=(jobname, i, conns, userObjects, jobConfig, expNameProvider)) for i in range(nsub)]
    for p in procs : p.start()
    for conn in conns : conn[1].close()
    pipes = [mp_proto(conn[0], dg_ref, 'prot-main') for conn in conns]

    skip_epics = jobConfig.getJobConfig('skip-epics', True)
    logging.info("skip-epics: %s", skip_epics)

    status = 0
    try :

        # create environment
        env = event.Env(jobname, calibDir=calibDir, expNameProvider=expNameProvider)
    
        dispatch = evt_dispatch(userObjects)
        num_events = jobConfig.getJobConfig('num-events')
        skip_events = jobConfig.getJobConfig('skip-events') or 0
        nevent = 0
        ndamage = 0
        damagemask = 0
        next = 0
        run = None
        expNum = None
        vmsize = _vmsize()

        # read datagrams one by one
        for dg, fileName, fpos in dgramGen( names ) :

            # explicitly run garbage collector if memory grows too much
            vmsize = _gc(gc_debug, gc_threshod, vmsize)
    
            if num_events is not None and nevent >= num_events+skip_events :
                logging.info("event limit reached (%d), terminating", nevent)
                break
    
            damage = dg.xtc.damage.value()
            svc = dg.seq.service()
        
            expNum = fileName.expNum()
            run = fileName.run()
            evt = event.Event(dg, run=run, expNum=expNum, env=env)
            
            # update environment
            env.update ( evt )

            if svc == xtc.TransitionId.L1Accept :

                if skip_epics and _epicsOnly(dg.xtc):
                    # datagram is likely filtered, has only epics data and users do not need to
                    # see it. Do not count it as an event too, just save EPICS data and move on.
                    continue
                
                evtnum = nevent
                nevent += 1
                
                if evtnum < skip_events :
                    # skip this event
                    continue
                else:
                    # calculate damage mask
                    if damage :
                        ndamage += 1
                        damagemask |= damage


            if procs:
                
                # pass all the data to children
                if svc == xtc.TransitionId.L1Accept :
                    # event data goes to next child
                    p = pipes[next]
                    next = (next+1) % len(pipes)
                    p.sendEventData(dg, fileName, fpos, run, expNum, env)
                else :
                    # each transitions goes to each child
                    for p in pipes : p.sendEventData(dg, fileName, fpos, run, expNum, env)

            else :
                
                # process all data
                dispatch.dispatch(evt, env)
                if evt.status() == pyana_.Terminate:
                    # stop right here, return some strange error code
                    return 12
                if evt.status() == pyana_.Stop:
                    # stop event loop
                    break
                    
                
        # finish
        evt = event.Event(None, run=run, expNum=expNum, env=env)
        dispatch.finish(evt, env)
    
    
        # finalization for multi-processors
        if pipes :
            
            # send FIN to all processes
            for p in pipes : 
                p.sendCode(OP_FINISH)
                
            # get results from all processes
            merge = merger() 
            for p in pipes : 
                merge.merge(p.getResult())
            merge.finish(env)

            # send END to all processes
            for p in pipes :
                p.sendCode(OP_END)
            
    except IOError, exc:
        
        # might mean communication with sub-processes error, stop here
        logging.error("exception caught: %s", exc)
        traceback.print_exc()
        status = exc.errno

    # wait for all processes
    for p in pipes : p.close()
    for p in procs : p.join()

    # done with the environment
    env.finish()

    logging.info("Processed %d events, %d damaged, with damage mask %#x", nevent-skip_events, ndamage, damagemask)

    ru2 = getrusage(RUSAGE_SELF), getrusage(RUSAGE_CHILDREN)
    _rusage( "self  time", ru1[0], ru2[0] )
    _rusage( "child time", ru1[1], ru2[1] )

    return status

#------------------------
# Exported definitions --
#------------------------

def pyana ( **kw ) :
    """ Following keywords are supported:
    
    argv - regular sys.argv, if present all other keywords are ignored;
    config [str] - name of the config file (same as --config)
    config_name [str] - name of the config section (same as --config-name)
    files [list] - list of file names
    num_events [int] - number of events to process (same as --num-events)
    skip_events [int] - number of events to skip (same as --skip-events)
    job_name [str] - job name (same as --job-name)
    modules [list] - list of module names (same as --module)
    num_cpu [int] - number of threads (same as --num-cpu)
    """

    if 'argv' in kw : return _pyana(kw['argv'])

    argv = [sys.argv[0]]
    if 'config' in kw : argv += ['--config', kw['config']]
    if 'config_name' in kw : argv += ['--config-name', kw['config_name']]
    if 'num_events' in kw : argv += ['--num-events', str(kw['num_events'])]
    if 'skip_events' in kw : argv += ['--skip-events', str(kw['skip_events'])]
    if 'job_name' in kw : argv += ['--job-name', kw['job_name']]
    if 'modules' in kw : 
        for mod in kw['modules']:
            argv += ['--module', mod]
    if 'num_cpu' in kw : argv += ['--num-cpu', str(kw['num_cpu'])]
    if 'files' in kw : argv += kw['files']
    
    return _pyana(argv)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    sys.path.insert(0, ".")
    sys.exit( pyana(argv=sys.argv) )
