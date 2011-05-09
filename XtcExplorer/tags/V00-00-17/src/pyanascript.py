#!/usr/bin/python2.4
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: pyana 1097 2010-07-07 23:51:00Z salnikov $
#
# Description:
#  Script pyxtcreader...
#
#------------------------------------------------------------------------

"""Analysis job.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgement.

@see RelatedModule

@version $Id: pyana 1097 2010-07-07 23:51:00Z salnikov $ 

@author Andrei Salnikov
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 1097 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import logging
import multiprocessing as mp
from resource import *

#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import xtc
from pyana import event
from pyana.input import dgramGen, threadedDgramGen
from pyana.userana import mod_import, evt_dispatch
from pyana.mp_proto import *
from pyana.config import config
from pyana.merger import merger

#---------------------
# Local definitions --
#---------------------

def _rusage(msg, ru1, ru2):
    
    print "%s: %.1f user %.1f sys" % ( msg, ru2.ru_utime-ru1.ru_utime, ru2.ru_stime-ru1.ru_stime )


def _proc(jobname, id, pipes, userObjects, dg_ref):
    """ method which is running in child process when we do multi-processing """

    for i in range(len(pipes)):
        pipes[i][0].close()
        if i != id : pipes[i][1].close()
    pipe = pipes[id][1]

    ru1 = getrusage(RUSAGE_SELF)
    
    # create own environment, make unique job name
    env = event.Env(jobname, subproc=id)

    logging.info("proc-%s: starting job %s", id, jobname)

    dispatch = evt_dispatch(userObjects)
    proto = mp_proto(pipe, dg_ref, 'prot-%d' % id)
    nevent = 0

    # event loop
    for req in proto.getRequest():

        # request is a tuple with first element being opcode
        if req[0] == OP_EVENT :

            epics_data = req[1]
            dg = req[2]
            run = req[3]

            evt = event.Event(dg, run)
            
            # update configuration objects
            env.updateConfig(evt)
    
            # update epics data
            env.m_epics.m_id2epics = epics_data
    
            # process all data
            dispatch.dispatch(evt, env)
    
            if evt.seq().service() == xtc.TransitionId.L1Accept : nevent += 1

        elif req[0] == OP_FINISH :
            
            logging.info("proc-%s: received FIN", id)
            dispatch.finish(env)
            
        elif req[0] == OP_RESULT :
            
            logging.info("proc-%s: result requested, tag=%s", id, req[1])
            proto.sendData( env.result() )

        elif req[0] == OP_END :
            
            logging.info("proc-%s: received END", id)
            # stop here
            break
            
    # done with the environment
    env.finish()

    logging.info("proc-%s: processed %d events", id, nevent)

    proto.close()

    ru2 = getrusage(RUSAGE_SELF)
    _rusage( "proc-%s time" % id, ru1, ru2 )

#---------------------------------
#  Application class definition --
#---------------------------------


def main( argv ) :

    ru1 = getrusage(RUSAGE_SELF), getrusage(RUSAGE_CHILDREN)

    # get all options from command line and config file
    jobConfig = config( argv[1:] )

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

    # start worker process if requested
    num_cpu = jobConfig.getJobConfig('num-cpu', 1)
    logging.info("num-cpu: %s", num_cpu)
    nsub = max(num_cpu-1,0)
    dg_ref = jobConfig.getJobConfig('dg-ref', False)
    logging.info("dg-ref: %s", dg_ref)

    conns = [ mp.Pipe() for i in range(nsub)]
    procs = [mp.Process(target=_proc, args=(jobname, i, conns, userObjects, dg_ref)) for i in range(nsub)]
    for p in procs : p.start()
    for conn in conns : conn[1].close()
    pipes = [mp_proto(conn[0], dg_ref, 'prot-main') for conn in conns]

    status = 0
    try :
    
        # create environment
        env = event.Env(jobname)
    
        dispatch = evt_dispatch(userObjects)
        num_events = jobConfig.getJobConfig('num-events')
        skip_events = jobConfig.getJobConfig('skip-events') or 0
        nevent = 0
        ndamage = 0
        damagemask = 0
        next = 0
    
        # read datagrams one by one
        for dgtup in dgramGen( names ) :
    
            if num_events is not None and nevent >= num_events+skip_events :
                logging.info("event limit reached (%d), terminating", nevent)
                break
    
            dg, fileName, fpos, run = dgtup
    
            damage = dg.xtc.damage.value()
            svc = dg.seq.service()
        
            evt = event.Event(dg, run)
            
            # update environment
            env.update ( evt )

            if svc == xtc.TransitionId.L1Accept :
                
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
                    p.sendEventData(dgtup, env)
                else :
                    # each transitions goes to each child
                    for p in pipes : p.sendEventData(dgtup, env)

            else :
                
                # process all data
                dispatch.dispatch(evt, env)
                    
                
        # finish
        dispatch.finish(env)
    
    
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
    
#
#  run application when imported as a main module
#
if __name__ == "__main__" :
    sys.path.insert(0, ".")
    sys.exit( main(sys.argv) )
