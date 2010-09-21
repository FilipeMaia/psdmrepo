#
# $Id$
#
# Copyright (c) 2010 SLAC National Accelerator Laboratory
# 

import logging
import threading
import Queue

from pypdsdata import io

_log = logging.getLogger("pyana.input")


def dgramGen(names):
    """ Datagram generator """

    # group files by run number
    runfiles = {}
    for n in names : 
        xname = io.XtcFileName(n)
        runfiles.setdefault(xname.run(), []).append(n)

    # scan all runs
    runs = runfiles.keys()
    runs.sort()
    for run in runs :
        
        names = runfiles[run]
        
        logging.info("Processing run number %s" % run)
        logging.info("File list: %s" % names)

        # read datagrams one by one
        dgiter = io.XtcMergeIterator( names )
        for dg in dgiter :
            
            fileName = dgiter.fileName()
            fpos = dgiter.fpos()
            run = dgiter.run()

            yield (dg, fileName, fpos, run)


class _DgramReaderThread ( threading.Thread ):
    
    def __init__(self, queue, names):
        
        threading.Thread.__init__(self, name="DgramReader")
        
        self.queue = queue
        self.names = names
        
    def run(self) :
        
        for dg in dgramGen(self.names) :
            self.queue.put(dg)
        # signal end of data
        self.queue.put(None)
            

def threadedDgramGen( names, queueSize = 10 ):

    """ datagram generator which does reading in a separate thread """

    queue = Queue.Queue(queueSize)
    thread = _DgramReaderThread(queue, names)
    thread.start()
    
    while True:
        
        dg = queue.get()
        if dg is None : break
        yield dg
        
    # join the thread
    thread.join()
    