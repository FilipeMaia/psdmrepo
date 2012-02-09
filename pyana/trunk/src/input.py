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
    """dgramGen(names) -> datagram
    
    This is the generator method which takes the list of XTC file names
    and produces a sequence of XTC datagrams. As a generated it should be 
    called repeatedly for each new datagram. Generation stops when there is 
    no more datagrams left in the input files.
    
    This method uses pypdsdata.io.XtcMergeIterator class to correctly
    merge multiple XTC streams and chunks into a single stream of 
    time-ordered histrograms.
    """

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
    """
    Class for thread which reads datagrams using dgramGen() generator
    and fills the queue with the datagrams
    """
    
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
    """threadedDgramGen(names, queueSize = 10) -> datagram
    
    This is the generator method which takes the list of XTC file names
    and produces a sequence of XTC datagrams very much like dgramGen(). 
    Unlike in dgramGen() actual reading of the datagfram happens in a 
    separate thread which should reduce waiting time for the consumer
    of the datagrams. Parameter ``queueSize`` controls maximum depth of
    the datagram queue, for optimal performance it should not be very large.
    """

    queue = Queue.Queue(queueSize)
    thread = _DgramReaderThread(queue, names)
    thread.start()
    
    while True:
        
        dg = queue.get()
        if dg is None : break
        yield dg
        
    # join the thread
    thread.join()
    