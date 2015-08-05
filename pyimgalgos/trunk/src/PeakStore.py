#------------------------------
"""Class helps to save text file with information about peaks found in peak-finders. 

Usage::

    # Imports
    from pyimgalgos.PeakStore import PeakStore

    # Usage
    pstore = PeakStore(env, 5, prefix='xxx', add_header='TitV1 TitV2 TitV3 ...', pbits=255)
    for peak in peaks :
        rec = '%s %d %f ...' % (peak[0], peak[5], peak[7],...)
        pstore.save_peak(evt, rec)
    pstore.close()

    # Print methods
    pstore.print_attrs()

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

#--------------------------------

import psana
import numpy as np
from time import strftime, localtime #, gmtime

#------------------------------

class PeakStore :

    def __init__(self, env, runnum=0, prefix=None, add_header='Evnum etc...' , pbits=0) :  
        self.fout   = None
        self.pbits  = pbits
        self.exp    = env.experiment()
        self.runnum = runnum 
        self.set_file_name(prefix)
        self.set_header(add_header)
        self.open_file()
        self.counter = 0

##-----------------------------

    def print_attrs(self) :
        msg = 'Attributes of %s' % self.__class__.__name__ +\
              '\n prefix: %s' % str(self.prefix) +\
              '\n fname: %s' % str(self.fname) +\
              '\n title: %s' % self.header
        print msg

##-----------------------------

    def set_file_name(self, prefix=None) :
        """Sets the name of the file with peak info
        """
        self.prefix = prefix        
        if prefix is None : prefix='peaks'
        tstamp = strftime('%Y-%m-%dT%H:%M:%S', localtime())
        self.fname = '%s-%s-r%04d-%s.txt' % (self.prefix, self.exp, self.runnum, tstamp)
 
##-----------------------------

    def open_file(self) :  
        if self.fname is not None :
            self.fout = open(self.fname,'w')
            self.fout.write('%s\n' % self.header)
            if self.pbits & 1 : print 'Open output file with peaks: %s' % self.fname

##-----------------------------

    def close_file(self) :  
        self.fout.close()
        if self.pbits & 1 : print 'Close file %s with %d peaks' % (self.fname, self.counter)

##-----------------------------

    def set_header(self, add_header='Evnum etc...') :  
        """Returns a string of comments for output file with list of peaks
        """
        self.header = '%s  %s' %\
            ('# Exp     Run  Date       Time      time(sec)   time(nsec) fiduc', add_header )
        if self.pbits & 2 : print 'Hdr : %s' % (self.header)

##-----------------------------

    def rec_evtid(self, evt) :
        """Returns a string with event identidication info: exp, run, evtnum, tstamp, etc.
        """
        evtid = evt.get(psana.EventId)
        time_sec, time_nsec = evtid.time()
        tstamp = strftime('%Y-%m-%d %H:%M:%S', localtime(time_sec))
        return '%8s  %3d  %s  %10d  %9d  %5d' % \
               (self.exp, evtid.run(), tstamp, time_sec, time_nsec, evtid.fiducials())

#------------------------------
    
    def save_peak(self, evt, peak_rec='') :
        """Save event id with peak record (string) in the text file 
        """
        rec = '%s %s' % (self.rec_evtid(evt), peak_rec)
        if self.fout is not None : self.fout.write('%s\n' % rec)
        self.counter += 1
        #self.evt_peaks.append(peak)
        if self.pbits & 2 : print '%5d: %s' % (self.counter, rec)
    
##-----------------------------

#--------------------------------
#-----------  TEST  -------------
#--------------------------------

if __name__ == "__main__" :
 
    pass

#--------------------------------
