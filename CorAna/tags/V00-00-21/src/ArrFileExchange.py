#------------------------------------------------------------------------
# File and Version Information:
# $Id$
#------------------------------------------------------------------------

"""Provides asynchronous exchange of ndarray between independent processes

Interface
=========

    Array producer and consumer should use the same "prefix" parameter
    for the numpy array through the file exchange.

    Poducer gets and writes each numpy array in temporary file. When it is done,
    temporary file is moved to the ring buffer with cyclic serial number.

    Consumer finds the latest available file in the ring buffer for given "prefix"
    and loads it, if necessary.

    #-----------------------
    # code in array producer
    #-----------------------
    afe = ArrFileExchange(prefix='./my-numpy-arr') # one call per object
    arr = ...                                      # supply array here
    afe.save_arr(arr)                              # as many times as you need

    #-----------------------
    # code in array consumer
    #-----------------------
    afe = ArrFileExchange(prefix='./my-numpy-arr') # one call per object
    arr = afe.get_arr_latest()                     # as many times as you need

    # optional methods:
    status = afe.is_new_arr_available()            # returns True/False if the new array IS/IS NOT available
                                                   # since last call to arr = afe.get_arr_latest()


Constructor parameters
======================
    prefix='./array' - path to the ring buffer file prefix
    
    rblen=3 - length of the ring buffer (number of saving files with numbers in the range [1,rblen]

    print_bits = 0 - print nothing
               + 1 - info about array is saved in file
               + 2 - info about getting array from file
               + 4 - in get_fname_time_latest() list available in the ring buffer files with creation time
               + 8 - time consumed to rename file
               +16 - input parameters
               +32 - latest found file

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN 
#------------------------------
__version__ = "$Revision$"
# $Source$

#------------------------------
import sys
import os
import numpy as np

from time import time, sleep # for test purpose only

#------------------------------
class ArrFileExchange ( object ) :
    """
    ArrFileExchange - provides asynchronous exchange of ndarray between independent processes
    """
    def __init__ (self, prefix='./array', rblen=3, print_bits=0) :
        self.prefix = prefix
        self.rblen  = rblen
        self.rbnum  = 0
        self.print_bits = print_bits

        self.fname_old  = ''
        self.time_old   = 0
        self.arr_old    = None
        self.is_loaded  = False

        if self.print_bits & 16 : self.print_input_pars()
 
#------------------------------

    def print_input_pars(self) :
        msg = 40*'='+'\nArrFileExchange.print_input_pars():' \
            + '\n  prefix=%s\n  rblen=%d\n  print_bits=%d' % (self.prefix, self.rblen, self.print_bits)
        print msg

#------------------------------
# Methods for writer/producer

    def get_fname_tmp(self) :
        return self.prefix + '-tmp.npy'


    def get_fname_pattern(self) :
        return self.prefix + '-rbn'


    def get_fname_next(self) :
        self.rbnum = self.rbnum+1 if self.rbnum < self.rblen else 1
        return self.get_fname_pattern() + '-%03d.npy'%self.rbnum


    def save_arr(self, arr) :
        """ Save input array in the file with the name composed from
            pre-defined prefix self.prefix and the number from the ring buffer of length self.rblen
        """
        fname_tmp  = self.get_fname_tmp()
        fname_next = self.get_fname_next()

        dir = os.path.dirname(fname_tmp)
        if dir == '' : dir = './'
        if not os.path.exists(dir) :
            print 'WARNING: Directory "%s" does NOT exist! File is NOT saved!' % dir
            return

        cmd = 'mv %s %s' % (fname_tmp, fname_next)
        t0_sec = time()
        np.save(fname_tmp, arr)
        if self.print_bits & 8 : print 'Time consumed to save file  : %f(sec)' % (time()-t0_sec)
        t0_sec = time()
        os.system(cmd)
        if self.print_bits & 8 : print 'Time consumed to rename file: %f(sec)' % (time()-t0_sec)
        if self.print_bits & 1 : print 'Numpy array is saved in file %s' % fname_next

#------------------------------
# Methods for reader/consumer

    def get_file_ctime(self, fname) :
        if os.path.exists(fname) : return os.path.getctime(fname) 
        else :                     return 0


    def get_fname_time_latest(self) :
        """ Returns latest available file name and its creation time in sec
            or empty string and 0 for time, if the file is non-available.
        """
        fname_latest = ''
        t_latest = 0

        dir = os.path.dirname(self.prefix)
        if dir == '' : dir = './'
        #print 'dir = ', dir
        if not os.path.exists(dir) :
            print 'WARNING: Directory "%s" does NOT exist!' % dir
            return fname_latest, t_latest
        
        list_of_files = os.listdir(dir)
        pattern = os.path.basename(self.get_fname_pattern())
        list_for_pattern = [fname for fname in list_of_files if pattern in fname]
        #print 'list_for_pattern = ', list_for_pattern

        for fname in list_for_pattern :
            path = os.path.join(dir,fname)
            ctime = self.get_file_ctime(path)
            if self.print_bits & 4 : print 'File %s creation time %d(s)' % (fname, ctime)
            if ctime > t_latest :
                t_latest = ctime
                fname_latest = path

        if self.print_bits & 32 : print 'Found latest file %s created at %d' % (fname_latest, t_latest)
        return fname_latest, t_latest


    def is_new_arr_available(self) :
        """ Returns True/False if the new file is available since last call to get_arr_latest()
        """
        fname, time = self.get_fname_time_latest()
        if fname == '' : return False      # if file does not exist

        if fname == self.fname_old \
        and time == self.time_old \
        and self.is_loaded : return False  # if file name and time were already seen and file is loaded

        self.fname_old = fname
        self.time_old  = time
        self.is_loaded = False
        return True                        # new file is available


    def get_arr_latest(self) :
        """ Returns the new array if available, othervice returnns old or None (if it was never available)
        """
        if self.is_new_arr_available() :
            if os.path.exists(self.fname_old) :
                if self.print_bits & 2 : print 'Get np.array from latest file: %s' % self.fname_old
                t0_sec = time()
                self.arr_old = np.load(self.fname_old)
                if self.print_bits & 8 : print 'Time consumed to read file : %f(sec)' % (time()-t0_sec)
                self.is_loaded = True
            else :
                if self.print_bits & 2 : print 'File "%s" is NOT available!' % self.fname_old
        else :
            if self.print_bits & 2 : print 'New array is not available, return old or None'

        return self.arr_old 


#------------------------------

if __name__ == "__main__" :

    #print 'sys.argv: ', sys.argv
    path = './roi-img' if len(sys.argv) < 2 else sys.argv[1]
    afe = ArrFileExchange(prefix=path, print_bits=0377)

    for i in range(10) :
        print 10*'='+'\nTest #%d' % i
        arr = afe.get_arr_latest()
        sleep(5)

    sys.exit ('End of %s' % sys.argv[0])

#------------------------------
#------------------------------
#------------------------------
