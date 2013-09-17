#!/usr/bin/env python

#--------------------

import numpy as np
import os
import sys

#--------------------

class SpectralArray :
    """Holds everything for 2D spactral array"""


    def __init__ ( self, fname ) :
        """Constructor initialization"""
        self.set_default_shape_parameters()
        self.get_shape_from_file(fname+'.sha') 
        self.get_array_from_file(fname)


    def get_array_from_file (self, fname) :
        print """Get array from file""", fname
        self.arr = np.loadtxt(fname, dtype=np.float32)


    def set_default_shape_parameters (self) :
        """Set default values for all shape parameters"""
        self.npixels =   0
        self.nbins   = 100
        self.amin    =   0
        self.amax    = 100
        self.nevents =   0
        self.fname   = ' '


    def get_shape_from_file (self, fname) :
        print """Get shape from file""", fname
        if os.path.exists(fname) :
            f=open(fname,'r')
            for line in f :
                if len(line) == 1 : continue # line is empty
                key = line.split()[0]
                val = line.split()[1]
                if   key == 'NPIXELS'  : self.npixels = int(val)
                elif key == 'NBINS'    : self.nbins   = int(val)
                elif key == 'AMIN'     : self.amin    = float(val)
                elif key == 'AMAX'     : self.amax    = float(val)
                elif key == 'NEVENTS'  : self.nevents = int(val)
                elif key == 'ARRFNAME' : self.fname   = val
                else :
                    print 'The record : %s %s \n is UNKNOWN in get_shape_from_file ()' % (key, val) 
            f.close()
        else :
            print 'The file %s does not exist' % (fname)
            print 'WILL USE DEFAULT CONFIGURATION PARAMETERS'


    def print_shape_parameters (self) :
        """Print shape parameters"""
        print 'NPIXELS  :', self.npixels
        print 'NBINS    :', self.nbins  
        print 'AMIN     :', self.amin   
        print 'AMAX     :', self.amax   
        print 'NEVENTS  :', self.nevents
        print 'ARRFNAME :', self.fname  


    def print_array_subset (self) :
        """Print a few array elements"""
        for pix in range(min(5,self.npixels)) :
            print '\nPixel', pix, 'spectrum:'
            for bin in range(self.nbins) :
                print self.arr[pix][bin],

#--------------------

def main_test() :
    """Test example"""
    sa = SpectralArray ('sxr16410-r0081-opal-camera-pix-spectra.txt')
    sa.print_shape_parameters ()
    sa.print_array_subset ()


if __name__ == "__main__" :
    main_test()
    sys.exit ( 'End of test' )

#--------------------
