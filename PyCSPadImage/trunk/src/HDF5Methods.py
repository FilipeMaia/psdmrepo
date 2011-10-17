#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module HDF5Methods...
#
#------------------------------------------------------------------------

"""This module provides access to the calibration parameters

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 2008-09-22$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#----------
#  Imports 
#----------
import sys
import os
import h5py
#import numpy as np

#import ConfigParameters as cp

#---------------------
#  Class definition --
#---------------------

class HDF5Methods(object) :
    """This class contains a few methods to manipulate with hdf5 files"""

#---------------------

    def __init__ (self) :
        #print """HDF5Methods: Initialization"""
        self.h5file = None
        self.dset   = None   

#---------------------

    def open_hdf5_file(self, fname_input=None) :

        if fname_input==None : #self.fname = cp.confpars.dirName+'/'+cp.confpars.fileName
            print 'open_hdf5_file(fname) : THE HDF5 FILE NAME NEEDS TO BE SPECIFIED !!!'
        else :
            self.fname = fname_input

        print '=== Open HDF5 file: ' + self.fname

        try :
            self.h5file = h5py.File(self.fname,'r') # open read-only
            return self.h5file

        except IOError:
            print 'IOError: CAN NOT OPEN FILE:', self.fname
            return None

#---------------------

    def close_hdf5_file(self) :
        self.h5file.close()
        print '=== Close HDF5 file ==='

#---------------------

    def get_dataset_from_hdf5_file(self,dsname) :
        print 'From hdf5 file get dataset :', dsname

        try :
            self.dset =  self.h5file[str(dsname)]
            return self.dset
        except KeyError:
            print 80*'!'
            print 'WARNING:', dsname, ' DATASET DOES NOT EXIST IN HDF5\n'
            print 80*'!'
            return None

#---------------------

def getOneCSPadEventForTest( fname  = '/reg/d/psdm/CXI/cxi35711/hdf5/cxi35711-r0009.h5',
                             dsname = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDs1.0:Cspad.0/data',
                             event  = 1 ) :
    file    = h5py.File(fname, 'r')
    dataset = file[dsname]
    return dataset[event]

#---------------------

hdf5mets = HDF5Methods()

#----------------------------------------------

def main() :
    event   = 1
    fname   = '/reg/d/psdm/CXI/cxi35711/hdf5/cxi35711-r0009.h5'
    dsname  = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDs1.0:Cspad.0/data'

    h5file = hdf5mets.open_hdf5_file(fname)
    arr = hdf5mets.get_dataset_from_hdf5_file(dsname)
    print 'arr[event]=\n', arr[event]
    hdf5mets.close_hdf5_file()

#---------------------

if __name__ == "__main__" :

    main()
    sys.exit ( 'End of test.' )

#----------------------------------------------
