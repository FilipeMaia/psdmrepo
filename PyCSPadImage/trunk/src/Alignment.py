#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Alignment...
#
#------------------------------------------------------------------------

"""This module provides examples of how to get and use the CSPad image

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 2011-11-18$

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
import numpy              as np

import CalibParsDefault   as cald
import CalibPars          as calp
import CalibParsEvaluated as cpe
import CSPadConfigPars    as ccp
import CSPadImageProducer as cip
import GlobalMethods      as gm # getCSPadArrayFromFile for pedestal subtraction 

import GlobalGraphics     as gg # For test purpose in main only
import HDF5Methods        as hm # For test purpose in main only

#----------------------------------------------

def main_alignment_test() :

    print 'Start test in main_alignment_test()'

    #path_calib = '/reg/d/psdm/CXI/cxi80410/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0'            # 2011-05-25
    #path_calib = '/reg/d/psdm/CXI/cxi37411/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0'            # 2011-08-10
    #path_calib = '/reg/d/psdm/CXI/cxi35711/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0'            # 2011-10-18
    #path_calib = '/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi43312-Dsd'         # 2012-01-12
    #path_calib = '/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi80410-r1150-Ds1'   # 2012-01-18
    #path_calib = '/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi39112-r0009-Ds1'    # 2012-02-17
    #path_calib = '/reg/d/psdm/CXI/cxi39112/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0'             # 2012-02-17
    #path_calib = '/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-xpp-2012-02-26/'      # 2012-02-26
    #path_calib = '/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi49812-r0073-Ds1'   # 2012-03-08
    #path_calib = '/reg/d/psdm/CXI/cxi49812/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/'           # 2012-03-08    
    #path_calib = '/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi49012-r0020-Ds1/'  # 2012-03-14
    path_calib = '/reg/d/psdm/CXI/cxi49012/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/'           # 2012-03-14    
    #path_calib = '/reg/d/psdm/XPP/xppcom10/calib/CsPad::CalibV1/XppGon.0:Cspad.0'            # 2012-03-23 check    

    #fname, runnum = '/reg/d/psdm/CXI/cxi35711/hdf5/cxi35711-r0009.h5',    9 
    #fname, runnum = '/reg/d/psdm/CXI/cxi37411/hdf5/cxi37411-r0080.h5',   80
    #fname, runnum = '/reg/d/psdm/CXI/cxi37411/hdf5/cxi37411-r0039.h5',   39 
    #fname, runnum = '/reg/d/psdm/CXI/cxi80410/hdf5/cxi80410-r1150.h5', 1150
    #fname, runnum = '/reg/d/psdm/CXI/cxi39112/hdf5/cxi39112-r0009.h5',    9 
    #fname, runnum = '/reg/d/psdm/XPP/xppcom10/hdf5/xppcom10-r1437.h5', 1437
    #fname, runnum = '/reg/d/psdm/CXI/cxi49812/hdf5/cxi49812-r0073.h5',   73
    fname, runnum = '/reg/d/psdm/CXI/cxi49012/hdf5/cxi49012-r0020-raw.h5', 20
    #fname, runnum = '/reg/d/psdm/CXI/cxi80410/hdf5/cxi80410-r0628.h5',  628

    #dsname = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/XppGon.0:Cspad.0/data'
    #dsname = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDsd.0:Cspad.0/data'
    dsname = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDs1.0:Cspad.0/data'

    event   = 1

    print 'Load calibration parameters from', path_calib 
    calp.calibpars.setCalibParsForPath ( run=runnum, path=path_calib )
    #calp.calibpars.printCalibPars()
    #calp.calibpars.printCalibFiles ()
    #calp.calibpars.printListOfCalibTypes()
    #cald.calibparsdefault.printListOfCalibTypes()
    #cald.calibparsdefault.printCalibParsDefault()
    #cald.calibparsdefault.printCalibParsDefault('center_global')
    cpe.cpeval.printCalibParsEvaluatedAll() 

    print 'Get raw CSPad event %d from file %s \ndataset %s' % (event, fname, dsname)
    #ds1ev = hm.getOneCSPadEventForTest( fname, dsname, event )
    ds1ev = hm.getAverageCSPadEvent( fname, dsname, event, nevents=50 )
    print 'ds1ev.shape = ',ds1ev.shape # should be (32, 185, 388)
    #print 'ds1ev = ',ds1ev[1,:]

    #print 'Subtract pedestals'
    #ped_fname = '/reg/neh/home/dubrovin/LCLS/calib-CSPad-pedestals/cspad-pedestals-cxi49812-r0072.dat' # shape = (5920, 388)
    #ped_fname = '/reg/neh/home/dubrovin/LCLS/calib-CSPad-pedestals/cspad-pedestals-cxi49012-r0008.dat' # shape = (5920, 388)
    #ped_fname = '/reg/neh/home/dubrovin/LCLS/calib-CSPad-pedestals/cspad-pedestals-cxi49012-r0038.dat' # shape = (5920, 388)
    ped_fname = '/reg/neh/home/dubrovin/LCLS/calib-CSPad-pedestals/cspad-pedestals-cxi49012-r0027.dat' # shape = (5920, 388)
    #ped_fname = '/reg/neh/home/dubrovin/LCLS/calib-CSPad-pedestals/cspad-pedestals-xppcom10-r1435.dat' # shape = (5920, 388) low gain
    #ped_fname = '/reg/d/psdm/CXI/cxi49012/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/pedestals/9-37.data' # shape = (5920, 388)
    #ds1ev  = gm.getCSPadArrayFromFile(ped_fname)
    #ds1ev -= gm.getCSPadArrayFromFile('/reg/neh/home/dubrovin/LCLS/calib-CSPad-pedestals/cspad-pedestals-cxi49012-r0027.dat')
    #ds1ev -= gm.getCSPadArrayFromFile('/reg/neh/home/dubrovin/LCLS/calib-CSPad-pedestals/cspad-pedestals-xppcom10-r1442.dat')

    ds1ev -= gm.getCSPadArrayFromFile(ped_fname)

    print 'Make the CSPad image from raw array'
    cspadimg = cip.CSPadImageProducer(rotation=0, tiltIsOn=True)#, mirror=True)
    #cspadimg.printInputPars()
    #cspadimg.printGeometryPars()
    #arr = cspadimg.getImageArrayForCSPadElement( ds1ev )
    arr = cspadimg.getCSPadImage( ds1ev )

    print 'Plot CSPad image'
    #AmpRange = (0, 800)
    #AmpRange = (900,  1400)
    AmpRange = (-10, 40)
    #AmpRange = (0, 100)

    gg.plotImage(arr,range=AmpRange,figsize=(11.6,10))
    gg.move(200,100)
    gg.plotSpectrum(arr,range=AmpRange)
    gg.move(50,50)
    #gg.plotImageAndSpectrum(arr,range=(1,2001))
    print 'To EXIT the test click on "x" in the top-right corner of each plot window.'
    gg.show()

    #ds1ev.shape = (5920, 388)
    #gm.saveNumpyArrayInFile(ds1ev, fname='cspad-arr.txt') 

#----------------------------------------------

if __name__ == "__main__" :

    main_alignment_test()
    #main_example_cxi() # see Examples.py
    #main_example_xpp() # see Examples.py
    sys.exit ( 'End of test.' )

#----------------------------------------------
