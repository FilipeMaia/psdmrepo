#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Examples...
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

import CalibPars          as calp
import CSPadConfigPars    as ccp
import CSPadImageProducer as cip

import GlobalGraphics     as gg # For test purpose in main only
import HDF5Methods        as hm # For test purpose in main only

#----------------------------------------------

def main_example_alignment_test() :

    print 'Start test in main_example_cxi()'

    #path_calib, runnum = '/reg/d/psdm/CXI/cxi80410/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0', 1            # 2011-05-25
    #path_calib, runnum = '/reg/d/psdm/CXI/cxi37411/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0', 1            # 2011-08-10
    #path_calib, runnum = '/reg/d/psdm/CXI/cxi35711/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0', 1            # 2011-10-18
    path_calib, runnum = '/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi43312-Dsd', 1         # 2012-01-12
    #path_calib, runnum = '/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi80410-r1150-Ds1', 1150 # 2012-01-18
    

    #fname  = '/reg/d/psdm/CXI/cxi35711/hdf5/cxi35711-r0009.h5'
    #fname  = '/reg/d/psdm/CXI/cxi37411/hdf5/cxi37411-r0080.h5'
    #fname   = '/reg/d/psdm/CXI/cxi37411/hdf5/cxi37411-r0039.h5'
    fname   = '/reg/d/psdm/CXI/cxi80410/hdf5/cxi80410-r1150.h5'

    dsname = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDsd.0:Cspad.0/data'
    #dsname  = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDs1.0:Cspad.0/data'
    event   = 0

    print 'Load calibration parameters from', path_calib 
    calp.calibpars.setCalibParsForPath ( run = runnum, path = path_calib )
    calp.calibpars.printCalibPars()
    #calp.calibpars.printCalibFiles ()
    #calp.calibpars.printListOfCalibTypes()

    print 'Get raw CSPad event %d from file %s \ndataset %s' % (event, fname, dsname)
    ds1ev = hm.getOneCSPadEventForTest( fname, dsname, event )
    print 'ds1ev.shape = ',ds1ev.shape

    print 'Make the CSPad image from raw array'
    cspadimg = cip.CSPadImageProducer()
    arr = cspadimg.getImageArrayForCSPadElement( ds1ev )

    print 'Plot CSPad image'

    AmpRange = (-10, 40)

    gg.plotImage(arr,range=AmpRange,figsize=(11.6,10))
    gg.move(200,100)
    gg.plotSpectrum(arr,range=AmpRange)
    gg.move(50,50)
    #gg.plotImageAndSpectrum(arr,range=(1,2001))
    print 'To EXIT the test click on "x" in the top-right corner of each plot window.'
    gg.show()

#----------------------------------------------

def main_example_cxi() :

    print 'Start test in main_example_cxi()'

    #path_calib, runnum = '/reg/d/psdm/CXI/cxi37411/calib/CsPad::CalibV1/CxiDsd.0:Cspad.0'
    #path_calib = '/reg/d/psdm/CXI/cxi35711/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0'
    path_calib  = '/reg/d/psdm/CXI/cxi37411/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0'

    #fname  = '/reg/d/psdm/CXI/cxi35711/hdf5/cxi35711-r0009.h5'
    #fname  = '/reg/d/psdm/CXI/cxi37411/hdf5/cxi37411-r0080.h5'
    fname   = '/reg/d/psdm/CXI/cxi37411/hdf5/cxi37411-r0039.h5'

    #dsname = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDsd.0:Cspad.0/data'
    dsname  = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDs1.0:Cspad.0/data'
    event   = 0

    print 'Load calibration parameters from', path_calib 
    calp.calibpars.setCalibParsForPath ( run = 1, path = path_calib )
    
    print 'Get raw CSPad event %d from file %s \ndataset %s' % (event, fname, dsname)
    ds1ev = hm.getOneCSPadEventForTest( fname, dsname, event )
    print 'ds1ev.shape = ',ds1ev.shape

    print 'Make the CSPad image from raw array'
    cspadimg = cip.CSPadImageProducer()
    arr = cspadimg.getImageArrayForCSPadElement( ds1ev )

    print 'Plot CSPad image'
    gg.plotImage(arr,range=(-10,90),figsize=(11.6,10))
    gg.move(200,100)
    gg.plotSpectrum(arr,range=(-10,90))
    gg.move(50,50)
    #gg.plotImageAndSpectrum(arr,range=(1,2001))
    print 'To EXIT the test click on "x" in the top-right corner of each plot window.'
    gg.show()

#----------------------------------------------

def main_example_xpp() :

    print 'Start test in main_example_xpp()'

    path_calib = '/reg/d/psdm/xpp/xpp47712/calib/CsPad::CalibV1/XppGon.0:Cspad.0'
    fname      = '/reg/d/psdm/xpp/xpp47712/hdf5/xpp47712-r0043.h5'
    dsname     = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/XppGon.0:Cspad.0/data'

    #path_calib = '/reg/d/psdm/XPP/xpp36211/calib/CsPad::CalibV1/XppGon.0:Cspad.0'
    #path_calib = '/reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-xpp36211-r0544'
    #fname      = '/reg/d/psdm/xpp/xpp36211/hdf5/xpp36211-r0073.h5'
    #dsname     = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/XppGon.0:Cspad.0/data'

    event      = 0

    print 'Load calibration parameters from', path_calib 
    calp.calibpars.setCalibParsForPath ( run = 1, path = path_calib )

    print 'Get raw CSPad event %d from file %s \ndataset %s' % (event, fname, dsname)
    ds1ev = hm.getOneCSPadEventForTest( fname, dsname, event )
    print 'ds1ev.shape = ',ds1ev.shape

    print 'Make the CSPad image from raw array'
    cspadimg = cip.CSPadImageProducer()
    arr = cspadimg.getImageArrayForCSPadElement( ds1ev )

    print 'Plot CSPad image'
    gg.plotImage(arr,range=(1200,2000),figsize=(11.6,10))
    gg.move(200,100)
    gg.plotSpectrum(arr,range=(1200,2000))
    gg.move(50,50)
    print 'To EXIT the test click on "x" in the top-right corner of each plot window.'
    gg.show()

#----------------------------------------------

if __name__ == "__main__" :

    main_example_alignment_test()
    #main_example_cxi()
    #main_example_xpp()
    sys.exit ( 'End of test.' )

#----------------------------------------------
