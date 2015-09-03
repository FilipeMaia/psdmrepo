#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CalibParsStore...
#
#------------------------------------------------------------------------

"""
:py:class:`PSCalib.CalibParsStore` - is a factory class/method to switch between different device-dependent
segments/sensors to access their pixel geometry uling :py:class:`PSCalib.SegGeometry` interface.

Usage::

    # Import
    from PSCalib.CalibParsStore import cps
    from PSCalib.GlobalUtils import *

    # Initialization
    calibdir = env.calibDir()  # or '/reg/d/psdm/<INS>/<experiment>/calib'
    group = None               # or 'CsPad::CalibV1'
    source = 'Camp.0:pnCCD.1'
    runnum = 10                # or evt.run()
    pbits = 255
    cpstore = cps.Create(calibdir, group, source, runnum, pbits)

    # Access methods
    nda = cpstore.pedestals()
    nda = cpstore.pixel_status()
    nda = cpstore.pixel_rms()
    nda = cpstore.pixel_mask()
    nda = cpstore.pixel_gain()
    nda = cpstore.pixel_bkgd()
    nda = cpstore.common_mode()

    status = gcp.status(ctype=PEDESTALS) # see list of ctypes in :py:class:`PSCalib.GlobalUtils`
    shape  = gcp.shape(ctype)
    size   = gcp.size(ctype)
    ndim   = gcp.ndim(ctype)

@see other interface methods in :py:class:`PSCalib.CalibPars`, :py:class:`PSCalib..CalibParsStore`


This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: 2013-03-08$

@author Mikhail S. Dubrovin
"""

#--------------------------------
#  Module's version from CVS --
#--------------------------------
__version__ = "$Revision$"
# $Source$
#--------------------------------

import sys
#import math
#import numpy as np
#from time import time

#------------------------------

import PSCalib.GlobalUtils            as gu
from PSCalib.GenericCalibPars         import GenericCalibPars

from PSCalib.CalibParsBaseAndorV1     import CalibParsBaseAndorV1    
from PSCalib.CalibParsBaseCameraV1    import CalibParsBaseCameraV1   
from PSCalib.CalibParsBaseCSPad2x2V1  import CalibParsBaseCSPad2x2V1 
from PSCalib.CalibParsBaseCSPadV1     import CalibParsBaseCSPadV1    
from PSCalib.CalibParsBaseEpix100aV1  import CalibParsBaseEpix100aV1 
from PSCalib.CalibParsBasePnccdV1     import CalibParsBasePnccdV1    
from PSCalib.CalibParsBasePrincetonV1 import CalibParsBasePrincetonV1
from PSCalib.CalibParsBaseAcqirisV1   import CalibParsBaseAcqirisV1

#------------------------------

class CalibParsStore() :
    """Factory class for CalibPars object of different detectors"""

#------------------------------

    def __init__(self) :
        self.name = self.__class__.__name__

#------------------------------

#------------------------------

    def Create(self, calibdir, group, source, runnum, pbits=0) :
        """ Factory method
            @param calibdir - [string] calibration directory, ex: /reg/d/psdm/AMO/amoa1214/calib
            @param group    - [string] group, ex: PNCCD::CalibV1
            @param source   - [string] data source, ex: Camp.0:pnCCD.0
            @param runnum   - [int]    run number, ex: 10
            @param pbits=0  - [int] print control bits, ex: 255
        """        

        dettype = gu.det_src_to_type(source)
        grp = group if group is not None else gu.dic_det_type_to_calib_group[dettype]

        if pbits : print '%s: Detector type = %d: %s' % (self.name, dettype, gu.dic_det_type_to_name[dettype])

        cbase = None
        if   dettype == gu.CSPAD     : cbase = CalibParsBaseCSPadV1()
        elif dettype == gu.CSPAD2X2  : cbase = CalibParsBaseCSPad2x2V1() 
        elif dettype == gu.PNCCD     : cbase = CalibParsBasePnccdV1()    
        elif dettype == gu.PRINCETON : cbase = CalibParsBasePrincetonV1()
        elif dettype == gu.ANDOR     : cbase = CalibParsBaseAndorV1()    
        elif dettype == gu.EPIX100A  : cbase = CalibParsBaseEpix100aV1() 
        elif dettype == gu.ACQIRIS   : cbase = CalibParsBaseAcqirisV1() 
        else :
            for det in (gu.OPAL1000, gu.OPAL2000, gu.OPAL4000, gu.OPAL8000, gu.TM6740, gu.ORCAFL40, gu.FCCD960) :
                if dettype == det : cbase = CalibParsBaseCameraV1()

        if cbase is not None :
            return GenericCalibPars(cbase, calibdir, grp, source, runnum, pbits)

        print 'Calibration parameters for source: %s are not implemented in class %s' % (source, self.__class__.__name__)
        return None

#------------------------------

cps = CalibParsStore()

#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------

import numpy as np

def print_nda(nda, cmt='') :
    arr = nda if isinstance(nda, np.ndarray) else np.array(nda) 
    str_arr = str(arr) if arr.size<5 else str(arr.flatten()[0:5])
    print '%s %s: shape=%s, size=%d, dtype=%s, data=%s' % \
          (cmt, type(nda), str(arr.shape), arr.size, str(arr.dtype), str_arr)

#------------------------------

def test_cps() :

    if len(sys.argv)==1   : print 'For test(s) use command: python %s <test-number=1-3>' % sys.argv[0]

    calibdir = '/reg/d/psdm/CXI/cxif5315/calib'
    group    = None # will be substituted from dictionary or 'CsPad::CalibV1' 
    source   = 'CxiDs2.0:Cspad.0'
    runnum   = 60
    pbits    = 0
 
    if(sys.argv[1]=='1') :
        cp = cps.Create(calibdir, group, source, runnum, pbits)
        cp.print_attrs()

        print_nda(cp.pedestals(),    'pedestals')
        print_nda(cp.pixel_rms(),    'pixel_rms')
        print_nda(cp.pixel_mask(),   'pixel_mask')
        print_nda(cp.pixel_status(), 'pixel_status')
        print_nda(cp.pixel_gain(),   'pixel_gain')
        print_nda(cp.common_mode(),  'common_mode')
        print_nda(cp.pixel_bkgd(),   'pixel_bkgd') 
        print_nda(cp.shape(),        'shape')
 
        print 'size=%d' % cp.size()
        print 'ndim=%d' % cp.ndim()

        statval = cp.status(gu.PEDESTALS)
        print 'status(PEDESTALS)=%d: %s' % (statval, gu.dic_calib_status_value_to_name[statval])

        statval = cp.status(gu.PIXEL_GAIN)
        print 'status(PIXEL_GAIN)=%d: %s' % (statval, gu.dic_calib_status_value_to_name[statval])
 
    else : print 'Non-expected arguments: sys.argv = %s use 1,2,...' % sys.argv

#------------------------------

if __name__ == "__main__" :
    test_cps()
    sys.exit( 'End of %s test.' % sys.argv[0])

#------------------------------
