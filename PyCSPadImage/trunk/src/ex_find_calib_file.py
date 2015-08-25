#!/usr/bin/env python
#------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ex_cspad_img...
#
#------------------------------------------------------------------------

"""This module provides an examples of how to get and plot CSPad image

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see PyCSPadImage.CalibPars, PyCSPadImage.CSPADPixCoords

@version $Id: 2014-07-02$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#----------
#  Imports 
#----------
import os
import sys
import numpy as np

import PyCSPadImage.CalibPars as calp

from PSCalib.CalibFileFinder import CalibFileFinder

#------------------------------

def test_01() :
    """ Test of access to calibration file
    """
    path_to_clib_types = '/reg/d/psdm/xpp/xppi0815/calib/CsPad::CalibV1/XppGon.0:Cspad.0'
    runnum = 120
    type = 'pedestals'
    print '  path_to_clib_types: %s\n  type: %s\n  runnum: %d' % (path_to_clib_types, type, runnum)

    #calibstore = calp.CalibPars(path=path_calib, run=runnum)
    #pedestals = calibstore.getCalibPars('pedestals', runnum)

    fname = calp.findCalibFile(path_to_clib_types, type, runnum) 
    print '  calibration file name: %s' % fname

#------------------------------

def test_02() :
    """ Test of access to calibration file
    """

    cdir  = '/reg/d/psdm/xpp/xppi0815/calib'
    group = 'CsPad::CalibV1'
    src   = 'XppGon.0:Cspad.0'
    type  = 'pedestals'
    rnum  = 120

    print '  cdir: %s\n  type: %s\n  runnum: %d' % (cdir, type, rnum)

    cff = CalibFileFinder(cdir, group, pbits=0377)
    fname = cff.findCalibFile(src, type, rnum)

    print '  calibration file name: %s' % fname

 
#------------------------------

if __name__ == "__main__" :
    if len(sys.argv) <2     : test_01()
    elif sys.argv[1] == '1' : test_01()
    elif sys.argv[1] == '2' : test_02()
    sys.exit ( 'End of %s' % sys.argv[0] )

#------------------------------
