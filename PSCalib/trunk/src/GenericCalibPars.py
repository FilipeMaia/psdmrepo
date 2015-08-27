#!/usr/bin/env python
#------------------------------
""" GenericCalibPars - implementation of CalibPars interface methods for generic detectors.

Usage::

    # THIS CLASS IS NOT SUPPOSED TO BE USED AS SELF-DEPENDENT...
    # USE :py:class:`PSCalib.CalibParsStore`

    from PSCalib.GenericCalibPars import GenericCalibPars

    from PSCalib.CalibParsBaseAndorV1     import CalibParsBaseAndorV1    
    from PSCalib.CalibParsBaseCameraV1    import CalibParsBaseCameraV1   
    from PSCalib.CalibParsBaseCSPad2x2V1  import CalibParsBaseCSPad2x2V1 
    ...
    from PSCalib.CalibParsBasePnccdV1     import CalibParsBasePnccdV1    

    cbase = CalibParsBasePnccdV1()    

    calibdir = '/reg/d/psdm/CXI/cxif5315/calib'
    group    = 'PNCCD::CalibV1'
    source   = 'CxiDs2.0:Cspad.0'
    runnum   = 60
    pbits    = 255

    gcp = GenericCalibPars(cbase, calibdir, group, source, runnum, pbits)

    nda = gcp.pedestals()
    nda = gcp.pixel_rms()
    nda = gcp.pixel_mask()
    nda = gcp.pixel_bkgd()
    nda = gcp.pixel_status()
    nda = gcp.pixel_gain()
    nda = gcp.common_mode()

    status = gcp.status(ctype=PEDESTALS) # see  list of ctypes in :py:class:`PSCalib.GlobalUtils`
    shape  = gcp.shape(ctype)
    size   = gcp.size(ctype)
    ndim   = gcp.ndim(ctype)
    
@see :py:class:`PSCalib.CalibPars`, :py:class:`PSCalib.CalibParsStore`, :py:class:`PSCalib.CalibParsCspad2x1V1, :py:class:`PSCalib.GlobalUtils`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Revision: $Revision$

@version $Id$

@author Mikhail S. Dubrovin
"""
#--------------------------------
__version__ = "$Revision$"
#--------------------------------

import sys
#import os
#import math
import numpy as np
from PSCalib.CalibPars import CalibPars
from PSCalib.CalibFileFinder import CalibFileFinder

from PSCalib.GlobalUtils import *

#------------------------------

class GenericCalibPars (CalibPars) :

#------------------------------

    def __init__(self, cbase, calibdir, group, source, runnum, pbits=255) : 
        """ Constructor
        """
        CalibPars.__init__(self)

        self.cbase    = cbase    # ex.: PSCalib.CalibParsBaseAndorV1
        self.calibdir = calibdir # ex.: '/reg/d/psdm/CXI/cxif5315/calib'
        self.group    = group    # ex.: 'PNCCD::CalibV1'
        self.source   = source   # ex.: 'CxiDs2.0:Cspad.0' 
        self.runnum   = runnum   # ex.: 10
        self.pbits    = pbits    # ex.: 255 

        self.cff      = CalibFileFinder(calibdir, group, 0377 if pbits else 0)   

        self.dic_status = {    
            PEDESTALS    : UNDEFINED,
            PIXEL_STATUS : UNDEFINED,
            PIXEL_RMS    : UNDEFINED,
            PIXEL_GAIN   : UNDEFINED,
            PIXEL_MASK   : UNDEFINED,
            PIXEL_BKGD   : UNDEFINED,
            COMMON_MODE  : UNDEFINED }

#------------------------------

    def print_attrs(self) :
        """ Prints attributes
        """
        inf = '\nAttributes of %s object:' % self.__class__.__name__ \
            + '\n  base object: %s' % self.cbase.__class__.__name__ \
            + '\n  calibdir   : %s' % self.calibdir \
            + '\n  group      : %s' % self.group    \
            + '\n  source     : %s' % self.source   \
            + '\n  runnum     : %s' % self.runnum   \
            + '\n  pbits      : %s' % self.pbits    
        print inf

#------------------------------

    def msgh(self, i=3) :
        """ Returns message header
        """
        if   i==3 : return '%s: source %s run=%d' % (self.__class__.__name__, self.source, self.runnum)
        elif i==2 : return '%s: source %s' % (self.__class__.__name__, self.source) 
        else      : return '%s:' % (self.__class__.__name__)

#------------------------------

    def msgw(self) :
        return '%s %s' % ('Implementation of method %s in class', self.__class__.__name__)

#------------------------------

    def constants_default(self, ctype) :
        """ Returns numpy array with default constants

        Logic:
        1) if constants for common mode - return default numpy array
        2) if base size of calibration constants is 0 (for variable image size cameras)
           - return None (they can be loaded from file only!
        3) for PEDESTALS, PIXEL_STATUS, PIXEL_BKGD return numpy array of **zeros** for base shape and dtype
        4) for all other calibration types return numpy array of **ones** for base shape and dtype

        """

        tname = dic_calib_type_to_name[ctype]
        if self.pbits : print 'INFO %s: load default constants of type %s' % (self.msgh(3), tname)

        if ctype == COMMON_MODE :
            self.dic_status[ctype] = DEFAULT
            return np.array(self.cbase.cmod, dtype = dic_calib_type_to_dtype[ctype])

        if self.cbase.size == 0 :
            if self.pbits : print 'WARNING %s: default constants of type %s' % (self.msgh(3), tname) \
                                  + ' are not available for variable size cameras.'\
                                  + '\n  Check if the file with calibration constanrs is available in calib directory.'
            return None

        self.dic_status[ctype] = DEFAULT

        if ctype == PEDESTALS \
        or ctype == PIXEL_STATUS \
        or ctype == PIXEL_BKGD :
            return np.zeros(self.cbase.shape, dtype = dic_calib_type_to_dtype[ctype])

        else : # for PIXEL_RMS, PIXEL_GAIN, PIXEL_MASK
            return np.ones(self.cbase.shape, dtype = dic_calib_type_to_dtype[ctype])

#------------------------------

    def constants(self, ctype) :
        """ Returns numpy array with calibration constants for specified type

        Logic:
        1) if calib file is not found:
           - return result from constants_default(ctype)
        2) try to load numpy array from file
           -- exception - return result from constants_default(ctype)
        3) if constants for common mode - return numpy array as is
        4) if base size==0 - return numpy array as is 
        5) if base size>0 and loaded size is not equal to the base size
           - return result from constants_default(ctype)
        6) reshape numpy array to the base shape and return.

        """

        tname = dic_calib_type_to_name[ctype]
        if self.pbits : print 'INFO %s: load constants of type %s' % (self.msgh(3), tname)

        fname = self.cff.findCalibFile(self.source, tname, self.runnum)

        if fname == '' :
            if self.pbits : print 'WARNING %s: calibration file for type %s is not found.' % (self.msgh(3), tname)
            self.dic_status[ctype] = NONFOUND
            return self.constants_default(ctype)

        if self.pbits : print self.msgw() % tname
        if self.pbits : print 'fname_name: %s' % fname

        nda = None 
        try :
            nda = np.loadtxt(fname, dtype=dic_calib_type_to_dtype[ctype])
        except :
            if self.pbits : print 'WARNING %s: calibration file for type %s is unreadable.' % (self.msgh(3), tname)
            self.dic_status[ctype] = UNREADABLE
            return self.constants_default(ctype)

        if ctype == COMMON_MODE :
            self.dic_status[ctype] = LOADED
            return nda

        if self.cbase.size == 0 :
            self.dic_status[ctype] = LOADED
            return nda

        if self.cbase.size>0 and nda.size != self.cbase.size :
            self.dic_status[ctype] = WRONGSIZE
            return self.constants_default(ctype)

        nda.shape = self.cbase.shape
        self.dic_status[ctype] = LOADED
        return nda

#------------------------------

    def pedestals(self) :
        """ Returns pedestals
        """
        return self.constants(PEDESTALS)

#------------------------------

    def pixel_status(self) :
        """ Returns pixel_status
        """
        return self.constants(PIXEL_STATUS)

#------------------------------

    def pixel_rms(self) :
        """ Returns pixel_rms
        """
        return self.constants(PIXEL_RMS)

#------------------------------

    def pixel_gain(self) :
        """ Returns pixel_gain
        """
        return self.constants(PIXEL_GAIN)

#------------------------------

    def pixel_mask(self) :
        """ Returns pixel_mask
        """
        return self.constants(PIXEL_MASK)

#------------------------------

    def pixel_bkgd(self) :
        """ Returns pixel_bkgd
        """
        return self.constants(PIXEL_BKGD)

#------------------------------

    def common_mode(self) :
        """ Returns common_mode
        """
        return self.constants(COMMON_MODE)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

    def ndim(self, ctype=PEDESTALS) :
        """ Returns ndim
        """
        if self.pbits & 128 : print self.msgw() % 'ndim(ctype)'
        if ctype == COMMON_MODE : return 1
        else                    : return self.cbase.ndim

#------------------------------

    def shape(self, ctype=PEDESTALS) :
        """ Returns shape
        """
        if self.pbits & 128 : print self.msgw() % 'shape(ctype)'
        if ctype == COMMON_MODE : return self.cbase.shape_cm
        else                    : return self.cbase.shape

#------------------------------

    def size(self, ctype=PEDESTALS) :
        """ Returns size
        """
        if self.pbits & 128 : print self.msgw() % 'size(ctype)'
        if ctype == COMMON_MODE : return self.cbase.size_cm
        else                    : return self.cbase.size

#------------------------------

    def status(self, ctype=PEDESTALS) :
        """ Returns status
        """
        if self.pbits & 128 : print self.msgw() % 'status(status)'
        return self.dic_status[ctype]

#------------------------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    sys.exit ('Test of %s is not implemented.' % sys.argv[0])

#------------------------------
#------------------------------
#------------------------------
#------------------------------


