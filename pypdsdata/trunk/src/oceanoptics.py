#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module oceanoptics...
#
#------------------------------------------------------------------------

"""Wrapper module for :py:mod:`_pdsdata.oceanoptics`. This module imports all 
:py:mod:`_pdsdata.oceanoptics` classes without any change. 

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

from _pdsdata.oceanoptics import *
from _pdsdata.oceanoptics import DataV1 as _DataV1
from _pdsdata.oceanoptics import DataV2 as _DataV2


def _fwd(method, type):
    unbm = getattr(type, method)
    def method(self):
        return unbm(self.__data)
    method.__doc__ = unbm.__doc__
    method.__name__ = unbm.__name__
    return method

def _fwd_cfg(method, type, doc):
    unbm = getattr(type, method)
    def method(self):
        return unbm(self.__data, self.__cfg)
    method.__doc__ = doc
    method.__name__ = unbm.__name__
    return method

class DataV1(object):
    """
    This is a wrapper for :py:class:`_pdsdata.oceanoptics.DataV1` which removes the need to pass 
    configuration objects to some methods.
    """
    
    iDataReadSize = _DataV1.iDataReadSize
    iNumPixels = _DataV1.iNumPixels
    iActivePixelIndex = _DataV1.iActivePixelIndex
    
    def __init__(self, data, cfg):
        """ Constructor takes one :py:class:`_pdsdata.oceanoptics.DataV1` and one 
        :py:class:`_pdsdata.oceanoptics.ConfigVx` object """
        self.__data = data
        self.__cfg = cfg
    
    frameCounter = _fwd('frameCounter', _DataV1)
    numDelayedFrames = _fwd('numDelayedFrames', _DataV1)
    numDiscardFrames = _fwd('numDiscardFrames', _DataV1)
    numSpectraInData = _fwd('numSpectraInData', _DataV1)
    numSpectraInQueue = _fwd('numSpectraInQueue', _DataV1)
    numSpectraUnused = _fwd('numSpectraUnused', _DataV1)
    durationOfFrame = _fwd('durationOfFrame', _DataV1)
    data = _fwd('data', _DataV1)
    timeFrameStart = _fwd('timeFrameStart', _DataV1)
    timeFrameFirstData = _fwd('timeFrameFirstData', _DataV1)
    timeFrameEnd = _fwd('timeFrameEnd', _DataV1)
    nonlinerCorrected = _fwd_cfg('nonlinerCorrected', _DataV1,
        "self.nonlinerCorrected() -> ndarray\n\nReturns 1-dim ndarray of floats, which is data corrected for non-linearity, size of array is `iNumPixels`")
    
class DataV2(object):
    """
    This is a wrapper for :py:class:`_pdsdata.oceanoptics.DataV2` which removes the need to pass 
    configuration objects to some methods.
    """

    iDataReadSize = _DataV2.iDataReadSize
    iNumPixels = _DataV2.iNumPixels
    iActivePixelIndex = _DataV2.iActivePixelIndex
    
    def __init__(self, data, cfg):
        """ Constructor takes one :py:class:`_pdsdata.oceanoptics.DataV2` and one 
        :py:class:`_pdsdata.oceanoptics.ConfigVx` object """
        self.__data = data
        self.__cfg = cfg
    
    frameCounter = _fwd('frameCounter', _DataV2)
    numDelayedFrames = _fwd('numDelayedFrames', _DataV2)
    numDiscardFrames = _fwd('numDiscardFrames', _DataV2)
    numSpectraInData = _fwd('numSpectraInData', _DataV2)
    numSpectraInQueue = _fwd('numSpectraInQueue', _DataV2)
    numSpectraUnused = _fwd('numSpectraUnused', _DataV2)
    durationOfFrame = _fwd('durationOfFrame', _DataV2)
    data = _fwd('data', _DataV2)
    timeFrameStart = _fwd('timeFrameStart', _DataV2)
    timeFrameFirstData = _fwd('timeFrameFirstData', _DataV2)
    timeFrameEnd = _fwd('timeFrameEnd', _DataV2)
    nonlinerCorrected = _fwd_cfg('nonlinerCorrected', _DataV2,
        "self.nonlinerCorrected() -> ndarray\n\nReturns 1-dim ndarray of floats, which is data corrected for non-linearity, size of array is `iNumPixels`")
    
