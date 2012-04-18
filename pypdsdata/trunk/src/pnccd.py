#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module pnccd...
#
#------------------------------------------------------------------------

"""Wrapper module for :py:mod:`_pdsdata.pnccd`, provides wrapper for 
:py:class:`_pdsdata.pnccd.FrameV1` class. All other classes are imported without change.

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
import numpy as np

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

from _pdsdata.pnccd import *

# extend FrameV1
class FrameV1(object) :
    """
    This is a wrapper for :py:class:`_pdsdata.pnccd.FrameV1` which makes a larger frame
    image out of four smaller frames. 
    """

    def __init__ (self, frames, cfg):
        """ Constructor takes list of :py:class:`_pdsdata.pnccd.FrameV1` and 
        one `_pdsdata.acqiris.ConfigV1` object """

        # at this moment can only work with 4 frames
        if len(frames) != 4 : 
            raise _pdsdata.Error("pnccd.FrameV1: Odd number of frames: %d" % len(frames))

        # these numbers area supposed to be the same for all frames
        self.__specialWord = frames[0].specialWord()
        self.__frameNumber = frames[0].frameNumber()

        # get the lowest timestamp value
        ts = [ (f.timeStampHi(),f.timeStampLo()) for f in frames ]
        ts.sort()
        self.__timeStampHi = ts[0][0]
        self.__timeStampLo = ts[0][1]
        
        # build large image
        data = [f.data(cfg) for f in frames]
        size = [ d.shape for d in data ]
        
        # we expect all sizes to be the same
        if len(set(size)) != 1 :
            raise _pdsdata.Error("pnccd.FrameV1: non-equal images sizes: %s" % size)
        ny, nx = size[0]

        # make empty array of the same type
        self.__data = np.empty((ny*2, nx*2), data[0].dtype)
        
        # copy the data over, change orientation for some
        self.__data[0:ny, 0:nx] = data[0]
        self.__data[0:ny, nx:2*nx] = data[3]
        self.__data[ny:2*ny, 0:nx] = data[1][::-1,::-1]
        self.__data[ny:2*ny, nx:2*nx] = data[2][::-1,::-1]
        
    def specialWord(self) :
        """self.specialWord() -> int
        
        Returns integer number.
        """
        return self.__specialWord
    
    def frameNumber(self) :
        """self.frameNumber() -> int
        
        Returns integer number.
        """
        return self.__frameNumber
    
    def timeStampHi(self) :
        """self.timeStampHi() -> int
        
        Returns integer number.
        """
        return self.__timeStampHi
    
    def timeStampLo(self) :
        """self.timeStampLo() -> int
        
        Returns integer number.
        """
        return self.__timeStampLo
    
    def data(self) :
        """self.data() -> numpy.ndarray
        
        Returns frame data as NumPy 2-dimensional array of integers of size 1024x1024
        """
        return self.__data
    
    def sizeofData(self) :
        """self.sizeofData() -> int
        
        Returns size of data in a frame (in pixels)
        """
        return self.__data.size

    