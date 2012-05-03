#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module cspad...
#
#------------------------------------------------------------------------

"""Wrapper module for :py:mod:`_pdsdata.cspad`, provides wrapper for ElementV*
classes. All other classes are imported without change.

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

from _pdsdata.cspad import *
from _pdsdata.cspad import ElementV1 as _ElementV1
from _pdsdata.cspad import ElementV2 as _ElementV2

# extend FrameV1
class ElementV1(object) :
    """
    This is a wrapper for :py:class:`_pdsdata.cspad.ElementV1` which removes the need to pass 
    configuration objects to several methods.
    """

    # class constants imported without change
    ColumnsPerASIC = _ElementV1.ColumnsPerASIC
    MaxRowsPerASIC = _ElementV1.MaxRowsPerASIC
    

    def __init__ (self, quad, cfg):
        """ Constructor takes one :py:class:`_pdsdata.cspad.ElementV1` and one 
        :py:class:`_pdsdata.cspad.ConfigV1` object """

        self.__virtual_channel = quad.virtual_channel()
        self.__lane = quad.lane()
        self.__tid = quad.tid()
        self.__acq_count = quad.acq_count()
        self.__op_code = quad.op_code()
        self.__quad = quad.quad()
        self.__seq_count = quad.seq_count()
        self.__ticks = quad.ticks()
        self.__fiducials = quad.fiducials()
        self.__frame_type = quad.frame_type()
        self.__sb_temp = map(quad.sb_temp, range(4))
        self.__data = quad.data(cfg)

    def virtual_channel(self):
        """self.virtual_channel() -> int
        
        Returns integer number.
        """
        return self.__virtual_channel

    def lane(self):
        """self.lane() -> int
        
        Returns integer number.
        """
        return self.__lane

    def tid(self):
        """self.tid() -> int
        
        Returns integer number.
        """
        return self.__tid

    def acq_count(self):
        """self.acq_count() -> int
        
        Returns integer number.
        """
        return self.__acq_count

    def op_code(self):
        """self.op_code() -> int
        
        Returns integer number.
        """
        return self.__op_code

    def quad(self):
        """self.quad() -> int
        
        Returns quadrant number.
        """
        return self.__quad

    def seq_count(self):
        """self.seq_count() -> int
        
        Returns sequence counter.
        """
        return self.__seq_count

    def ticks(self):
        """self.ticks() -> int
        
        Returns integer number.
        """
        return self.__ticks

    def fiducials(self):
        """self.fiducials() -> int
        
        Returns integer number.
        """
        return self.__fiducials

    def frame_type(self):
        """self.frame_type() -> int
        
        Returns integer number.
        """
        return self.__frame_type

    def sb_temp(self, i):
        """self.sb_temp(i: int) -> int
        
        Retuns integer number, index i in the range (0..3).
        """
        return self.__sb_temp[i]

    def data(self):
        """self.data() -> numpy.ndarray
        
        Returns data array for this quadrant.
        """
        return self.__data


# extend FrameV1
class ElementV2(object) :
    """
    This is a wrapper for :py:class:`_pdsdata.cspad.ElementV2` which removes the need to pass 
    configuration objects to several methods.
    """

    # class constants imported without change
    ColumnsPerASIC = _ElementV2.ColumnsPerASIC
    MaxRowsPerASIC = _ElementV2.MaxRowsPerASIC
    

    def __init__ (self, quad, cfg):
        """ Constructor takes one :py:class:`_pdsdata.cspad.ElementV2` and one 
        :py:class:`_pdsdata.cspad.ConfigV2` object """

        self.__virtual_channel = quad.virtual_channel()
        self.__lane = quad.lane()
        self.__tid = quad.tid()
        self.__acq_count = quad.acq_count()
        self.__op_code = quad.op_code()
        self.__quad = quad.quad()
        self.__seq_count = quad.seq_count()
        self.__ticks = quad.ticks()
        self.__fiducials = quad.fiducials()
        self.__frame_type = quad.frame_type()
        self.__sb_temp = map(quad.sb_temp, range(4))
        self.__data = quad.data(cfg)

    def virtual_channel(self):
        """self.virtual_channel() -> int
        
        Returns integer number.
        """
        return self.__virtual_channel

    def lane(self):
        """self.lane() -> int
        
        Returns integer number.
        """
        return self.__lane

    def tid(self):
        """self.tid() -> int
        
        Returns integer number.
        """
        return self.__tid

    def acq_count(self):
        """self.acq_count() -> int
        
        Returns integer number.
        """
        return self.__acq_count

    def op_code(self):
        """self.op_code() -> int
        
        Returns integer number.
        """
        return self.__op_code

    def quad(self):
        """self.quad() -> int
        
        Returns quadrant number.
        """
        return self.__quad

    def seq_count(self):
        """self.seq_count() -> int
        
        Returns sequence counter.
        """
        return self.__seq_count

    def ticks(self):
        """self.ticks() -> int
        
        Returns integer number.
        """
        return self.__ticks

    def fiducials(self):
        """self.fiducials() -> int
        
        Returns integer number.
        """
        return self.__fiducials

    def frame_type(self):
        """self.frame_type() -> int
        
        Returns integer number.
        """
        return self.__frame_type

    def sb_temp(self, i):
        """self.sb_temp(i: int) -> int
        
        Retuns integer number, index i in the range (0..3).
        """
        return self.__sb_temp[i]

    def data(self):
        """self.data() -> numpy.ndarray
        
        Returns data array for this quadrant.
        """
        return self.__data


def wrapElement(quad, cfg):
    """Function that wraps low-level Element class with the correct wrapper class"""
    if quad.__class__ is _ElementV1 :
        return ElementV1(quad, cfg)
    else:
        return ElementV2(quad, cfg)
