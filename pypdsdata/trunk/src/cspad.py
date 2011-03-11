#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module cspad...
#
#------------------------------------------------------------------------

"""Wrapper module for _pdsdata.cspad.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

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

    # class constants imported without change
    ColumnsPerASIC = _ElementV1.ColumnsPerASIC
    MaxRowsPerASIC = _ElementV1.MaxRowsPerASIC
    

    def __init__ (self, quad, cfg):
        """ Constructor takes one _pdsdata.cspad.ElementV1 and one 
        _pdsdata.cspad.ConfigV1 object """

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
        return self.__virtual_channel

    def lane(self):
        return self.__lane

    def tid(self):
        return self.__tid

    def acq_count(self):
        return self.__acq_count

    def op_code(self):
        return self.__op_code

    def quad(self):
        return self.__quad

    def seq_count(self):
        return self.__seq_count

    def ticks(self):
        return self.__ticks

    def fiducials(self):
        return self.__fiducials

    def frame_type(self):
        return self.__frame_type

    def sb_temp(self, i):
        return self.__sb_temp[i]

    def data(self):
        return self.__data


# extend FrameV1
class ElementV2(object) :

    # class constants imported without change
    ColumnsPerASIC = _ElementV2.ColumnsPerASIC
    MaxRowsPerASIC = _ElementV2.MaxRowsPerASIC
    

    def __init__ (self, quad, cfg):
        """ Constructor takes one _pdsdata.cspad.ElementV2 and one 
        _pdsdata.cspad.ConfigV2 object """

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
        return self.__virtual_channel

    def lane(self):
        return self.__lane

    def tid(self):
        return self.__tid

    def acq_count(self):
        return self.__acq_count

    def op_code(self):
        return self.__op_code

    def quad(self):
        return self.__quad

    def seq_count(self):
        return self.__seq_count

    def ticks(self):
        return self.__ticks

    def fiducials(self):
        return self.__fiducials

    def frame_type(self):
        return self.__frame_type

    def sb_temp(self, i):
        return self.__sb_temp[i]

    def data(self):
        return self.__data


def wrapElement(quad, cfg):
    """Function that wraps low-level Element class with the correct wrapper class"""
    if quad.__class__ is _ElementV1 :
        return ElementV1(quad, cfg)
    else:
        return ElementV2(quad, cfg)
