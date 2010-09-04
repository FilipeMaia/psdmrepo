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

# extend FrameV1
class ElementV1(object) :

    # class constants imported without change
    ColumnsPerASIC = _ElementV1.ColumnsPerASIC
    ColumnsPerASIC = _ElementV1.MaxRowsPerASIC
    

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
        self.__sb_temp = map(quad.sb_temp(i), range(4))

        data = quad.data(cfg)
        shape = data.shape

        # make empty array of the same type
        self.__data = np.empty((shape[0]*2, shape[1], shape[2]), data[0].dtype)

        # copy over the data
        for i in range(shape[0]) :
            for j in range(2) :
                self.__data[i*2+j, ...] = data[i, ..., j]
                
        # change rows/columns
        np.swapaxes(self.__data, 1, 2)

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

    def pixel(self, asic, row, col):
        return self.__data[asic, row, col]


