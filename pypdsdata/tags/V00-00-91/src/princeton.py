#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module princeton...
#
#------------------------------------------------------------------------

"""Wrapper module for :py:mod:`_pdsdata.princeton`, provides wrapper for 
:py:class:`_pdsdata.princeton.FrameV1` class. All other classes are imported without change.

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

from _pdsdata.princeton import *

class FrameV1(object):
    """
    This is a wrapper for :py:class:`_pdsdata.princeton.FrameV1` which removes the need to pass 
    configuration objects to several methods.
    """
    
    def __init__(self, frame, cfg):
        """ Constructor takes list of :py:class:`_pdsdata.princeton.FrameV1` and 
        one `_pdsdata.princeton.ConfigV1` object """
        self.__frame = frame
        self.__cfg = cfg
    
    def shotIdStart(self):
        """self.shotIdStart() -> int
        
        Returns integer number
        """
        return self.__frame.shotIdStart()

    def readoutTime(self):
        """self.shotIdStart() -> float
        
        Returns floating number
        """
        return self.__frame.readoutTime()
    
    def data(self):
        """self.data() -> numpy.ndarray

        Returns 2-dim array of integer numbers
        """
        return self.__frame.data(self.__cfg)
