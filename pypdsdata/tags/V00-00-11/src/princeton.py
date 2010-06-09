#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module princeton...
#
#------------------------------------------------------------------------

"""Wrapper module for _pdsdata.princeton.

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

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

from _pdsdata.princeton import *

class FrameV1(object):
    """Python class wrapping C++ Pds::Princeton::FrameV1 class"""
    
    def __init__(self, frame, cfg):
        self.__frame = frame
        self.__cfg = cfg
    
    def shotIdStart(self):
        return self.__frame.shotIdStart()

    def readoutTime(self):
        return self.__frame.readoutTime()
    
    def data(self):
        return self.__frame.data(self.__cfg)
