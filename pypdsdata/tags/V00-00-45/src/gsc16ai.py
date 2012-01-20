#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module gsc16ai...
#
#------------------------------------------------------------------------

"""Wrapper module for _pdsdata.gsc16ai.

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

from _pdsdata.gsc16ai import *

class DataV1(object):
    """Python class wrapping C++ Pds::Gsc16ai::DataV1 class"""
    
    def __init__(self, data, cfg):
        self.__data = data
        self.__cfg = cfg
    
    def timestamps(self):
        return self.__data.timestamps()

    def channelValues(self):
        return self.__data.channelValues(self.__cfg)
