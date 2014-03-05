#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module gsc16ai...
#
#------------------------------------------------------------------------

"""Wrapper module for :py:mod:`_pdsdata.gsc16ai`, provides wrapper for 
:py:class:`_pdsdata.gsc16ai.DataV1` class. All other classes are imported without change.

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

from _pdsdata.gsc16ai import *

class DataV1(object):
    """
    This is a wrapper for :py:class:`_pdsdata.gsc16ai.DataV1` which removes the need to pass 
    configuration objects to several methods.
    """
    
    def __init__(self, data, cfg):
        """ Constructor takes one :py:class:`_pdsdata.gsc16ai.DataV1` and one 
        :py:class:`_pdsdata.gsc16ai.ConfigV1` object """
        self.__data = data
        self.__cfg = cfg
    
    def timestamps(self):
        """self.timestamps() -> numpy.ndarray
        
        Returns array of integers.
        """
        return self.__data.timestamps()

    def channelValues(self):
        """self.channelValues() -> numpy.ndarray
        
        Returns array of integers.
        """
        return self.__data.channelValues(self.__cfg)
