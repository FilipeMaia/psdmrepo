#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigParameters...
#
#------------------------------------------------------------------------

"""Configuration parameters for Img.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id:$

@author Mikhail S. Dubrovin
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

#---------------------
#  Class definition --
#---------------------
class ConfigParameters :
    """This class contains all configuration parameters
    """

    def __init__ ( self ) :
        """Constructor"""
        self.setRunTimeParametersInit()
        self.setDefaultParameters()


    def setRunTimeParametersInit ( self ) :
        self.ImgGUIIsOpen         = False
        self.nWindows             = 3

    def setDefaultParameters ( self ) :
        """Set default configuration parameters hardwired in this module"""
        print 'setDefaultParameters'

#---------------------------------------
# Makes a single object of this class --
#---------------------------------------

imgconfpars = ConfigParameters ()

#---------------------------------------
