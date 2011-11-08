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
class ImgConfigParameters :
    """This class contains all configuration parameters
    """

    def __init__ ( self ) :
        self.setRunTimeParametersInit()
        self.setDefaultParameters()
        print 'ImgConfigParameters initialization'


    def setRunTimeParametersInit ( self ) :
        self.ImgGUIIsOpen         = False
        #self.nWindows            = 3
        self.widg_img             = None # widget of the image for access

    def setDefaultParameters ( self ) :
        """Set default configuration parameters hardwired in this module"""
        #print 'setDefaultParameters'
        pass

#---------------------------------------
# Makes a single object of this class --
#---------------------------------------

imgconfpars = ImgConfigParameters ()

#---------------------------------------
