#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module system...
#
#------------------------------------------------------------------------

"""Pylons controller for the Inteface Controlles system-level resource.

This software was developed for the SIT project.  If you use all or 
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
import sys

#---------------------------------
#  Imports of base class module --
#---------------------------------
from icws.lib.base import BaseController

#-----------------------------
# Imports for other modules --
#-----------------------------
from pylons.decorators import jsonify
from icws.lib.base import *
from icws.model.icdb_model import IcdbModel

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class SystemController ( BaseController ) :

    #-------------------
    #  Public methods --
    #-------------------

    @jsonify
    def status ( self, id = None ) :

        model = IcdbModel()
        return model.controller_status(id)

    @jsonify
    def stop ( self, id ) :

        model = IcdbModel()
        return model.controller_stop(id)

