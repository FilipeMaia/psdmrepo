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

        # check user's privileges
        h.checkAccess('', '', 'read')

        model = IcdbModel()
        res = model.controller_status(id)
        for r in res :
            if 'log' in r and r['log'] :
                try :
                    log = r['log'].split('/')
                    path = 'system/'+log[-2]+'/'+log[-1]
                    r['log_url'] = h.url_for(controller='/log', action='show', mode='html', path=path)
                except :
                    pass
        if id is None:
            return res
        elif res :
            return res[0]
        else :
            return {}

    @jsonify
    def stop ( self, id ) :

        # check user's privileges
        h.checkAccess('', '', 'delete')

        model = IcdbModel()
        return model.controller_stop(id)

