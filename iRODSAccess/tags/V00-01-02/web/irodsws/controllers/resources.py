#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module resources
#------------------------------------------------------------------------

""" Pylons controller for the iRODS resources.

This software was developed for the LUSI project.  If you
use all or part of it, please give an appropriate acknowledgement.

Copyright (C) 2006 SLAC

@version $Id$ 

@author Andy Salnikov
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------
from irodsws.lib.base import BaseController

#-----------------------------
# Imports for other modules --
#-----------------------------
from pylons import request, response, session, tmpl_context as c
from pylons.controllers.util import abort, redirect_to
from pylons.decorators import validate
from pylons.decorators import jsonify
from irodsws.lib.base import *
from irodsws.model.irods_model import IrodsModel

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_log = logging.getLogger(__name__)

class ResourcesController(BaseController):
    """REST Controller styled on the Atom Publishing Protocol"""
    # To properly map this controller, ensure your config/routing.py
    # file has a resource setup:
    #     map.resource('resource', 'resources')

    @h.catch_all
    @jsonify
    def index(self):
        """GET /resources: All items in the collection"""
        
        model = IrodsModel()
        res = model.resources()
        res = [ dict( id=x, url=h.url_for( action='show', id=x, qualified=True ) ) for x in res ]
        return res

    @h.catch_all
    @jsonify
    def show(self, id):
        """GET /resources/id: Show a specific item"""
        # url('resource', id=ID)

        model = IrodsModel()
        res = model.resource(id)
            
        if not res :
            abort(404, "Unknown resource: "+str(id))
        elif len(res) == 1 :
            return res[0]
        else :
            return res
