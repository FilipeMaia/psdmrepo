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
import formencode

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

class _ShowForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    if_key_missing = None
    recursive = formencode.validators.Int()

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class FilesController(BaseController):

    @jsonify
    def show(self, path=''):
        """GET /resources/*path: Show a specific item"""
        # url('resource', id=ID)

        # see if user can have an access
        h.checkAccess(path)

        # validate parameters
        schema = _ShowForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort( 400, unicode(error) )

        # empty exp_id ito become NULL
        try:
            recursive = int(form_result['recursive'])
        except :
            recursive = None

        model = IrodsModel()
        res = model.files( '/'+path, recursive )
        if res is None :
            abort(404)
        else :
            return res
