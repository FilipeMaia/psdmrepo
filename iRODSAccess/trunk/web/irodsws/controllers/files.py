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
    recursive = formencode.validators.Int(if_empty=0)

class _RemoveForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    if_key_missing = None
    replica = formencode.validators.Int(not_empty=True)
    unregister = formencode.validators.Bool(if_empty=False)

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class FilesController(BaseController):

    @h.catch_all
    @jsonify
    def show(self, path=''):
        """
        GET /files/*path: Show full info for files and collections.
        Takes path of the object or collection.
        Additionally accepts parameter "recursive", if non-zero then
        show recursively content of the nested collections.
        """
        
        # see if user can have an access
        h.checkAccess(path)

        # validate parameters
        schema = _ShowForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort( 400, str(error) )

        recursive = form_result['recursive']

        model = IrodsModel()
        res = model.files( '/'+path, recursive )

        if res is None :
            abort(404, "Collection does not exist: "+str(path))
        else :
            
            # add URI for each file or collection
            for r in res :
                if r['type'] == 'collection' :
                    name = r['name'].lstrip('/')
                    url = h.url_for( action='show', path=name, qualified=True )
                    r['url'] = url
                elif r['type'] == 'object' :
                    name = r['collName']+'/'+r['name']
                    name = name.lstrip('/')
                    url = h.url_for( action='show', path=name, qualified=True )
                    r['url'] = url
                    
            return res

    @h.catch_all
    @jsonify
    def remove(self, path):
        """
        DELETE /files/*path: Remove specified object from irods.
        Takes path of the object (not collection).
        Currently requires parameter "replica" which must be non-negative number.
        Optionally accepts parameter "unregister", if present then files will
        not be removed from resources, only unregistered.
        """

        # see if user can have an access
        h.checkAccess(path)

        # validate parameters
        schema = _RemoveForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort( 400, str(error) )

        # replica number must be non-negative
        replica = form_result['replica']
        if replica < 0: abort( 400, "Replica number must be non-negative" )

        unregister = form_result['unregister']

        model = IrodsModel()
        model.removeObj('/'+path, replica, unregister)

        return []

    def environ(self):
        """
        Dump all variables in the environment, useful for tests only.
        """
        
        result = '<html><body><h1>Environ</h1><table>'
        keys = request.environ.keys()
        keys.sort()
        for key in keys :
            result += '<tr><td>%s</td><td>%r</td></tr>'%(key, request.environ[key])
        result += '</table></body></html>'
        return result
