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
    unregister = formencode.validators.Bool(if_empty=False)

class _ReplicateForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    if_key_missing = None
    src_resource = formencode.validators.String(if_empty=None)
    dst_resource = formencode.validators.String(not_empty=True)

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
                    replica_url = h.url_for(action='replica_show', path=name, replica_num=r['replica'], qualified=True)
                    r['replica_url'] = replica_url
                    
            return res

    @h.catch_all
    @jsonify
    def replica_show(self, path, replica_num):
        """
        GET /replica/*path/{replica_num}: Show full info for specific replica of a file.
        Takes path of the object and replica number.
        """
                
        # see if user can have an access
        h.checkAccess(path)

        model = IrodsModel()
        res = model.fileInfo('/'+path)

        if res is None :
            abort(404, "Object does not exist: "+str(path))
        else :
            
            replica = int(replica_num)
            res = [r for r in res if r.get('replica') == replica]
            if not res: abort(404, "Replica does not exist: "+str(path)+":"+str(replica))
            
            # add URI for each file or collection
            r = res[0]
            name = r['collName']+'/'+r['name']
            name = name.lstrip('/')
            url = h.url_for(action='show', path=name, replica_num=None, qualified=True)
            r['url'] = url
            replica_url = h.url_for(action='replica_show', path=name, replica_num=replica_num, qualified=True)
            r['replica_url'] = replica_url
                    
            return r

    @h.catch_all
    @jsonify
    def replica_delete(self, path, replica_num):
        """
        DELETE /replica/*path/{replica_num}: Remove specified object from irods.
        Takes path of the object (not collection) and replica number.
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
            abort(400, str(error))
        unregister = form_result['unregister']

        # replica number, integer
        replica = int(replica_num)

        model = IrodsModel()
        model.removeObj('/'+path, replica, unregister)

        return dict(path='/'+path, replica=replica)

    @h.catch_all
    @jsonify
    def replicate(self, path):
        """
        POST /files/*path: Make new replicate of specified object.
        Takes path of the object (not collection).
        Currently requires parameter "dst_resource" which must be a name of 
        destination resource.
        Optionally accepts parameter "src_resource" which is a name of the 
        source resource.
        """

        # see if user can have an access
        h.checkAccess(path)

        # validate parameters
        schema = _ReplicateForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort(400, str(error))

        # replica number must be non-negative
        src_resource = form_result['src_resource']
        dst_resource = form_result['dst_resource']

        model = IrodsModel()
        model.replicate('/'+path, dst_resource, src_resource)

        return dict(path='/'+path, dst_resource=dst_resource, src_resource=src_resource)

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
