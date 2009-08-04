#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module roles
#------------------------------------------------------------------------

""" Pylons controller for the Roles database.

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
from roledb.lib.base import BaseController
import formencode

#-----------------------------
# Imports for other modules --
#-----------------------------
from pylons.decorators import validate
from pylons.decorators import jsonify
from roledb.lib.base import *
from roledb.model.roles_model import RolesModel

#----------------------------------
# Local non-exported definitions --
#----------------------------------

class _CreateNewRoleForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    if_key_missing = ""
    app = formencode.validators.String(not_empty=True)
    role = formencode.validators.String(not_empty=True)
    privileges = formencode.validators.String()

class _UpdateNewRoleForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    if_key_missing = ""
    privileges = formencode.validators.String()


_log = logging.getLogger(__name__)

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class RolesController(BaseController):

        
    @jsonify
    def index(self):
        """return all roles defined in database """
        
        model = RolesModel()
        res = model.roles ()
        
        return res


    def new (self):

        # check user's privileges
        h.checkAccess('create')
        
        # display form
        return render('/newrole.html')

    @jsonify
    def create(self):
        """ create new record in Roles database """

        # check user's privileges
        h.checkAccess('create')

        # validate parameters
        schema = _CreateNewRoleForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort(400,str(error))
        
        # empty exp_id ito become NULL
        app = form_result['app']
        role = form_result['role']
        privs = form_result['privileges']
        privs = privs.split(',')
        privs = [ p.strip() for p in privs ]

        model = RolesModel()

        # check that role does not exist yet
        rid = model.findRole ( app, role )
        if rid is not None :
            abort(400,"RolesController.create: role %s/%s already exists" % (app, role) )
        
        # create it
        rid = model.addRole ( app, role )
        if rid is None :
            abort(500,"RolesController.create: failed to add new role %s/%s" % (app, role) )
            
        # add all privileges for this role
        for p in privs :
            pid = model.addPrivilege ( app, role, p )
            if pid is None :
                abort(500,"RolesController.create: failed to add new privilege %s/%s/%s" % (app, role,p) )

        # return complete record for result
        return dict( app=app, role=role, privileges=privs )


    @jsonify
    def update(self, app, role):
        """ create new record in Roles database """

        # check user's privileges
        h.checkAccess('update')

        # validate parameters
        schema = _UpdateNewRoleForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort(400,str(error))
        
        # make the list of privileges
        privs = form_result['privileges']
        privs = [ p.strip() for p in privs.split(',') ]
        privs = [ p for p in privs if p ]

        model = RolesModel()

        # find or create the role
        rid = model.findRole ( app, role )
        if rid is None :
            rid = model.addRole ( app, role )
        if rid is None :
            abort(500,"RolesController.update: failed to add new role %s/%s" % (app, role) )

        # add all privileges for this role
        for p in privs :
            pid = model.addPrivilege ( app, role, p )
            if pid is None :
                abort(400,"RolesController.create: failed to add new privilege %s/%s/%s" % (app, role, p) )

        # return complete record for result
        return dict( app=app, role=role, privileges=privs )


    @jsonify
    def delete(self, app, role):
        """ delete role from Roles database """

        # check user's privileges
        h.checkAccess('delete')

        model = RolesModel()
        model.deleteRole( app, role )

        # returns empty reply
