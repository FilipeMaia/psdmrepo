#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module LusiPyApp
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

class _NewUserRoleForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    if_key_missing = None
    exp_id = formencode.validators.Int()
    app = formencode.validators.String(not_empty=True)
    user = formencode.validators.String(not_empty=True)
    role = formencode.validators.String(not_empty=True)

_log = logging.getLogger(__name__)

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class UserrolesController(BaseController):
    """REST Controller styled on the Atom Publishing Protocol"""
    # To properly map this controller, ensure your config/routing.py
    # file has a resource setup:
    #     map.resource('userrole', 'userroles')

    @jsonify
    def index(self ):
        """GET /userroles: All items in the collection"""

        model = RolesModel()
        return model.userroles()


    def new(self):
        """GET /userroles/new: Form to create a new item"""
        
        # check user's privileges
        h.checkAccess('create')
        
        # display form
        return render('/newuserrole.html')
    

    @jsonify
    def create(self):
        """POST /userroles: Create a new item"""

        # check user's privileges
        h.checkAccess('create')

        # validate parameters
        schema = _NewUserRoleForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort( 400, unicode(error) )
        
        # empty exp_id ito become NULL
        if form_result['exp_id'] :
            exp_id = int(form_result['exp_id'])
        else :
            exp_id = None        
        app = form_result['app']
        user = form_result['user']
        role = form_result['role']

        model = RolesModel()
        
        # pass data to model
        id = model.addUserRole ( app, exp_id, user, role )
        if not id :
            abort(400, "UserrolesController.create: failed to create user role %s/%s/%s", (app, user, role) )

        # return complete record for result
        return form_result

    @jsonify
    def update(self, app, user, role, exp_id=None ):
        """PUT /userroles/{app}~{exp_id}/{user}/{role}: Create a new item"""
        
        # check user's privileges
        h.checkAccess('update')

        model = RolesModel()
        
        # pass data to model
        id = model.addUserRole ( app, exp_id, user, role )

        # return complete record for result
        res = {}
        res['app'] = app
        res['user'] = user
        res['role'] = role
        if exp_id : res['exp_id'] = exp_id
        return res


    @jsonify
    def delete(self, app, user, exp_id=None, role=None):
        """DELETE /userroles/{app}~{exp_id}/{user}/{role}: Delete an existing item"""

        # check user's privileges
        h.checkAccess('delete')

        model = RolesModel()
        if role :
            # delete one role
            model.deleteUserRole( app, exp_id, user, role )
        else :
            # delete all user roles for an application
            model.deleteUserRoles( app, exp_id, user )


    @jsonify
    def deleteUser(self, user ):
        """DELETE /userroles/{app}~{exp_id}/{user}/{role}: Delete an existing item"""

        # check user's privileges
        h.checkAccess('delete')

        # delete all user roles
        model = RolesModel()
        model.deleteUser( user )


    @jsonify
    def show(self, app, user, exp_id=None ):
        """GET /userroles/{app}~{exp_id}/{user}: Show a specific item"""

        if exp_id is not None : exp_id = int(exp_id)
        model = RolesModel()
        res = model.getUserRoles( app, exp_id, user )
        return res


    @jsonify
    def showPriv(self, app, user, exp_id=None ):
        """GET /userroles/{app}~{exp_id}/{user}/privileges: Show a specific item"""

        if exp_id is not None : exp_id = int(exp_id)
        model = RolesModel()
        res = model.getUserPrivileges( app, exp_id, user )
        return res
