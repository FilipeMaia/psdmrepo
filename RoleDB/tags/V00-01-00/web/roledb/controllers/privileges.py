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

class _CreateNewPrivsForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    privileges = formencode.validators.String(not_empty=True)


_log = logging.getLogger(__name__)

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class PrivilegesController(BaseController):
    """REST Controller styled on the Atom Publishing Protocol"""
    # To properly map this controller, ensure your config/routing.py
    # file has a resource setup:
    #     map.resource('privilege', 'privileges')



    @jsonify
    def index(self, app, role ):
        """GET /privileges/{app}/{role}: All items in the collection"""
        
        model = RolesModel()
        res = model.privileges ( app, role )
        
        return res

    @jsonify
    def create(self, app, role):
        """Create new privileges"""
        
        # check user's privileges
        h.checkAccess('create')

        # validate parameters
        schema = _CreateNewPrivsForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort(400,str(error))
        
        # empty exp_id ito become NULL
        privs = form_result['privileges']
        privs = privs.split(',')
        privs = [ p.strip() for p in privs ]

        model = RolesModel()
        
        # find or create the role
        rid = model.findRole ( app, role )
        if not rid :
            abort(400, "PrivilegesController.create: role %s/%s does not exist" % (app,role) )

        # add all privileges for this role
        for p in privs :
            
            pid = model.addPrivilege ( app, role, p )
            if pid is None :
                abort(400, "PrivilegesController.create: failed to create privilege %s/%s/%s, it may exist already" 
                      % (app, role, p) )

        # return complete record for result
        return dict( app=app, role=role, privileges=privs )


    def new(self, app, role):
        """GET /privileges/{app}/{role}/new: Form to create a new item"""
        
        # check user's privileges
        h.checkAccess('create')
        
        # display form
        return render('/newprivilege.html')


    @jsonify
    def update(self, app, role, privilege):
        """ Create new privilege """

        # check user's privileges
        h.checkAccess('update')
        
        model = RolesModel()
        
        # role must exist already
        rid = model.findRole ( app, role )
        if rid is None :
            abort(400, "PrivilegesController.update: role %s/%s does not exist" % (app,role) )

        # add all privileges for this role
        pid = model.addPrivilege ( app, role, privilege )
        if pid is None :
            abort(400, "PrivilegesController.update: failed to create privilege %s/%s/%s, it may exist already" 
                  % (app, role, p) )
            
        # return complete record for result
        return dict( app=app, role=role, privileges=[privilege] )


    @jsonify
    def delete(self, app, role, privilege):
        """ Delete an existing item"""
        
        # check user's privileges
        h.checkAccess('delete')

        model = RolesModel()
        model.deletePrivilege( app, role, privilege )

        # returns empty reply
