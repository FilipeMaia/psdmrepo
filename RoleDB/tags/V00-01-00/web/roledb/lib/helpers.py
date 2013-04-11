"""Helper functions

Consists of functions to typically be used within templates, but also
available to Controllers. This module is available to templates as 'h'.
"""
# Import helpers as desired, or define your own, ie:
from routes import url_for
from webhelpers import *
from webhelpers.html.tags import *
from webhelpers.html import escape
from pylons.controllers.util import abort
from pylons import request

from roledb.model.roles_model import *

def checkAccess (mode) :
    """ Checks that user is authorized """
    
    if 'REMOTE_USER' not in request.environ :
        abort(401)
    user = request.environ['REMOTE_USER']
    model = RolesModel()
    privs = model.getUserPrivileges( 'RoleDB', None, user )
    if mode not in privs :
        abort( 401 )
    return
