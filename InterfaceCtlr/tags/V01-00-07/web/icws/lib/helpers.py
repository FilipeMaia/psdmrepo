"""Helper functions

Consists of functions to typically be used within templates, but also
available to Controllers. This module is available to templates as 'h'.
"""
# Import helpers as desired, or define your own, ie:
#from webhelpers.html.tags import checkbox, password
from pylons import request
from pylons import config
from pylons.controllers.util import abort
from routes import url_for

from WSClient.WSApp import WSApp
from WSClient.WSResource import WSResource


def checkAccess (instrument, experiment, mode) :
    """ Checks that user is authorized to access particular resource """
    
    # must have user name here
    if 'REMOTE_USER' not in request.environ :
        abort(401)
    user = request.environ['REMOTE_USER']

    # get parameters from config
    app_conf = config['app_conf']
    base_uri = app_conf.get( 'roledb.base_uri', '' )
    ws_app = app_conf.get( 'roledb.ws_app', 'roledb' )
    appname = app_conf.get( 'roledb.app_name', 'icws' )

    # See what user wants to see, users can access following areas:
    
    app = WSApp( ws_app, base_uri=base_uri )

    if instrument and experiment :
        # check access right for specific experiment        
        res = "/userroles/%s.%s-%s/%s/privileges" % ( appname, instrument, experiment, user )
        res = WSResource( app, res )
        try :
            data = res.get()
            if data and mode in data : return True
        except :
            pass

    if instrument :
        # check access right for specific experiment        
        res = "/userroles/%s.%s/%s/privileges" % ( appname, instrument, user )
        res = WSResource( app, res )
        try :
            data = res.get()
            if data and mode in data : return True
        except :
            pass

    # authorization with the specific experiment name may fail because
    # this experiment may not exist, in this case retry with the global
    # access rights
    
    res = "/userroles/%s/%s/privileges" % ( appname, user )
    res = WSResource( app, res )
    try :
        data = res.get()
        if data and mode in data : return True
    except :
        pass

    # nothing works, just fail
    abort( 401 )

