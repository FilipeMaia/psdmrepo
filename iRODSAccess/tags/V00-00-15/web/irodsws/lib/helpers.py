"""Helper functions

Consists of functions to typically be used within templates, but also
available to Controllers. This module is available to templates as 'h'.
"""
# Import helpers as desired, or define your own, ie:
#from webhelpers.html.tags import checkbox, password
from pylons import request, response
from pylons import config
from pylons.controllers.util import abort
from routes import url_for

from WSClient.WSApp import WSApp
from WSClient.WSResource import WSResource

from pylons.decorators import decorator
from iRODSAccess.Exceptions import *
from webob.exc import HTTPException

@decorator
def catch_all(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except ConnectionError, ex:
        response.status = '503 Service Unavailable'
        response.content_type = 'text/plain'
        return str(ex)
    except MissingError, ex:
        response.status = '404 Not Found'
        response.content_type = 'text/plain'
        return str(ex)
    except HTTPException, ex:
        response.status = '%d %s' % (ex.code, ex.title)
        response.content_type = 'text/plain'
        return ex.detail
    except Exception, ex:
        response.status = '500 Internal Server Error'
        response.content_type = 'text/plain'
        return str(ex)

def checkAccess (path) :
    """ Checks that user is authorized to see particular path """
    
    # must have user name here
    if 'REMOTE_USER' not in request.environ :
        abort( 401, "User name missing, permissions not granted")
    user = request.environ['REMOTE_USER']

    # get parameters from config
    app_conf = config['app_conf']
    base_uri = app_conf.get( 'roledb.base_uri', '' )
    ws_app = app_conf.get( 'roledb.ws_app', 'roledb' )
    appname = app_conf.get( 'roledb.app_name', 'irodsws' )

    # path is /zone/a/b/c, split it into single elements
    ph = [ x for x in path.split('/') if x ]
    
    # See what user wants to see, users can access following areas:
    #  /zone/exp/{Instrument}/{ExpName} if have "read" privilege for experiment
    #  /zone/home/{user} - no special privileges needed
    #  /zone/....  - must have global read privilege
    
    if len(ph) >= 3 and ph[1] == 'home' and ph[2] == user :
        return True

    app = WSApp( ws_app, base_uri=base_uri )
    if len(ph) >= 4 and ph[1] == 'exp' :
        
        # check access right for specific experiment
        
        exp = ph[2]+'-'+ph[3]
        res = "/userroles/%s.%s/%s/privileges" % ( appname, exp, user )
        res = WSResource( app, res )    
        try :
            data = res.get()
            if data and 'read' in data : return True
        except :
            pass

    # authorization with the specific experiment name may fail because
    # this experiment may not exist, in this case retry with the generic
    # access rights
    
    res = "/userroles/%s/%s/privileges" % ( appname, user )
    res = WSResource( app, res )    
    try :
        data = res.get()
        if data and 'read' in data : return True
    except :
        pass

    # nothing works, just fail
    abort( 401, "User '%s' lacks permissions to access path %s:%s" % (user, appname, path) )
    
    
def server_uri() :
    """ return the URI of the server, code adapted from PEP333 """
    
    environ = request.environ
    
    url = environ['wsgi.url_scheme']+'://'

    if environ.get('HTTP_HOST'):
        url += environ['HTTP_HOST']
    else:
        url += environ['SERVER_NAME']
    
        if environ['wsgi.url_scheme'] == 'https':
            if environ['SERVER_PORT'] != '443':
               url += ':' + environ['SERVER_PORT']
        else:
            if environ['SERVER_PORT'] != '80':
               url += ':' + environ['SERVER_PORT']
    
    return url
