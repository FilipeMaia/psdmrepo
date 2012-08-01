"""Routes configuration

The more specific and detailed routes should be defined first so they
may take precedent over the more generic routes. For more information
refer to the routes manual at http://routes.groovie.org/docs/
"""
from pylons import config
from routes import Mapper

def make_map():
    """Create, configure and return the routes Mapper"""
    map = Mapper(directory=config['pylons.paths']['controllers'],
                 always_scan=config['debug'])
    map.minimization = False

    # The ErrorController route (handles 404/500 error pages); it should
    # likely stay at the top, ensuring it can always be resolved
    map.connect('/error/{action}', controller='error')
    map.connect('/error/{action}/{id}', controller='error')

    # CUSTOM ROUTES HERE

    # bunch of routes for iRODS services
    
    cond_get = dict(method=['GET'])
    cond_put = dict(method=['PUT'])
    cond_post = dict(method=['POST'])
    cond_delete = dict(method=['DELETE'])
    
    ###### resources controller
    
    # GET /resources
    # list all defined resources
    map.connect('/resources', controller='resources', action='index', conditions=cond_get )
    
    # GET /resources/{id}
    # show complete info for the specified resource ID
    map.connect('/resources/{id}', controller='resources', action='show', conditions=cond_get )
    

    ###### files controller

    map.connect('/environ', controller='files', action='environ', conditions=cond_get )
    
    # GET /files/...
    # list the files
    map.connect('/files', controller='files', action='show', conditions=cond_get )
    map.connect('/files/*path', controller='files', action='show', conditions=cond_get )
    map.connect('/files/*path', controller='files', action='remove', conditions=cond_delete )

    ###### runs controller
    
    # GET /runs/{instrument}/{experiment}/{type}
    # list the run numbers
    map.connect('/runs/{instrument}/{experiment}/{type}', controller='runs', action='index', conditions=cond_get )

    # GET /runs/{instrument}/{experiment}/{type}/{runs}
    # list the files
    map.connect('/runs/{instrument}/{experiment}/{type}/{runs}', controller='runs', action='show', conditions=cond_get )

        

    # default actions

    #map.connect('/{controller}/{action}')
    #map.connect('/{controller}/{action}/{id}')

    return map
