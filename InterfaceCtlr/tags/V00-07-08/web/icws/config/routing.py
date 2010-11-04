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

    cond_get = dict(method=['GET'])
    cond_put = dict(method=['PUT'])
    cond_post = dict(method=['POST'])
    cond_delete = dict(method=['DELETE'])

    # GET /system
    # returns IC system status
    map.connect('/system', controller='system', action='status', conditions=cond_get )

    # GET /system/{id}
    # returns IC system status
    map.connect('/system/{id}', controller='system', action='status', conditions=cond_get )

    # DELETE /system/{id}
    # returns IC system status
    map.connect('/system/{id}', controller='system', action='stop', conditions=cond_delete )

    # ============= Request-level operations ===================

    # GET /request
    # returns list of all requests
    map.connect('/request', controller='request', action='requests', conditions=cond_get )

    # GET /request/{id}
    # returns list of all requests
    map.connect('/request/{id}', controller='request', action='requests', conditions=cond_get )

    # DELETE /request/{id}
    # remove request
    map.connect('/request/{id}', controller='request', action='delete', conditions=cond_delete )

    # PUT /request/{id}
    # update request
    map.connect('/request/{id}', controller='request', action='update', conditions=cond_put )

    # POST /request
    # create new request
    map.connect('/request', controller='request', action='create', conditions=cond_post )

    # ============== manage active experiments =============== 

    # GET /active_exp
    # returns list of active experiments
    map.connect('/active_exp', controller='active', action='index', conditions=cond_get )

    # GET /active_exp/{instrument}/{experiment}
    # returns one active experiment
    map.connect('/active_exp/{instrument}/{experiment}', controller='active', action='index', conditions=cond_get )

    # PUT /active_exp/{instrument}/{experiment}
    # add one more experiment to the list
    map.connect('/active_exp/{instrument}/{experiment}', controller='active', action='add', conditions=cond_put )

    # DELETE /active_exp/{instrument}/{experiment}
    # delete one active experiment
    map.connect('/active_exp/{instrument}/{experiment}', controller='active', action='delete', conditions=cond_delete )

    # ============== Search interface ==================

    # GET /exp
    # returns list of all instruments/experiments
    map.connect('/exp', controller='request', action='experiments', conditions=cond_get )

    # GET /exp/{instrument}/{experiment}
    # returns list of all requests for given experiment
    map.connect('/exp/{instrument}/{experiment}', controller='request', action='exp_requests', conditions=cond_get )

    # ============== Access logs ==================

    # GET /log/{mode}/...
    # returns log file as HTML document
    map.connect('/log/{mode}/*path', controller='log', action='show', conditions=cond_get )

    return map
