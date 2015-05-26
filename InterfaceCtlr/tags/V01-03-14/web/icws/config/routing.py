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
    map.connect('/system.{renderer}', controller='system', action='status', conditions=cond_get,
                requirements={'renderer' : 'json|xml'})
    map.connect('/system', controller='system', action='status', renderer="json", conditions=cond_get )

    # GET /system/{id}
    # returns IC system status
    map.connect('/system/{id}.{renderer}', controller='system', action='status', conditions=cond_get,
                requirements={'renderer' : 'json|xml'})
    map.connect('/system/{id}', controller='system', action='status', renderer="json", conditions=cond_get )

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

    # ============== Search interface ==================

    # GET /config
    # returns list of configuration sections
    map.connect('/config.{renderer}', controller='config', action='index', conditions=cond_get,
                requirements={'renderer' : 'json|xml'})
    map.connect('/config', controller='config', action='index', renderer="json", conditions=cond_get)

    # GET /config/{section_id}
    # returns list configuration parameters for a section
    map.connect('/config/{section_id}.{renderer}', controller='config', action='section', conditions=cond_get,
                requirements={'renderer' : 'json|xml'})
    map.connect('/config/{section_id}', controller='config', action='section', renderer="json", conditions=cond_get)

    # GET /config/{section_id}/{param_id}
    # returns full info for one configuration parameter
    map.connect('/config/{section_id}/{param_id}.{renderer}', controller='config', action='parameter', conditions=cond_get,
                requirements={'renderer' : 'json|xml'})
    map.connect("param_url", '/config/{section_id}/{param_id}', controller='config', action='parameter', renderer="json", conditions=cond_get)

    # POST /config
    # create configuration parameter
    map.connect('/config', controller='config', action='create', conditions=cond_post)

    # PUT /config/{section_id}/{param_id}
    # update configuration parameter
    map.connect('/config/{section_id}/{param_id}', controller='config', action='update', conditions=cond_put)

    # DELETE /config/{section_id}/{param_id}
    # delete single configuration parameter
    map.connect('/config/{section_id}/{param_id}', controller='config', action='delete', conditions=cond_delete)

    # DELETE /config/{section_id}
    # delete all configuration parameters in one section
    map.connect('/config/{section_id}', controller='config', action='delete', param_id=None, conditions=cond_delete)

    # GET /config-full
    # returns combined list of all configuration parameters from all sections
    map.connect('/config-full.{renderer}', controller='config', action='show_full', conditions=cond_get,
                requirements={'renderer' : 'json|xml'})
    map.connect('/config-full', controller='config', action='show_full', renderer="json", conditions=cond_get)


    # ============== Access logs ==================

    # GET /log/{mode}/...
    # returns log file as HTML document
    map.connect('/log/{mode}/*path', controller='log', action='show', conditions=cond_get )

    return map
