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

    # bunch of routes for Roles services

    
    cond_get = dict(method=['GET'])
    cond_put = dict(method=['PUT'])
    cond_post = dict(method=['POST'])
    cond_delete = dict(method=['DELETE'])

    ###### roles controller
    
    # GET /roles
    # list all defined roles
    map.connect('/roles', controller='roles', action='index', conditions=cond_get )
    
    # GET /roles/new
    # shop HTML for for defining new roles
    map.connect('/roles/new', controller='roles', action='new', conditions=cond_get )
    
    # POST /roles
    # create new role
    map.connect('/roles', controller='roles', action='create', conditions=cond_post )
    
    # PUT /roles/{app}/{role}
    # create new role
    map.connect('/roles/{app}/{role}', controller='roles', action='update', conditions=cond_put )
    
    # DELETE /roles/{app}/{role}
    # delete role (including privileges and user roles)
    map.connect('/roles/{app}/{role}', controller='roles', action='delete', conditions=cond_delete )
    
    
    ###### privileges controller
    
    # GET /roles/{app}/{role}
    # list privileges 
    map.connect('/roles/{app}/{role}', controller='privileges', action='index', conditions=cond_get )
    
    # GET /roles/{app}/{role}/new
    # show HTML form for defining new privileges
    map.connect('/roles/{app}/{role}/new', controller='privileges', action='new', conditions=cond_get )
    
    # POST /roles/{app}/{role}
    # create new privilege(s) for a role
    map.connect('/roles/{app}/{role}', controller='privileges', action='create', conditions=cond_post )
    
    # PUT /roles/{app}/{role}/{privilege}
    # create new privilege for a role
    map.connect('/roles/{app}/{role}/{privilege}', controller='privileges', action='update', conditions=cond_put )
    
    # DELETE /roles/{app}/{role}/{privilege}
    # remove privilege from a role
    map.connect('/roles/{app}/{role}/{privilege}', controller='privileges', action='delete', conditions=cond_delete )


    ###### userroles controller

    # GET /userroles
    # list all defined user roles
    map.connect('/userroles', controller='userroles', action='index', conditions=cond_get )

    # GET /userroles/new
    # display HTML form for creating ne user role
    map.connect('/userroles/new', controller='userroles', action='new', conditions=cond_get )

    # POST /userroles
    # define new user role
    map.connect('/userroles', controller='userroles', action='create', conditions=cond_post )

    # GET /userroles/{app}.{exp_id}/{user}
    # list user roles for specific user/app/exp
    map.connect('/userroles/{app}.{exp_id}/{user}', controller='userroles', action='show', conditions=cond_get )
    
    # GET /userroles/{app}.{exp_id}/{user}/privileges
    # list user privileges for specific user/app/exp
    map.connect('/userroles/{app}.{exp_id}/{user}/privileges', controller='userroles', action='showPriv', conditions=cond_get )
    
    # GET /userroles/{app}/{user}
    # list user roles for specific user/app
    map.connect('/userroles/{app}/{user}', controller='userroles', action='show', conditions=cond_get )

    # GET /userroles/{app}/{user}/privileges
    # list user privileges for specific user/app
    map.connect('/userroles/{app}/{user}/privileges', controller='userroles', action='showPriv', conditions=cond_get )
    
    # PUT /userroles/{app}.{exp_id}/{user}/{role}
    # create new user role
    map.connect('/userroles/{app}.{exp_id}/{user}/{role}', controller='userroles', action='update', conditions=cond_put )

    # PUT /userroles/{app}/{user}/{role}
    # create new user role
    map.connect('/userroles/{app}/{user}/{role}', controller='userroles', action='update', conditions=cond_put )

    # DELETE /userroles/{app}.{exp_id}/{user}/{role}
    # delete single user role
    map.connect('/userroles/{app}.{exp_id}/{user}/{role}', controller='userroles', action='delete', conditions=cond_delete )

    # DELETE /userroles/{app}/{user}/{role}
    # delete single user role
    map.connect('/userroles/{app}/{user}/{role}', controller='userroles', action='delete', conditions=cond_delete )
    
    # DELETE /userroles/{app}.{exp_id}/{user}
    # delete user roles
    map.connect('/userroles/{app}.{exp_id}/{user}', controller='userroles', action='delete', conditions=cond_delete )
    
    # DELETE /userroles/{app}/{user}
    # delete user roles
    map.connect('/userroles/{app}/{user}', controller='userroles', action='delete', conditions=cond_delete )
    
    # DELETE /userroles/{user}
    # delete all user roles
    map.connect('/userroles/{user}', controller='userroles', action='deleteUser', conditions=cond_delete )



    ###### standard stuff

    map.connect('/{controller}/{action}')
    map.connect('/{controller}/{action}/{id}')

    return map
