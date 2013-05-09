"""Pylons middleware initialization"""

import re

from beaker.middleware import CacheMiddleware, SessionMiddleware
from paste.cascade import Cascade
from paste.registry import RegistryManager
from paste.urlparser import StaticURLParser
from paste.deploy.converters import asbool
from pylons import config
from pylons.middleware import ErrorHandler, StatusCodeRedirect
from pylons.wsgiapp import PylonsApp
from routes.middleware import RoutesMiddleware

from irodsws.config.environment import load_environment

class MakeScriptName(object):
    ''' Some servers (mod_scgi) pass empty SCRIPT_NAME, fix it here '''
    
    def __init__(self, app, regex = None):
        self.app = app
        self.regex = regex
        if self.regex: self.regex = re.compile(self.regex)
        
    def __call__(self, environ, start_response):
        if self.regex and not environ.get('SCRIPT_NAME'):
            path = environ.get('PATH_INFO')
            match = self.regex.match(path)
            if match:
                script = match.group(0)
                environ['PATH_INFO'] = path[len(script):]
                environ['SCRIPT_NAME'] = script
            
        return self.app(environ, start_response)


class _MyApp(PylonsApp):
    """
    Subclass of PylonsApp which returns text/plain message when resource is 
    not found instead of html document.
    """

    def dispatch(self, controller, environ, start_response):

        if not controller:
            body = "Resource '%s' does not exist" % environ.get('REQUEST_URI')
            headers = [('Content-Type', 'text/plain; charset=utf8'),
                       ('Content-Length', str(len(body)))]
            start_response("404 Not Found", headers) 
            return [body]
        else:
            return PylonsApp.dispatch(self, controller, environ, start_response)


def make_app(global_conf, full_stack=True, static_files=True, **app_conf):
    """Create a Pylons WSGI application and return it

    ``global_conf``
        The inherited configuration for this application. Normally from
        the [DEFAULT] section of the Paste ini file.

    ``full_stack``
        Whether this application provides a full WSGI stack (by default,
        meaning it handles its own exceptions and errors). Disable
        full_stack when this application is "managed" by another WSGI
        middleware.

    ``static_files``
        Whether this application serves its own static files; disable
        when another web server is responsible for serving them.

    ``app_conf``
        The application's local configuration. Normally specified in
        the [app:<name>] section of the Paste ini file (where <name>
        defaults to main).

    """
    # Configure the Pylons environment
    load_environment(global_conf, app_conf)

    # The Pylons WSGI app
    app = _MyApp()

    # Routing/Session/Cache Middleware
    app = RoutesMiddleware(app, config['routes.map'])
    app = SessionMiddleware(app, config)
    app = CacheMiddleware(app, config)

    # CUSTOM MIDDLEWARE HERE (filtered by error handling middlewares)

    if asbool(full_stack):
        # Handle Python exceptions
        app = ErrorHandler(app, global_conf, **config['pylons.errorware'])

        # Display error documents for 401, 403, 404 status codes (and
        # 500 when debug is disabled)
        if asbool(config['debug']):
            app = StatusCodeRedirect(app)
        else:
            app = StatusCodeRedirect(app, [400, 401, 403, 404, 500])

    # Establish the Registry for this application
    app = RegistryManager(app)

    if asbool(static_files):
        # Serve static files
        static_app = StaticURLParser(config['pylons.paths']['static_files'])
        app = Cascade([static_app, app])

    app = MakeScriptName(app, '/ws[^/]*/irodsws(?=/)')

    return app
