#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module WSResource...
#
#------------------------------------------------------------------------

"""Representation of the web-service resource (RESTful stuff).

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import urllib
import urllib2 as u2

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from WSClient.RequestWithMethod import RequestWithMethod
import simplejson as json

#----------------------------------
# Local non-exported definitions --
#----------------------------------


#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class WSResource ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, app, resource ) :
        """Constructor.

        @param baseURL  URL of the service
        @param resource name of the resource
        """
        # define instance variables
        self._app = app
        self._resource = resource

    #-------------------
    #  Public methods --
    #-------------------

    def get( self, data=None ) :
        """ Talk to WS using GET. """
        return self.request ( 'GET', data ) ;

    def post( self, data=None ) :
        """ Talk to WS using POST. """
        return self.request ( 'POST', data ) ;

    def put( self, data=None ) :
        """ Talk to WS using PUT. """
        return self.request ( 'PUT', data ) ;

    def delete( self, data=None ) :
        """ Talk to WS using DELETE. """
        return self.request ( 'DELETE', data ) ;

    def request ( self, method, data=None ) :

        # call the application object
        resp = self._app.method ( method, self._resource, data )
        self.uri = resp['uri']
        self.info = resp['info']
        self.code = resp['code']
        self.data = resp['data']
        
        # handle json stuff
        data = self.data
        if self.info['Content-Type'] == 'application/json' :
            data = json.loads(data)
        
        return data

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
