#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module WSApp...
#
#------------------------------------------------------------------------

"""Representation of the web-service application

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
import os
import codecs
import socket
import urllib
import urllib2 as u2

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from WSClient.RequestWithMethod import RequestWithMethod

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class WSApp ( object ) :
    
    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, app, base_uri=None, kerberos=False, pwdfile=None, user=None, password=None ) :

        # build opener
        self._opener = u2.build_opener()

        # get the base URI
        if not base_uri : base_uri = os.environ.get('WSCLIENT_BASE_URI',None)
        if not base_uri : base_uri = 'https://pswww.slac.stanford.edu'

        # authentication location
        auloc = '/ws'

        # add Kerberos authentication handler if option is given
        if kerberos :
            import urllib2_kerberos
            self._opener.add_handler(urllib2_kerberos.HTTPKerberosAuthHandler())
            auloc = '/ws-kerb'
            
        if pwdfile and not user and not password :
            # if password file is given then read user name and password
            f = codecs.open(pwdfile,'r','base64')
            user, password = tuple(f.read().split())
            f.close()

        if user or password :
            auloc = '/ws-auth'

        # construct full application URI
        if not app.startswith('/') : app = '/'+app
        self.uri = base_uri + auloc + app
            
        # add basic authentication handler if user name or password is given
        if user or password :
            pwdmgr = u2.HTTPPasswordMgrWithDefaultRealm()
            pwdmgr.add_password( None, self.uri, user, password )
            self._opener.add_handler(u2.HTTPBasicAuthHandler(pwdmgr))


    #-------------------
    #  Public methods --
    #-------------------

    def method ( self, method, resource, data=None ) :

        # construct full uri
        if not resource.startswith('/') : resource = '/'+resource
        uri = self.uri + resource

        if data : 
            # encode data dictionary
            data = urllib.urlencode(data)
            # for GET we attach all parameters to URI
            if method == 'GET' and data :
                uri = uri +'?' + data
                data = None
            
        # build request
        req = RequestWithMethod ( method, uri )

        # send request, get response
        resp = self._opener.open( req, data )
        data = resp.read()
        info = resp.info()
        
        # return all info
        return dict ( uri = uri,
                      method = method,
                      info = info,
                      code = resp.code,
                      data = data )

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
