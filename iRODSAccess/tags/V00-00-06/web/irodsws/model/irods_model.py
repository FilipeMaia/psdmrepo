#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module irods_model...
#
#------------------------------------------------------------------------

"""Brief one-line description of the module.

Following paragraphs provide detailed description of the module, its
contents and usage. This is a template module (or module template:)
which will be used by programmers to create new Python modules.
This is the "library module" as opposed to executable module. Library
modules provide class definitions or function definitions, but these
scripts cannot be run by themselves.

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
import codecs

#---------------------------------
#  Imports of base class module --
#---------------------------------
from iRODSAccess.IrodsClient import IrodsClient

#-----------------------------
# Imports for other modules --
#-----------------------------
from pylons import config
import irods

#----------------------------------
# Local non-exported definitions --
#----------------------------------

class _Connection( object ) :

    def __init__( self, config_name='irods' ):

        self._config_name = config_name
        self._conn = None

    def __del__(self):
        if self._conn : self._conn.disconnect() 

    def connection(self):
        
        if self._conn : return self._conn

        # some parameters come from the configuration
        app_conf = config['app_conf']
        host = app_conf.get( self._config_name+'.host', '127.0.0.1' )
        port = int( app_conf.get( self._config_name+'.port', 3306 ) )
        zone = app_conf.get( self._config_name+'.zone', "" )
        user = app_conf.get( self._config_name+'.user', None )
        passwd = app_conf.get( self._config_name+'.password', None )
        pwdfile = app_conf.get( self._config_name+'.pwdfile', None )
        if pwdfile and ( not user or not passwd ) :
            f = codecs.open(pwdfile,'r','base64')
            u, p = tuple(f.read().split())
            f.close()
            if not user : user = u
            if not passwd : passwd = p
    
        conn, errMsg = irods.rcConnect(host, port, user, zone)
        if conn :
            if passwd :
                status = irods.clientLoginWithPassword(conn,passwd)
                if status != 0:
                    conn.disconnect()
                    conn = None
        self._conn = conn

        return self._conn

    
#------------------------
# Exported definitions --
#------------------------


#---------------------
#  Class definition --
#---------------------
class IrodsModel ( IrodsClient ) :
    
    def __init__ ( self ) :
        IrodsClient.__init__( self, _Connection() )


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
