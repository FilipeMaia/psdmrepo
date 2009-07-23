#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module LusiPyApp
#------------------------------------------------------------------------

""" Data model for the Roles database.

This software was developed for the LUSI project.  If you
use all or part of it, please give an appropriate acknowledgement.

Copyright (C) 2006 SLAC

@version $Id$ 

@author Andy Salnikov
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
from pylons import config
import MySQLdb as db
from MySQLdb import cursors

#---------------------------------
#  Imports of base class module --
#---------------------------------
from RoleDB.RoleDB import RoleDB

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

class _Connection( object ):
    
    def __init__(self, config_name):

        self._config_name = config_name
        self._conn = None

    def connection(self):
        """ Method that returns connection to a database """
        
        if self._conn :
            try :
                self._conn.ping()    # test connection
            except db.OperationalError, message: # loss of connection
                self._conn = None    # we lost database connection
                
        if self._conn is None :
    
            # some parameters come from the configuration
            app_conf = config['app_conf']
            host = app_conf.get( self._config_name+'.host', '127.0.0.1' )
            port = int( app_conf.get( self._config_name+'.port', 3306 ) )
            user = app_conf.get( self._config_name+'.user', None )
            dbname = app_conf.get( self._config_name+'.dbname', None )
    
            self._conn = db.connect( host=host, port=port, user=user, db=dbname )
            
        return self._conn

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class RolesModel ( RoleDB ):

    def __init__ (self) :

        RoleDB.__init__ ( self, _Connection('roledb'), _Connection('regdb') )

