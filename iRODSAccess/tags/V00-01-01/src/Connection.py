#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Connection...
#
#------------------------------------------------------------------------

"""Standard connection class for irods clients.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Andy Salnikov
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

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import irods

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class Connection( object ) :

    def __init__( self ):

        self._conn = None

    def __del__(self):
        if self._conn : self._conn.disconnect() 

    def connection(self):
        
        if self._conn : return self._conn

        # some parameters come from the configuration
        myEnv, status = irods.getRodsEnv()
        if status < 0:
            return self._conn
            
        host = myEnv.getRodsHost()
        port = myEnv.getRodsPort()
        user = myEnv.getRodsUserName()
        zone = myEnv.getRodsZone()
    
        conn, errMsg = irods.rcConnect(host, port, user, zone)
        if conn :
            status = irods.clientLogin(conn)
            if status != 0:
                conn.disconnect()
                conn = None
        self._conn = conn

        return self._conn

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
