#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module DbConnection...
#
#------------------------------------------------------------------------

"""Class providing connection object for MySQL database.

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
import logging
import MySQLdb as db
import codecs
import logging
import time

#---------------------------------
#  Imports of base class module --
#---------------------------------


#-----------------------------
# Imports for other modules --
#-----------------------------
#import PkgPackage.PkgModule
#from PkgPackage.PkgModule import PkgClass


#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class DbConnection ( object ) :
    
    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, host, port, database, user, passwd, passwd_file=None, timeout=15 ) :
        """Constructor.

        @param host      server host name
        @param port      server port number or None
        @param database  server port number or None
        @param user      user name or None
        @param passwd    password or None
        @param passwd_file file with user name and password, or None
        """

        self._host = host
        self._port = port
        self._database = database
        
        if not user and not passwd and passwd_file :
            # read user name and password from file
            f = codecs.open(passwd_file,'r','base64')
            user, passwd = tuple(f.read().split())
            f.close()

        self._user = user
        self._passwd = passwd
        
        self._timeout = timeout
            
        self._conn = None

        self._log = logging.getLogger('db_conn')

    #-------------------
    #  Public methods --
    #-------------------

    # ===========================================================
    # Return database connection, attempt to reconnect if dropped
    # ===========================================================

    def connection(self):
        """ Method that returns connection to a database """
        
        if self._conn :
            try :
                self._conn.ping()
            except db.OperationalError, message:
                # we lost database connection
                self._conn = None
        
        if not self._conn :
            using_password = "NO"
            if self._passwd : using_password = "YES"
            connstr = "'%s'@'%s':%s (using password: %s)" % (
                self._user,self._host, self._port, using_password)
            self._log.debug( 'DbConnection[%s]: connecting to database: %s', self._database, connstr )
        
        t0 = time.time()
        while self._conn is None :
    
            # retry several times
            try :
                cparm = dict( host=self._host, db=self._database )
                if self._port : cparm['port']=self._port
                if self._user : cparm['user']=self._user
                if self._passwd : cparm['passwd']=self._passwd
                self._conn = db.connect( **cparm )
            except db.Error, ex :
                self._log.error( 'DbConnection[%s]: database connection failed: %s', self._database, str(ex) )
                # wait max 15 seconds
                if time.time() - t0 > self._timeout :
                    self._log.warning( 'DbConnection[%s]: retry limit exceeded, abort', self._database )
                    raise
                else :
                    time.sleep(3)
            
            if self._conn :
                
                # initialize session
                cursor = self._conn.cursor()
                cursor.execute("SET SESSION SQL_MODE='ANSI'")
            
            
        return self._conn

    #
    # Make a cursor object
    #
    def cursor (self, dict=False):
        if dict: 
            return self.connection().cursor(db.cursors.SSDictCursor)
        else:
            return self.connection().cursor()


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
