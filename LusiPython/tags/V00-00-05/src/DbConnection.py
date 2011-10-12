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
import time

#---------------------------------
#  Imports of base class module --
#---------------------------------


#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#
# Internal method which parses connection string
#
def _parse_conn_string( conn_string ) :
    """
    This method parses ODBC-like connection string and returns
    a dictionary of parameters.
    """ 

    kwords = {
        'server'   : ('host',   lambda x : x),
        'database' : ('db',     lambda x : x),
        'port'     : ('port',   int),
        'uid'      : ('user',   lambda x : x),
        'pwd'      : ('passwd', lambda x : x),
        'connection timeout' : ('timeout', int),
    }

    params = {}
    
    parts = conn_string.split(';')
    for part in parts :
        part = part.strip()
        if not part : continue
        
        # split at equal sign
        try :
            key, val = part.split('=')
        except :
            raise ValueError ( "invalid format of connection string (missing =): " + conn_string )
        
        # remove leading/trailing blanks from key
        key = key.strip().lower()
        # group multiple blanks into one
        key = ' '.join(key.split())
        
        # remove leading/trailing blanks from value
        val = val.strip()
    
        # map key to db.connect keys
        try :
            ckey, fun = kwords[key]
        except KeyError:
            raise ValueError ( "invalid key in connection string (%s): %s" % ( key, conn_string ) )

        # convert the value
        try :
            val = fun(val)
        except :
            raise ValueError ( "invalid value in connection string (%s=%s): %s" % ( key, val, conn_string ) )
            
        params[ckey] = val
        
    return params
        

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
    def __init__ ( self, **kwargs ) :
        """Constructor accepts a number of keyword parameters.

        @param conn_string  connection string
        @param host         server host name
        @param db           database name
        @param port         port number
        @param user         user name
        @param passwd       password
        @param timeout      connection timeout in seconds
        
        Format of the connection string is :
        file:path, or
        Server=hostname;Port=1234;Database=myDataBase;Uid=myUsername;Pwd=myPassword;Connection Timeout=5;
        (all parts are optional, default timeout value is 15 sec)
        
        Connection parameters are initialized from connection string (if present),
        all remaining keyword arguments can be used to override values in connection string.
        If connection string is not given then only keyword arguments are used.
        If connection string starts with 'file:' string then connection string is read from 
        a file name which follows this prefix.
        
        Constructor may throw ValueError exception if connection string has unrecognized format.
        """

        # check keyword arguments
        for kw in kwargs :
            if kw not in ['conn_string', 'host', 'db', 'port', 'user', 'passwd', 'timeout'] :
                raise KeyError( 'DbConnection: unexpected keyword argument: '+kw )

        self._log = logging.getLogger('db_conn')

        self._conn_parm = {}
        self._timeout = 15
        
        if 'conn_string' in kwargs :
            
            # connection string given as kw-parameter
            conn_string = kwargs['conn_string']
            
            if conn_string.startswith('file:') :
                # read connection string from file
                conn_string = file(conn_string[5:]).read()
        
            # parse connection string if it was given
            if conn_string :
                self._conn_parm = _parse_conn_string( conn_string )

        # check all other kw parameters, they override connection string
        for kw in ['host', 'db', 'port', 'user', 'passwd', 'timeout'] :
            if kw in kwargs and kwargs[kw] is not None:
                self._conn_parm[kw] = kwargs[kw]

        # check if timeout was set in a connection string or as a kw parameter
        if 'timeout' in self._conn_parm :
            self._timeout = self._conn_parm['timeout']
            del self._conn_parm['timeout']
        
        self._conn = None

    #-------------------
    #  Public methods --
    #-------------------

    # ===========================================================
    # Return database connection, attempt to reconnect if dropped
    # ===========================================================

    def connection(self):
        """ Method that returns connection to a database. If connection
        cannot be established immediately it retries for a number of 
        seconds given in timeout parameter to constructor. If connection
        cannot be established after timeout expires an exception of a type
        MySQLdb.Error (subclass of StandardError).
        """
        
        if self._conn :
            try :
                self._conn.ping()
            except db.OperationalError:
                # we lost database connection
                self._conn = None
        
        t0 = time.time()
        while self._conn is None :
    
            # retry several times
            try :
                self._conn = db.connect( **self._conn_parm )
            except db.Error, ex :
                self._log.error( 'DbConnection[%s]: database connection failed: %s', self._conn_parm.get('db','<None>'), str(ex) )
                # wait max TIMEOUT seconds
                if time.time() - t0 > self._timeout :
                    self._log.warning( 'DbConnection[%s]: retry limit exceeded, abort', self._conn_parm.get('db','<None>') )
                    raise
                else :
                    # Retry in 3 seconds
                    time.sleep(3)
            
            if self._conn :
                
                # initialize session
                cursor = self._conn.cursor()
                cursor.execute("SET SESSION SQL_MODE='ANSI'")
            
        return self._conn

    #
    # Make a cursor object
    #
    def cursor (self, dictcursor=False):
        """Returns cursor object. It tries to establish connection to
        database first using connection() method and then returns 
        cursor for that connection. If dictcursor argument is true then 
        a 'dictionary cursor' is returned, otherwise standard cursor.
        This method will generate exception if connection to database cannot 
        be established within specified timeout.
        """
        
        
        if dictcursor: 
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
