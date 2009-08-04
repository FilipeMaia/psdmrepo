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

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# role assigned to users from RegDB
_regdb_role = "RegDB-ExpLeader"

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class RoleDB (object):

    def __init__ (self, roledb_conn, regdb_conn):
        """
        Constructor takes two parameters which are connection objects
        for roledb and regdb databases. Connection objects are instances 
        which have a method called connection() returning regular 
        database connection object.
        """

        self._roledb_conn = roledb_conn
        self._regdb_conn = regdb_conn


    def addRole ( self, app, role ):
        """ Create new role in the roles database """
        
        # try insert new row
        conn = self._roledb_conn.connection()
        cursor = conn.cursor ()
        cursor.execute ( "INSERT INTO role (name,app) VALUES ( %s, %s )", (role, app) )
        id = cursor.connection.insert_id()
        
        conn.commit()
    
        return id
    
    def findRole ( self, app, role ):
        """ Create new role in the roles database """
        
        conn = self._roledb_conn.connection()
        cursor = conn.cursor ()
        cursor.execute ( "SELECT id FROM role WHERE name=%s AND app=%s", (role, app) )
        id = None
        row = cursor.fetchone()
        if row : id = row[0]
        
        conn.commit()
    
        return id
    
    
    def roles ( self ):
        """ Dump the roles table """
    
        conn = self._roledb_conn.connection()
        cursor = conn.cursor ( cursors.DictCursor )
        cursor.execute ( "SELECT app, name as role FROM role" )
        res = cursor.fetchall()
        
        conn.commit()
        
        return res
    
    
    def deleteRole ( self, app, role ):
        """ Create new role in the roles database """
        
        # delete all foreign keys
        conn = self._roledb_conn.connection()
        cursor = conn.cursor ()
        cursor.execute ( "DELETE FROM priv WHERE role_id IN (SELECT id FROM role WHERE name=%s AND app=%s)", (role, app) )
        cursor.execute ( "DELETE FROM user WHERE role_id IN (SELECT id FROM role WHERE name=%s AND app=%s)", (role, app) )
        cursor.execute ( "DELETE FROM role WHERE name=%s AND app=%s", (role, app) )
        
        res = cursor.rowcount
        conn.commit()
        
        return res

    def _regdbapps(self):
        """ Return the list of applications for which there is a "RegDB" role """

        # next get the list of applications for which "RegDB" role is defined
        conn = self._roledb_conn.connection()
        cursor = conn.cursor()
        cursor.execute ( "SELECT app FROM role WHERE name = %s", (_regdb_role,) )
        regdb_apps = cursor.fetchall()
        
        conn.commit()
        
        return [ a[0] for a in regdb_apps ]

    #
    # Methods for priv table
    #
    
    def addPrivilege ( self, app, role, priv ):
        """ Add privilege to a role """
        
        conn = self._roledb_conn.connection()
        cursor = conn.cursor ()
        cursor.execute ( "INSERT INTO priv (name,role_id) SELECT %s, id FROM role WHERE name=%s AND app=%s", 
                         (priv, role, app) )
        id = cursor.connection.insert_id()
        
        conn.commit()
    
        return id
    
    def privileges ( self, app, role ):
        """ Get all privileges defined for a role """
        
        conn = self._roledb_conn.connection()
        cursor = conn.cursor ()
        cursor.execute ( "SELECT priv.name FROM priv, role " +
                         " WHERE role.id = priv.role_id AND role.app=%s AND role.name=%s",
                         (app, role) )
        res = cursor.fetchall()
        privs = set ( [ x[0] for x in res ] )
        
        conn.commit()
                
        return list(privs)
    
    def deletePrivilege( self, app, role, privilege ) :
        
        conn = self._roledb_conn.connection()
        cursor = conn.cursor ()
        cursor.execute ( "DELETE FROM priv" +
                         " WHERE role_id IN (SELECT id FROM role WHERE name=%s AND app=%s) AND name=%s", 
                         (role, app, privilege) )
        res = cursor.rowcount
        conn.commit()

        return res

    #
    # Methods for user table
    #
    
    def addUserRole ( self, app, expNameOrId, user, role ):
        """ Create new user record in the roles database """
        
        try :
            exp_id = self._expId (expNameOrId)
        except :
            return None
        
        conn = self._roledb_conn.connection()
        cursor = conn.cursor ()
        cursor.execute ( "INSERT INTO user (exp_id, user, role_id) " +
                         "SELECT %s, %s, id FROM role WHERE name=%s AND app=%s",
                         (exp_id, user, role, app) )
        id = cursor.connection.insert_id()
        
        conn.commit()
    
        return id

    
    def userroles ( self ):
        """ Dump the user table"""

        # get the list of the explicitly defined roles first
        conn = self._roledb_conn.connection()
        cursor = conn.cursor ( cursors.DictCursor )
        cursor.execute ( "SELECT u.exp_id, r.app as app, u.user as user, r.name as role "+
                         " FROM user as u, role as r " +
                         " WHERE u.role_id=r.id" )
        res_roles = list(cursor.fetchall())
        
        # next get the list of applications for which "RegDB" role is defined
        cursor1 = conn.cursor ()
        cursor1.execute ( "SELECT app FROM role WHERE name = %s", (_regdb_role,) )
        regdb_apps = [ row[0] for row in cursor1.fetchall() ]

        # done with this database
        conn.commit()

        # get the list of experiments and user names from RegDB 
        regdb = self._regdb_conn.connection()
        cursor = regdb.cursor ()
        cursor.execute ( "SELECT id, leader_account as user FROM experiment" )
        res_regdb = cursor.fetchall()
        regdb.commit()

        # merge it all together
        for reg in res_regdb :
            for app in regdb_apps :
                role = dict( exp_id=reg[0], user=reg[1], app=app, role=_regdb_role )
                res_roles.append ( role )

        return res_roles
    
    def getUserRoles ( self, app, expNameOrId, user ):
        """ Find all the roles for a given user/application """

        try :
            exp_id = self._expId (expNameOrId)
        except :
            return None
        
        # find the roles explicitly defined in the database
        conn = self._roledb_conn.connection()
        cursor = conn.cursor()
        if exp_id is None :
            cursor.execute ( "SELECT role.name FROM user, role " +
                             " WHERE role.id = user.role_id " +
                             "   AND user.exp_id IS NULL " +
                             "   AND role.app=%s AND user.user=%s",
                             (app, user) )
        else :
            cursor.execute ( "SELECT role.name FROM user, role " +
                             " WHERE role.id = user.role_id " +
                             "   AND ( user.exp_id=%s OR user.exp_id IS NULL ) " +
                             "   AND role.app=%s AND user.user=%s",
                             (exp_id, app, user) )
        res = cursor.fetchall()
                
        roles = set ( [ x[0] for x in res ] )

        # now there may be also implicit roles, but they only work with explicit experiment name
        if exp_id is not None :
            
            cursor.execute ( "SELECT COUNT(*) FROM role WHERE name = %s AND app = %s", (_regdb_role,app) )
            imp_count = cursor.fetchone()[0]
            
            if imp_count > 0 :
                
                # get the list of the experiments for this user from RegDB
                regdb = self._regdb_conn.connection()
                cursor = regdb.cursor ()
                cursor.execute ( "SELECT COUNT(*) FROM experiment WHERE id = %s AND leader_account = %s", (exp_id,user) )
                regdb_count = cursor.fetchone()[0]
                
                regdb.commit()
    
                if regdb_count : roles.add(_regdb_role)
        
        # done with database
        conn.commit()

        return list(roles)
    

    def getUserPrivileges ( self, app, expNameOrId, user ):
        """ Find all the privileges for a given user/application """
        
        try :
            exp_id = self._expId (expNameOrId)
        except :
            return None
        
        res = []
        
        # get RegDB role's privileges
        conn = self._roledb_conn.connection()
        cursor = conn.cursor()
        cursor.execute ( "SELECT priv.name FROM priv, role " +
                         " WHERE role.id = priv.role_id " +
                         "   AND role.app=%s AND role.name=%s",
                        (app, _regdb_role) )
        implicit = cursor.fetchall()
        
        if implicit :
            
            # get the list of the experiments for this user from RegDB
            regdb = self._regdb_conn.connection()
            rcursor = regdb.cursor ()
            if exp_id is None :
                rcursor.execute ( "SELECT COUNT(*) FROM experiment WHERE leader_account = %s", (user,) )
            else :
                rcursor.execute ( "SELECT COUNT(*) FROM experiment WHERE id = %s AND leader_account = %s", (exp_id,user) )
            count_regdb = cursor.fetchall()[0]
            regdb.commit()
            
            if res_regdb > 0 : res += implicit

        # get explicit privileges
        if exp_id is None :
            cursor.execute ( "SELECT priv.name FROM user, priv, role " +
                             " WHERE role.id = user.role_id " +
                             "   AND role.id = priv.role_id " +
                             "   AND user.exp_id IS NULL " +
                             "   AND role.app=%s AND user.user=%s",
                             (app, user) )
        else :
            cursor.execute ( "SELECT priv.name FROM user, priv, role " +
                             " WHERE role.id = user.role_id " +
                             "   AND role.id = priv.role_id " +
                             "   AND ( user.exp_id = %s OR user.exp_id IS NULL )" +
                             "   AND role.app=%s AND user.user=%s",
                             (exp_id, app, user) )
        res += cursor.fetchall()
        
        conn.commit()

        # get first column of the result set
        roles = set ( [ x[0] for x in res ] )
        
        return list(roles)
    
    def deleteUserRole( self, app, expNameOrId, user, role ) :
        """ Delete one user role """
        
        try :
            exp_id = self._expId (expNameOrId)
        except :
            return None
        
        conn = self._roledb_conn.connection()
        cursor = conn.cursor()
        if exp_id is None :
            cursor.execute ( "DELETE FROM user" +
                             " WHERE role_id IN (SELECT id FROM role WHERE name=%s AND app=%s)" +
                             " AND user = %s and exp_id IS NULL", (role, app, user) )
        else :
            cursor.execute ( "DELETE FROM user" +
                             " WHERE role_id IN (SELECT id FROM role WHERE name=%s AND app=%s)" +
                             " AND user = %s and exp_id = %s", (role, app, user, exp_id) )
        res = cursor.rowcount
        conn.commit()
        
        return res

    def deleteUserRoles( self, app, expNameOrId, user ) :
        """ Delete all roles for a user/app """
        
        try :
            exp_id = self._expId (expNameOrId)
        except :
            return None
        
        conn = self._roledb_conn.connection()
        cursor = conn.cursor()
        if exp_id is None :
            cursor.execute ( "DELETE FROM user" +
                             " WHERE role_id IN (SELECT id FROM role WHERE app=%s)" +
                             " AND user = %s and exp_id IS NULL", (app, user) )
        else :
            cursor.execute ( "DELETE FROM user" +
                             " WHERE role_id IN (SELECT id FROM role WHERE app=%s)" +
                             " AND user = %s and exp_id = %s", (app, user, exp_id) )
        res = cursor.rowcount
        conn.commit()
        
        return res

    def deleteUser( self, user ) :
        """ Delete all roles for a user """
        
        conn = self._roledb_conn.connection()
        cursor = conn.cursor()
        cursor.execute ( "DELETE FROM user WHERE user = %s", (user,) )

        res = cursor.rowcount
        conn.commit()

        return res

    #--------------------
    #  Private methods --
    #--------------------
    
    def _expId (self, expNameOrId):
        
        # None is special
        if expNameOrId is None : return None
        
        try :
            # if it is a number then it is just Id
            expId = int(expNameOrId)
            return expId
        except :
            # Not a number, expect it to be InstrName-ExpName
            w = expNameOrId.split('-')
                    
            if len(w) != 2 : raise ValueError(expNameOrId)

            instr = w[0]
            exp = w[1]
            
            # query RegDB for corresponding experiment id
            regdb = self._regdb_conn.connection()
            cursor = regdb.cursor ()
            q = "SELECT e.id FROM experiment as e, instrument as i where i.name=%s and e.name=%s and i.id=e.instr_id"
            cursor.execute ( q, (instr,exp) )
            res = cursor.fetchall()
            regdb.commit()

            # must be one row e
            if len(res) != 1 : raise ValueError(expNameOrId)

            return res[0][0]
            