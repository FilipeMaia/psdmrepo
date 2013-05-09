#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module roles_model
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

#---------------------------------
#  Imports of base class module --
#---------------------------------
from RoleDB.RoleDB import RoleDB
from DbTools.DbConnection import DbConnection

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class RolesModel ( RoleDB ):

    def __init__ (self) :

        app_conf = config['app_conf']
        roledb = DbConnection(conn_string=app_conf.get('roledb.conn'))
        regdb = DbConnection(conn_string=app_conf.get('regdb.conn'))
        RoleDB.__init__ ( self, roledb, regdb )

