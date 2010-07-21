#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: $
#
# Description:
#  Module online_iface...
#
#------------------------------------------------------------------------

"""Python module defining the interface to RegDb for online.

This module contains a collection of functions/classes to be used 
directly by online when it wants to interact with RegDb. 
This is a simplified API with few things like database connection
strings predefined so that online does not need to care about nuisance 
parameters.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see LusiPython

@version $Id: $

@author Igor Gaponenko
"""


#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: $"
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
from LusiPython.DbConnection import DbConnection
from RegDB.RegDb import RegDb

#----------------------------------
# Local non-exported definitions --
#----------------------------------
_REGDB_CONN_STR = 'file:/reg/g/psdm/psdatmgr/regdb/.regdb-conn'

#------------------------
# Exported definitions --
#------------------------

def id2experiment ( id ):
    """ finds an experiment by its identifier and returns instrument and experiment names.
    
    @param id   experiment identifier, integer number >0

    A dictionary describing the requested experiment and its instrument
    will be returned in case of success. If no experiment exists in the database
    for the specified id then NOne or Falase will be returned.
    
    This function will raise an exception in case of errors, exception
    type is most probably a database-related type.
    """
    
    # setup connection to database
    conn = DbConnection ( conn_string=_REGDB_CONN_STR )
    regdb = RegDb ( conn )

    # proceed with the operation
    regdb.begin()
    exper = regdb.find_experiment_by_id( id )
    regdb.commit()
    return exper

#--------------------------------------------------
# Units tests for the module can be placed below --
#--------------------------------------------------

if __name__ == "__main__" :
    sys.exit ( "Module is not supposed to be run as main module" )
