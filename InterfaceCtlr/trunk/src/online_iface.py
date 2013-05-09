#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module online_iface...
#
#------------------------------------------------------------------------

"""Python module defining the interface to InterfaceDb for online.

This module contains a collection of functions/classes to be used 
directly by online when it wants to interact with InterfaceDb. 
This is a simplified API with few things like database connection
strings predefined so that online does not need to care about nuisance 
parameters.

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

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from DbTools.DbConnection import DbConnection
from InterfaceCtlr.InterfaceDb import InterfaceDb

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_ICDB_CONN_STR = 'file:/reg/g/psdm/psdatmgr/ic/.icdb-conn'

#------------------------
# Exported definitions --
#------------------------

def new_fileset ( instr, exper, runnum, runtype, xtcfiles ):
    """ Register new fileset in the Interface Controller database.
    
    @param instr     instrument name, a string
    @param exper     experiment name, a string
    @param runnum    run number, integer number >0
    @param runtype   run type, string, one of 'DATA', 'CALIB', etc.
    @param xtcfiles  sequence of the XTC file names, full path for each file
        
    This function will raise an exception in case of errors, exception
    type is most probably a database-related type. 
    """
    
    # setup connection to database
    conn = DbConnection ( conn_string=_ICDB_CONN_STR )
    idb = InterfaceDb ( conn )
    
    # Do not allow duplicate entries from online
    duplicate = False
    status = 'Waiting_Translation'
    idb.new_fileset ( instr, exper, runnum, runtype, xtcfiles, 
                      duplicate=duplicate, status=status )


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
