#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Exceptions...
#
#------------------------------------------------------------------------

"""Set of exception classes for iRODSAccess package.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Andy Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"

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

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class IrodsException(Exception):
    
    def __init__(self, *args):
        Exception.__init__(self, *args)
        
class ConnectionError(IrodsException):
    
    def __init__(self, *args):
        IrodsException.__init__(self, "No connection to iRODS server")
        
class MissingError(IrodsException):
    pass

class ObjectMissing(MissingError):

    def __init__(self, objectName):
        MissingError.__init__(self, "Object '%s' does not exist" % objectName)
        
class CollectionMissing(MissingError):

    def __init__(self, collName):
        MissingError.__init__(self, "Collection '%s' does not exist" % collName)
        
class ObjectReplicaMissing(MissingError):

    def __init__(self, objectName, replica):
        MissingError.__init__(self, "Object or specified replica does not exist: '%s', replica=%s" % (objectName, replica))
        
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
