#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Package...
#
#------------------------------------------------------------------------

"""DDL class representing a package.

This software was developed for the SIT project.  If you use all or 
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

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class Package ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name ) :
        self.name = name
         
        self.types = []
        self.constants = []
        self.enums = []

    def findType(self, typename):
        for t in types:
            if t.name == typename : return t
        return None
    
    def __str__(self):
        return "<Package(" + self.name + ")>"

    def __repr__(self):
        return "<Package(" + self.name + ")>"


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
