#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Constant...
#
#------------------------------------------------------------------------

"""Class representing a constant definition.

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
class Constant ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, value, parent, **kw ) :
        
        self.name = name
        self.value = value
        self.parent = parent
        self.included = kw.get('included')
        self.comment = kw.get('comment', '')

        if self.parent: self.parent.add(self)
    
    def __str__(self):
        return "<Constant(" + self.name + " = " + str(self.value) + ")>"

    def __repr__(self):
        return "<Constant(" + self.name + " = " + str(self.value) + ")>"

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
