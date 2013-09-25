#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Enum...
#
#------------------------------------------------------------------------

"""Class representing enum definition.

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
from psddl.Namespace import Namespace
from psddl.Type import Type

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
class Enum ( Namespace ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, parent, **kw ) :

        Namespace.__init__(self, name, parent)

        self.included = kw.get('included')
        self.comment = kw.get('comment', '')
        self.base = self.lookup(kw.get('base', 'int32_t'), Type)
        
        self.basic = True
        self.value_type = True

    def __str__(self):
        res = "<Enum(" + self.name
        for c in self.constants() :
            res += ", "
            res += c.name
            if c.value:
                res += " = "
                res += str(c.value)
        res += ")>"
        return res

    def __repr__(self):
        res = "<Enum(" + self.name + ")>"
        return res

    def unique_constants(self):
        '''
        This method returns the list of enum constants which have unique values.
        '''
        
        values = set()
        res = []
        for const in self.constants():
            if not const.value: 
                res.append(const)
            elif const.value not in values:
                values.add(const.value)
                res.append(const)

        return res

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
