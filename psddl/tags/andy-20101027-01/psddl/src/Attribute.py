#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Attribute...
#
#------------------------------------------------------------------------

"""DDL object describing type's attributes.

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
from psddl.Shape import Shape
from psddl.ExprVal import ExprVal

#----------------------------------
# Local non-exported definitions --
#----------------------------------
def _dim(str):
    if str == '*' : return None
    return str
    
def _dims(dimstr):
    if not dimstr : return None
    return [_dim(d) for d in dimstr.split(',')]

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class Attribute ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, **kw ) :
        self.name = name
        self.type = kw.get('type')
        self.parent = kw.get('parent')
        self.dimensions = kw.get('dimensions')
        if self.dimensions: 
            self.dimensions = Shape(self.dimensions, self)
        self.comment = kw.get('comment')
        self.offset = kw.get('offset')
        self.access = kw.get('access')

    def align(self):
        return self.type.align

    def sizeBytes(self):
        """Calculate full size in bytes of the attribute and return it as 
        number or expression"""
        size = ExprVal(self.type.size)
        if size.value is None:
            size = ExprVal("sizeof(%s)" % self.type)
        if self.dimensions:
            size *= self.dimensions.size()
        return size

    def isfixed(self):
        """Returns true if both offset and dimensions are fixed"""
        offset = ExprVal(self.offset)
        return offset.isconst(self.parent) and self.dimensions.isfixed()

    def __str__(self):
        return "<Attribute(%s)>" % self.__dict__

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
