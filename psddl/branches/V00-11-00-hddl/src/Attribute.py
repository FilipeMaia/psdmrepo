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
from psddl.Enum import Enum

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
        self.shape = kw.get('shape')
        if self.shape: 
            self.shape = Shape(self.shape, self.parent)
        self.comment = kw.get('comment')
        self.offset = kw.get('offset')
        self.access = kw.get('access')
        self.accessor = kw.get('accessor')
        self._shape_method = kw.get('shape_method')  # as defined in DDL
        self.shape_method = None                     # constructed
        if self.shape:
            if self._shape_method:
                if self._shape_method != "None" :
                    self.shape_method = self._shape_method
            elif kw.get('accessor_name') :
                self.shape_method = kw.get('accessor_name') + '_shape'
            else:
                self.shape_method = name + '_shape'
        self.tags = kw.get('tags', {}).copy()

        self.bitfields = []

        if self.parent: self.parent.add(self)

    @property
    def stor_type(self):
        if isinstance(self.type, Enum): return self.type.base
        return self.type

    def align(self):
        return self.stor_type.align

    def sizeBytes(self):
        """Calculate full size in bytes of the attribute and return it as 
        number or expression"""
        size = ExprVal(self.stor_type.size)
        if size.value is None:
            size = ExprVal("sizeof(%s)" % self.type.name)
        if self.shape:
            size *= self.shape.size()
        return size

    def isfixed(self):
        """Returns true if both offset and shape are fixed"""
        offset = ExprVal(self.offset)
        if not offset.isconst(): return False
        if self.shape and not self.shape.isfixed(): return False
        return True

    def __str__(self):
        return "<Attribute(%s)>" % self.__dict__

    def __repr__(self):
        return "<Attribute(%s)>" % self.__dict__

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
