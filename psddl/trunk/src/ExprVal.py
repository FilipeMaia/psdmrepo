#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ExprVal...
#
#------------------------------------------------------------------------

"""Class which represents compile-time expression.

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
import types
import operator

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.Constant import Constant

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class ExprVal ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, val = None, ns = None ) :
        if val is None : 
            self.value = None
            self.const = False
        elif type(val) == ExprVal :
            # copy constructor
            self.value = val.value
            self.const = val.const
        elif type(val) == types.IntType: 
            # integer value is constant
            self.value = val
            self.const = True
        else:
            if ns is None: raise TypeError("ExprVal requires namespace object")
            self.value = val
            self.const = False
            
            # try to resolve the expression (recursively)
            v = ns.lookup(self.value, Constant)
            while v is not None:
                self.const = True
                self.value = v.value
                try:
                    self.value = int(self.value)
                    break
                except ValueError:
                    v = ns.lookup(self.value, Constant)

    #-------------------
    #  Public methods --
    #-------------------

    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return str(self.value)
    
    def _genop(self, other, op, strop):
        """generic operator method"""
        
        # any operation with None is None
        if self.value is None: return None
        if other.value is None: return None
        
        # if any of them is string then do string expression
        if isinstance(self.value, types.StringType) and isinstance(other.value, types.StringType):
            return "(%s)%s(%s)" % (self.value, strop, other.value)
        if isinstance(self.value, types.StringType):
            return "(%s)%s%s" % (self.value, strop, other.value)
        if isinstance(other.value, types.StringType):
            return "%s%s(%s)" % (self.value, strop, other.value)

        # otherwise do a regular operator
        return op(self.value, other.value)

    def __add__(self, other):
        expr = ExprVal(self)
        expr += other
        return expr

    def __sub__(self, other):
        expr = ExprVal(self)
        expr -= other
        return expr

    def __iadd__(self, other):
        self.value = self._genop(other, operator.add, '+')
        self.const = self.const and other.const
        return self

    def __isub__(self, other):
        self.value = self._genop(other, operator.sub, '-')
        self.const = self.const and other.const
        return self

    def __mul__(self, other):
        expr = ExprVal(self)
        expr *= other
        return expr

    def __div__(self, other):
        expr = ExprVal(self)
        expr /= other
        return expr

    def __imul__(self, other):
        self.value = self._genop(other, operator.mul, '*')
        self.const = self.const and other.const
        return self

    def __idiv__(self, other):
        self.value = self._genop(other, operator.div, '/')
        self.const = self.const and other.const
        return self

    def __cmp__(self, other):
        if type(other) == ExprVal:
            return self.value == other.value
        else:
            return self.value == other

    def isconst(self):
        """Returns true if the expression is constant"""
        return self.const

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
