#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Shape...
#
#------------------------------------------------------------------------

"""Class describing the shape of the arrays.

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
from psddl.ExprVal import ExprVal

#----------------------------------
# Local non-exported definitions --
#----------------------------------
def _dim(str):
    if str == '*' : return None
    if str.isdigit() : return int(str)
    return str
    
def _dims(dimstr):
    if not dimstr : return None
    return [_dim(d) for d in dimstr.split(',')]

def _str(dim):
    if dim is None : return '*'
    return str(dim)
            

#------------------------
# Exported definitions --
#------------------------
                
#---------------------
#  Class definition --
#---------------------
class Shape ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, val, ns ) :
        """Constructor takes the ready tuple, list or a string"""
        
        if type(val) is type(()) :
            self.dims = tuple(val)
        elif type(val) is type([]) :
            self.dims = tuple(val)
        else:
            self.dims = tuple(_dims(val))

        self.ns = ns

    #-------------------
    #  Public methods --
    #-------------------

    def isfixed(self):
        """Return true if all dimensions have fixed size"""
        for dim in self.dims:
            expr = ExprVal(dim, self.ns)
            if not expr.isconst(): return False
        return True

    @property
    def rank(self):
        return len(self.dims)
    
    def size(self):
        """Return total array size as ExprVal or None for unknown size array"""
        return reduce(operator.mul, [ExprVal(dim, self.ns) for dim in self.dims])

    def decl(self):
        return ''.join(['[{0}]'.format(ExprVal(d, self.ns)) for d in self.dims])

    def cs_dims(self):
        '''Returns a list of comma-separated dimensions'''
        return ', '.join([str(ExprVal(d, self.ns)) for d in self.dims])

    def __str__(self):
        return ','.join([_str(d) for d in self.dims])

    def __repr__(self):
        return "<Shape("+str(self)+")>"

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
