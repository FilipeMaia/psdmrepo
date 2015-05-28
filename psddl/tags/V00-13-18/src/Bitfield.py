#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Bitfield...
#
#------------------------------------------------------------------------

"""Class representing bitfield definition

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
class Bitfield ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, **kw ) :

        self.name = name
        self.offset = kw.get('offset')
        self.size = kw.get('size')
        self.type = kw.get('type')
        self.parent = kw.get('parent')
        self.comment = kw.get('comment')
        self.accessor = kw.get('accessor')

        self.parent.bitfields.append(self)

    @property
    def bitmask(self):
        return (1<<self.size)-1

    def expr(self):
        
        expr = "@self."+self.parent.name
        if self.offset > 0 :
            expr = "(%s>>%d)" % (expr, self.offset)
        expr = "%s(%s & %#x)" % (self.type.name, expr, self.bitmask)
        return expr

    def assignExpr(self, name):
        
        expr = "((%s) & %#x)" % (name, self.bitmask)
        if self.offset > 0 :
            expr = "(%s<<%d)" % (expr, self.offset)
        return expr

    def __str__(self):
        return "<Bitfield(%s)>" % self.__dict__

    def __repr__(self):
        return "<Bitfield(%s)>" % self.name


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
