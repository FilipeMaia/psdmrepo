#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Method...
#
#------------------------------------------------------------------------

"""DDL type describing type's methods.

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
class Method ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, **kw ) :
        
        self.name = name
        self.type = kw.get('type')
        self.rank = kw.get('rank', 0)
        self.parent = kw.get('parent')
        self.attribute = kw.get('attribute')
        self.bitfield = kw.get('bitfield')
        self.args = kw.get('args', [])
        self.expr = kw.get('expr', {}).copy()
        self.code = kw.get('code', {}).copy()
        self.comment = kw.get('comment')
        self.access = kw.get('access', "public")
        self.static = kw.get('static', False)
        self.tags = kw.get('tags', {}).copy()

        if self.parent: self.parent.add(self)

    def __str__(self):
        return "<Method(%s)>" % self.__dict__

    def __repr__(self):
        return "<Method(%s)>" % self.name


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
