#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Constructor...
#
#------------------------------------------------------------------------

"""Class describing type's constructor.


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
class Constructor ( object ) :
    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent, **kw ) :
        
        self.parent = parent

        self.args = kw.get('args', [])
        self.attr_init = kw.get('attr_init', {})
        self.comment = kw.get('comment')
        self.access = kw.get('access', "public")
        self.tags = kw.get('tags', {}).copy()
        
        self.parent.ctors.append(self)

    #-------------------
    #  Public methods --
    #-------------------

    def __str__(self):
        return "<%s(%s)>" % (self.parent.name, self.args)

    def __repr__(self):
        return "<%s(%s)>" % (self.parent.name, self.__dict__)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
