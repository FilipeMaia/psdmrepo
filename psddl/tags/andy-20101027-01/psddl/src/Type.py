#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Type...
#
#------------------------------------------------------------------------

"""DDL class representing a type (class).

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
class Type ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, **kw ) :
        self.name = name
        self.version = kw.get('version')
        self.type_id = kw.get('type_id')
        self.levels = kw.get('levels')
        self.comment = kw.get('comment')
        self.package = kw.get('package')
        self.size = kw.get('size')
        self.align = kw.get('align')
        self.pack = kw.get('pack')
        if self.pack : self.pack = int(self.pack)
        
        self.enums = []
        self.constants = []
        self.xtcConfig = []
        self.attributes = []
        self.methods = []
        self.repeat = None

    def __str__(self):
        
        return "<Type(%s)>" % self.__dict__

    def __repr__(self):
        
        return "<Type(%s)>" % self.name

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
