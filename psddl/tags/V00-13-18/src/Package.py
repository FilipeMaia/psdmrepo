#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Package...
#
#------------------------------------------------------------------------

"""DDL class representing a package.

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
class Package ( Namespace ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, parent = None, **kw) :
        
        Namespace.__init__(self, name, parent)
        
        self.comment = kw.get('comment')
        self.tags = kw.get('tags', {}).copy()
        self.use = []

    @property
    def included(self):
        """Package is included if all entities in it are included"""
        # constants
        for const in self.constants() :
            if not const.included :
                return False

        # loop over packages, enums and types
        for ns in self.namespaces() :
            if not ns.included:
                return False

        # loop over all h5schemas
        for type in self.types():
            for schema in type.h5schemas:
                if not schema.included:
                    return False

        return True

    @property
    def external(self):
        if 'external' in self.tags : return True
        if self.parent: return self.parent.external
        return False
    
    def fullName(self, lang=None, topNs=None):
        if self.external: topNs = None
        sep = {'C++' : '::'}.get(lang, '.')
        name = self.name
        if lang == 'C++' and 'c++-name' in self.tags: name = self.tags['c++-name']
        if self.parent: 
            parent = self.parent.fullName(lang, topNs)
            if parent: 
                name = parent + sep + name
            elif topNs:
                name = topNs + sep + name
        return name
    
    def __str__(self):
        return "<Package(" + self.name + ")>"

    def __repr__(self):
        return "<Package(" + self.name + ")>"


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
