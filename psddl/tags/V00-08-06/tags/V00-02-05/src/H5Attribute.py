#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module H5Attribute...
#
#------------------------------------------------------------------------

"""Representation of the attribute in HDF5 schema

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Andy Salnikov
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
from psddl.Method import Method

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class H5Attribute ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, **kw ) :
        
        self.name = kw.get('name')         # attribute name
        self._type = kw.get('type', None)   # optional type
        self.method = kw.get('method', None)   # corresponding method name in pstype, default is the same as name
        self.rank = kw.get('rank', 0)      # attribute data type

    #-------------------
    #  Public methods --
    #-------------------

    def __str__(self):
        
        return "<H5Attribute(name=%s, type=%s, rank=%d)>" % (self.name, self.type.name, self.rank)

    def __repr__(self):
        
        return "<H5Attribute(name=%s, type=%s, rank=%d)>" % (self.name, self.type.name, self.rank)

    @property
    def type(self):
        """Get attribute type"""
        
        # if type is explicitly defined then return it
        if self._type: return self._type
        
        # otherwise find corresponding method and use its type
        methodname = self.method or self.name

        # find pstype
        pstype = self.parent.parent.pstype
        if not pstype: raise ValueError('no corresponding pstype found')
        
        # find method
        method = pstype.lookup(methodname, Method)
        if not pstype: raise ValueError('no corresponding method is found in pstype')

        type = method.type
        
        # it's not going to change so remember it
        self._type = type
        
        return type

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
