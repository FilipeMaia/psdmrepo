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
        self.parent = kw.get('parent')      # parent dataset
        self._type = kw.get('type', None)   # optional type
        self.method = kw.get('method', self.name)   # corresponding method name in pstype, default is the same as name
        self._rank = kw.get('rank', -1)      # attribute array rank, -1 if unknown
        self.schema_version = kw.get('schema_vetsion', 0)      # attribute schema version

        self._shape = None

    #-------------------
    #  Public methods --
    #-------------------

    def __str__(self):
        
        return "<H5Attribute(name=%s, type=%s, rank=%s, method=%s)>" % (self.name, self.type.name, self.rank, self.method)

    def __repr__(self):
        
        return "<H5Attribute(name=%s, type=%s, rank=%s, method=%s)>" % (self.name, self.type.name, self.rank, self.method)


    def _method(self):
        '''find corresponding method object and return it'''
        
        # find pstype
        pstype = self.parent.pstype
        if not pstype: raise ValueError('no corresponding pstype found')
        
        # find method
        method = pstype.lookup(self.method, Method)
        if not method: raise ValueError('no corresponding method is found in pstype: %s', self.method)

        return method

    @property
    def type(self):
        """Get attribute type"""
        
        if self._type is None :
            self._type = self._method().type
        return self._type

    @property
    def rank(self):
        """Get attribute rank"""

        if self._rank < 0:
            self._rank = self._method().rank
        return self._rank

    @property
    def shape(self):
        """Get attribute shape"""

        if self._shape is None:
            attr = self._method().attribute
            if attr: self._shape = attr.shape
        return self._shape

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
