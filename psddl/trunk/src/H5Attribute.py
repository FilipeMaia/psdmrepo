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
from psddl.Enum import Enum
from psddl.Method import Method
from psddl.Shape import Shape

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
        self.method = kw.get('method') or self.name  # corresponding method name in pstype, default is the same as name
        self._rank = kw.get('rank', -1)      # attribute array rank, -1 if unknown
        self._shape = kw.get('shape')        # shape, None if not specified
        if self._shape: 
            self._shape = Shape(self._shape, self.parent.pstype)
        self.schema_version = kw.get('schema_version', 0)      # attribute schema version
        self.tags = kw.get('tags', {}).copy()

    #-------------------
    #  Public methods --
    #-------------------

    def __str__(self):
        
        return "<H5Attribute(name=%s, type=%s, rank=%s, method=%s)>" % (self.name, self.type.name, self.rank, self.method)

    def __repr__(self):
        
        return "<H5Attribute(name=%s, type=%s, rank=%s, method=%s)>" % (self.name, self.type.name, self.rank, self.method)


    def _method(self):
        '''find corresponding method object and return it'''
        
        if 'external' in self.tags: return None
        
        if not self.parent: raise ValueError('attribute has no parent dataset, attr=%s' % self.name)

        # find pstype
        pstype = self.parent.pstype
        if not pstype: raise ValueError('no corresponding pstype found, attr=%s parent=%s' % (self.name, self.parent))
        
        # find method
        method = pstype.lookup(self.method, Method)
        if not method: raise ValueError('no corresponding method is found in pstype: %s' % self.method)

        return method

    @property
    def external(self):
        '''External attribute means that it has no corresponding method and 
        will be initialized separately'''
        return 'external' in self.tags

    @property
    def type(self):
        """Get attribute type"""
        
        if self._type is None :
            meth = self._method()
            if meth: self._type = meth.type
        return self._type

    @property
    def stor_type(self):
        """Get attribute storage type"""
        type = self.type
        if isinstance(type, Enum): return type.base
        return type

    @property
    def rank(self):
        """Get attribute rank"""

        if self._rank < 0:
            meth = self._method()
            if meth: self._rank = self._method().rank
        return self._rank

    @property
    def shape(self):
        """Get attribute shape"""

        if self._shape is None:
            meth = self._method()
            if meth: 
                attr = meth.attribute
                if attr: self._shape = attr.shape
        return self._shape

    def sizeIsConst(self):
        """Returns true for scalar attribute or attribute whose dimensions are known and constant"""
        return self.shape is not None and self.shape.isfixed()

    def sizeIsVlen(self):
        """Returns true for array attribute whose dimensions can change every event"""
        return self.rank > 0 and 'vlen' in self.tags

    def h5schema(self):
        '''find a schema for this attribute, attribute type should be user-defined type'''
        return self.type.h5schema(self.schema_version)


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
