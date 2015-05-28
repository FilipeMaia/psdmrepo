#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module H5Dataset...
#
#------------------------------------------------------------------------

"""Class corresponding to HDF5 dataset in the schema

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

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
class H5Dataset ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, **kw ) :
        
        self.name = kw.get('name')          # dataset name
        self.parent = kw.get('parent')      # corresponding H5Type object
        self.pstype = kw.get('pstype')      # corresponding Type object
        self._type = kw.get('type', None)   # optional type
        self._mthd = kw.get('method') or self.name     # corresponding method name in pstype, default is the same as name
        self._rank = kw.get('rank', -1)     # attribute array rank, -1 if unknown
        self.schema_version = kw.get('schema_version', 0)      # attribute schema version
        self.attributes = []                # list of H5Attribute objects
        self.tags = kw.get('tags', {}).copy()

        self._shape = None

    #-------------------
    #  Public methods --
    #-------------------

    def className(self):
        '''Returns name of C++ class/struct for this dataset'''
        return "dataset_{0}".format(self.name)

    def classNameNs(self):
        '''Returns full name of C++ class/struct for this dataset including parent namespace'''
        return '::'.join([self.pstype.parent.fullName('C++'), self.parent.nsName(), self.className()])

    def _method(self):
        '''find corresponding method object and return it'''
        
        # find pstype
        pstype = self.pstype
        if not pstype: raise ValueError('no corresponding pstype found, ds=%s' % (self.name,))
        
        # find method
        method = pstype.lookup(self._mthd, Method)
        if not method: raise ValueError('no corresponding method is found in pstype: %s' % self._mthd)

        return method

    @property
    def method(self):
        if self.attributes: return None
        return self._mthd

    @property
    def shape_method(self):
        """Returns name of the shape method  or None"""
        if self.attributes: return None
        attr = self._method().attribute
        if attr: return attr.shape_method

    @property
    def type(self):
        """Get type"""
        if self.attributes: return None
        if self._type is None :
            self._type = self._method().type
        return self._type

    @property
    def rank(self):
        """Get rank"""
        if self.attributes: return None
        if self._rank < 0:
            self._rank = self._method().rank
        return self._rank

    @property
    def shape(self):
        """Get shape"""
        if self.attributes: return None
        if self._shape is None:
            attr = self._method().attribute
            if attr: self._shape = attr.shape
        return self._shape

    def sizeIsConst(self):
        """Returns true for scalar attribute or attribute whose dimensions are known and constant"""
        return self.shape is not None and self.shape.isfixed()

    def sizeIsVlen(self):
        """Returns true for array attribute whose dimensions can change every event"""
        return self.rank > 0 and 'vlen' in self.tags

    def h5schema(self):
        '''find a schema for this dataset, dataset type should be user-defined type'''
        return self.type.h5schema(self.schema_version)

    def __str__(self):
        
        return "<H5Dataset(name=%s, attributes=%s)>" % (self.name, self.attributes)

    def __repr__(self):
        
        return "<H5Dataset(name=%s, attributes=%s)>" % (self.name, self.attributes)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
