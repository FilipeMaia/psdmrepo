#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module H5Type...
#
#------------------------------------------------------------------------

"""Class defining schema for single type in HDF5.

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
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.H5Dataset import H5Dataset
from psddl.H5Attribute import H5Attribute

#----------------------------------
# Local non-exported definitions --
#----------------------------------
_log = logging.getLogger("H5Type")

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class H5Type ( object ) :

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, **kw ) :
        
        self.name = name
        self.package = kw.get('package')         # parent Package object
        self.pstype = kw.get('pstype', None)     # corresponding Type object
        self.datasets = []                       # List of H5Dataset objects
        self.version = kw.get('version', 0)      # version number
        self.included = kw.get('included')
        self.location = kw.get('location')
        self.tags = kw.get('tags', {}).copy()
        
        self.enum_map = {}
        
    #-------------------
    #  Public methods --
    #-------------------
    
    def nsName(self):
        '''Returns namespace for all C++ constructs in this schema'''
        return "ns_{name}_v{version}".format(name=self.name, version=self.version)

    def className(self):
        '''Returns class for all C++ constructs in this schema'''
        return "{name}_v{version}".format(name=self.name, version=self.version)

    def __str__(self):
        
        return "<H5Type(name=%s, version=%s, datasets=%s)>" % (self.name, self.version, self.datasets)

    def __repr__(self):
        
        return "<H5Type(name=%s, version=%s, datasets=%s)>" % (self.name, self.version, self.datasets)

    @staticmethod
    def defaultSchema(type):
        """Generate default schema for a types from type itself"""

        _log.debug("_defaultSchema: type=%s", type)

        # get a list of all public methods which are accessors to attributes or bitfields or take no arguments
        methods = [ meth for meth in type.methods() 
                   if meth.access == "public" and meth.name != '_sizeof' and 
                   (meth.attribute is not None or meth.bitfield is not None or not meth.args)]
        
        # schema instance
        schema = H5Type(type.name, package=type.package, pstype=type, version=0, included=type.included)
        
        # All non-array attributes of value-types will go into separate dataset.
        # All 1-dim character arrays (strings) are included here too
        # Dataset name is 'data' for event data or 'config' for config types.
        ds = None
        for meth in methods:
            if not meth.type: continue
            if (meth.rank == 0 and meth.type.value_type) or (meth.rank == 1 and meth.type.name == 'char'):
                if not ds:
                    dsname = 'config' if "config-type" in type.tags else 'data' 
                    ds = H5Dataset(name=dsname, parent=schema, pstype=type)
                    schema.datasets.append(ds)
                attr = H5Attribute(name=meth.name, type=meth.type, rank=meth.rank, method=meth.name, parent=ds)
                if meth.rank == 1 and meth.type.name == 'char': attr.tags['vlen'] = None
                ds.attributes.append(attr)
        if ds: _log.debug("_defaultSchema: scalars dataset: %s", ds)

        # for non-array attributes of user-defined types create individual datasets
        for meth in methods:
            if not meth.type: continue
            if meth.rank == 0 and not meth.type.value_type:
                # get/make that type schema
                if not meth.type.h5schemas:
                    meth.type.h5schemas = [H5Type.defaultSchema(meth.type)]
                # find its schema v0
                mschema = [s for s in meth.type.h5schemas if s.version == 0]
                if not mschema: raise ValueError("cannot find schema V0 for type "+meth.type.name)
                mschema = mschema[0]
                if mschema: _log.debug("_defaultSchema: sub-typedataset: %s", mschema)
                if len(mschema.datasets) != 1: raise ValueError("schema for sub-type "+type.name+"."+meth.type.name+" contains more than 1 dataset")
                # copy it into this schema
                ds = H5Dataset(name=meth.name, parent=schema, pstype=type, type=meth.type, rank=meth.rank, method=meth.name)
                schema.datasets.append(ds)

        # for array attributes create individual datasets
        for meth in methods:
            if not meth.type: continue
            if meth.rank > 0 and not (meth.type.name == 'char' and meth.rank == 1):
                ds = H5Dataset(name=meth.name, parent=schema, pstype=type, type=meth.type, rank=meth.rank, method=meth.name)
                schema.datasets.append(ds)

        return schema

    
    def enumConstName(self, enum_name, const_name):
        return self.enum_map.get(enum_name, {}).get(const_name, const_name)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
