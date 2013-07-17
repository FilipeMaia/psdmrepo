#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module XmlReader...
#
#------------------------------------------------------------------------

"""Class which validates DDL.

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
import os
import os.path
import elementtree.ElementTree as ET
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.Attribute import Attribute
from psddl.Bitfield import Bitfield
from psddl.Constant import Constant
from psddl.Enum import Enum
from psddl.Method import Method
from psddl.Namespace import Namespace
from psddl.Package import Package
from psddl.Type import Type

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_const_tag = 'const'
_enum_tag = 'enum'
_enum_const_tag = 'enum-const'

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class XmlReader ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, xmlfiles, inc_dir ) :
        
        self.files = xmlfiles
        self.inc_dir = inc_dir
        
        self.included = []

    #-------------------
    #  Public methods --
    #-------------------

    def read( self ) :

        # model is the global namespace
        model = Package('')
        self._initTypes(model)

        for file in self.files:
            self._readFile( file, model, False )

        # return the model
        return model


    def _readFile( self, file, model, external ) :
        """Parse XML tree and return the list of packages"""

        # read element tree from file
        et = ET.parse(file)
        
        # get root element
        root = et.getroot() 

        # root must be psddl
        if root.tag != 'psddl' :
            raise TypeError('Root element must be psddl and not '+root.tag)
        
        # scan its direct children
        for topel in list(root) :

            # all children must be packages
            if topel.tag == 'package' : 
                
                self._parsePackage(topel, model, external)
                
            elif topel.tag == 'use' : 
                
                self._parseUse(topel, model, external)
                
            else:
                
                raise TypeError('Unexpected element in the root document: '+repr(topel.tag))



    def _parseUse(self, useel, model, external):

        file = useel.get('file')
        
        if file in self.included :
            logging.debug("XmlReader._parseUse: file %s was already included", file)
            return

        logging.debug("XmlReader._parseUse: locating file %s", file)
        path = self._findInclude(file)
        if path is None:
            raise ValueError("Cannot find include file: "+file)
        logging.debug("XmlReader._parseUse: found file %s", path)

        model.use.append(file)

        logging.debug("XmlReader._parseUse: reading file %s", path)
        self._readFile( path, model, True )

    def _parsePackage(self, pkgel, model, external):
        
        # package must have a name
        pkgname = pkgel.get('name')
        if not pkgname: raise ValueError('Package element missing name')

        # make package object if it does not exist yet
        pkg = model
        for name in pkgname.split('.'):
            obj = pkg.localName(name)
            if obj is None:
                pkg = Package(name, pkg)
            else:
                # make sure that existing name is a package
                if not isinstance(obj, Package):
                    raise ValueError('Package %s already contains name: %s ' % (pkg.name,name) )
                pkg = obj

        for subpkgel in list(pkgel) :

            # package children can be types, constants or enums
            if subpkgel.tag == 'pstype' :

                self._parseType(subpkgel, pkg, external)

            elif subpkgel.tag == _const_tag :

                self._parseConstant(subpkgel, pkg, external)

            elif subpkgel.tag == _enum_tag :

                self._parseEnum(subpkgel, pkg, external)

            else:
                
                raise TypeError('Package contains unexpected element: '+subpkgel.tag)


    def _parseType(self, typeel, pkg, external):

        # every type must have a name
        typename = typeel.get('name')
        if not typename: raise ValueError('pstype element missing name')

        # make new type object
        type = Type(typename,
                    version = typeel.get('version'),
                    type_id = typeel.get('type_id'),
                    levels = typeel.get('levels', []),
                    object_size = typeel.get('object-size'),
                    total_size = typeel.get('total-size'),
                    pack = typeel.get('pack'),
                    comment = typeel.text.strip(),
                    package = pkg,
                    external = external )

        # get attributes
        for propel in list(typeel) :

            if propel.tag == _const_tag :
                
                self._parseConstant(propel, type, external)
                
                
            elif propel.tag == 'enum' :
                
                self._parseEnum(propel, type, external)
                
            elif propel.tag == 'attribute' :
                
                self._parseAttr(propel, type)

            elif propel.tag == 'xtc-config' :
                
                # every attribute must have a name
                cfgname = propel.get('name')
                if not cfgname: raise ValueError('xtc-config element missing name')
                
                # find type for this name
                cfg = pkg.lookup(cfgname, Type)
                if not cfg: raise ValueError('unknown xtc-config name: '+cfgname)
                
                # add it to the list of config classes
                type.xtcConfig.append(cfgname)
                
            elif propel.tag == 'repeat' :
                
                # every attribute must have a name
                count = propel.get('count')
                if not count: raise ValueError('repeat element missing count attribute')
                
                # add it to the list of config classes
                type.repeat = count
                
        # calculate offsets for the data members
        type.calcOffsets()
    

    def _parseAttr(self, attrel, type):
    
        # every attribute must have a name
        attrname = attrel.get('name')
        if not attrname: raise ValueError('attribute element missing name')
        
        # find type object
        atypename = attrel.get('type')
        if not atypename: raise ValueError('attribute element missing type')
        atype = type.lookup(atypename, Type)
        if not atype: raise ValueError('attribute element has unknown type '+atypename)

        # get offset, make a number from it if possible
        attroffset = attrel.get('offset')
        if attroffset and attroffset.isdigit(): attroffset = int(attroffset)

        # create attribute
        attr = Attribute( attrname,
                          type = atype,
                          parent = type, 
                          dimensions = attrel.get('dimensions'), 
                          comment = attrel.text.strip(), 
                          offset = attroffset,
                          access = attrel.get('access') )
        logging.debug("XmlReader._parseAttr: new attribute: %s", attr)

        # accessor method for it
        accessor = attrel.get('accessor')
        if accessor :
            method = Method(accessor, 
                            attribute = attr, 
                            parent = type, 
                            type = atype,
                            comment = attr.comment)
            logging.debug("XmlReader._parseAttr: new method: %s", method)

        # get bitfields
        bfoff = 0
        for bitfel in list(attrel) :
            
            if bitfel.tag == "bitfield" :

                size = int(bitfel.get('size'))
                bftypename = bitfel.get('type', atypename)
                bftype = type.lookup(bftypename, (Type, Enum))
                if not bftype: raise ValueError('attribute bitfield has unknown type '+bftypename)

                
                bf = Bitfield(bitfel.get('name'), 
                              offset = bfoff, 
                              size = size,
                              parent = attr,
                              type = bftype,
                              comment = bitfel.text.strip() )
                logging.debug("XmlReader._parseAttr: new bitfield: %s", bf)
                bfoff += size
                
                accessor = bitfel.get('accessor')
                if accessor :
                    method = Method(accessor, 
                                    bitfield = bf, 
                                    parent = type, 
                                    type = bftype,
                                    comment = bf.comment)
                    logging.debug("XmlReader._parseAttr: new method: %s", method)


                        
    def _parseConstant(self, constel, parent, external):

        # every constant must have a name and value
        cname = constel.get('name')
        if not cname: raise ValueError('const element missing name')
        cval = constel.get('value')
        if not cval: raise ValueError('const element missing value')
        Constant(cname, cval, parent, external=external)
            
    def _parseEnum(self, enumel, parent, external):
        
        enum = Enum(enumel.get('name'), parent, external=external)
        for econst in list(enumel):
            if econst.tag != _enum_const_tag : raise ValueError('expecting %s tag'%_enum_const_tag)
            Constant(econst.get('name'), econst.get('value'), enum)

    def _findInclude(self, inc):
        
        # look in every directory in include path
        for dir in self.inc_dir:            
            path = os.path.join(dir, inc)
            if  os.path.isfile(path):
                return path
        
        # Not found
        return None

    def _initTypes(self, ns):
        """ Define few basic types in global namespace """
        Type("char", size=1, align=1, package=ns)
        Type("int8_t", size=1, align=1, package=ns)
        Type("uint8_t", size=1, align=1, package=ns)
        Type("int16_t", size=2, align=2, package=ns)
        Type("uint16_t", size=2, align=2, package=ns)
        Type("int32_t", size=4, align=4, package=ns)
        Type("uint32_t", size=4, align=4, package=ns)
        Type("int64_t", size=8, align=8, package=ns)
        Type("uint64_t", size=8, align=8, package=ns)
        Type("float", size=4, align=4, package=ns)
        Type("double", size=8, align=8, package=ns)
    
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
