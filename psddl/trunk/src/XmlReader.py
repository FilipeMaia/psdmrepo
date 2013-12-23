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
import logging
try:
    import xml.etree.ElementTree as ET
except:
    import elementtree.ElementTree as ET


#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.Attribute import Attribute
from psddl.Bitfield import Bitfield
from psddl.Constant import Constant
from psddl.Constructor import Constructor, CtorArg, CtorInit
from psddl.Enum import Enum
from psddl.Method import Method
from psddl.Namespace import Namespace
from psddl.Package import Package
from psddl.Type import Type
from psddl.H5Type import H5Type
from psddl.H5Dataset import H5Dataset
from psddl.H5Attribute import H5Attribute

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_const_tag = 'const'
_enum_tag = 'enum'
_enum_const_tag = 'enum-const'

#------------------------
# Exported definitions --
#------------------------

def _cmpFiles(f1, f2):
    """Brute force file compare"""
    c1 = file(f1).read()
    c2 = file(f2).read()
    return c1 == c2

def _setTag(obj, tagelem):
    """Add tag to object"""
    
    # every tag must have a name
    tagname = tagelem.get('name')
    if not tagname: raise ValueError('tag element missing name')

    obj.tags[tagname] = tagelem.get('value')

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
        
        self.processed = []
        self.location = []

    #-------------------
    #  Public methods --
    #-------------------

    def _processed(self, file):
        for f in self.processed:
            if _cmpFiles(file, f):
                return True
        return False

    def read( self ) :

        # model is the global namespace
        model = Package('')
        self._initTypes(model)

        for file in self.files:
            if self._processed(file): continue 
            logging.debug("XmlReader.read: opening file %s", file)
            self._readFile( file, model, False )

        # return the model
        return model


    def _readFile( self, file, model, included ) :
        """Parse XML tree and return the list of packages"""

        # remember current file name 
        self.location.append( file )

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
                
                self._parsePackage(topel, model, included)
                
            elif topel.tag == 'use' : 
                
                self._parseUse(topel, model)
                
            else:
                
                raise TypeError('Unexpected element in the root document: '+repr(topel.tag))

        self.processed.append(file)
        del self.location[-1]


    def _parseUse(self, useel, model):

        file = useel.get('file')
        headers = useel.get('cpp_headers','').split()

        logging.debug("XmlReader._parseUse: locating include file %s", file)
        path = self._findInclude(file)
        if path is None:
            raise ValueError("Cannot find include file: "+file)
        logging.debug("XmlReader._parseUse: found file %s", path)

        # if this file was processed already just skip it
        if self._processed(path):
            logging.debug("XmlReader._parseUse: file %s was already included", file)
            return

        # if the include file is in the list of regular files then process it as regular file
        included = True
        for f in self.files:
            if f not in self.processed and _cmpFiles(path, f) :
                included = False
                break

        model.use.append(dict(file=file, cpp_headers=headers))

        logging.debug("XmlReader._parseUse: reading file %s", path)
        self._readFile( path, model, included )

    def _parsePackage(self, pkgel, model, included):
        
        # package must have a name
        pkgname = pkgel.get('name')
        if not pkgname: raise ValueError('Package element missing name')

        # make package object if it does not exist yet
        pkg = model
        for name in pkgname.split('.'):
            obj = pkg.localName(name)
            if obj is None:
                pkg = Package(name, pkg, 
                              comment = pkgel.text.strip(),
                              tags = self._tags(pkgel))
            else:
                # make sure that existing name is a package
                if not isinstance(obj, Package):
                    raise ValueError('Package %s already contains name: %s ' % (pkg.name,name) )
                pkg = obj

        for subpkgel in list(pkgel) :

            # package children can be types, constants or enums
            if subpkgel.tag == 'pstype' :

                self._parseType(subpkgel, pkg, included)

            elif subpkgel.tag == 'h5schema' :

                self._parseH5Type(subpkgel, pkg, included)

            elif subpkgel.tag == _const_tag :

                self._parseConstant(subpkgel, pkg, included)

            elif subpkgel.tag == _enum_tag :

                self._parseEnum(subpkgel, pkg, included)

            elif subpkgel.tag == 'tag' :

                pass

            else:
                
                raise TypeError('Package contains unexpected element: '+subpkgel.tag)


    def _parseType(self, typeel, pkg, included):

        # every type must have a name
        typename = typeel.get('name')
        if not typename: raise ValueError('pstype element missing name')

        base = typeel.get('base')
        if base: base = pkg.lookup(base, Type)

        # make new type object
        type = Type(typename,
                    version = typeel.get('version'),
                    type_id = typeel.get('type_id'),
                    levels = typeel.get('levels', []),
                    object_size = typeel.get('object-size'),
                    total_size = typeel.get('total-size'),
                    pack = typeel.get('pack'),
                    base = base,
                    comment = typeel.text.strip(),
                    tags = self._tags(typeel),
                    package = pkg,
                    included = included,
                    location = self.location[-1] )

        # get all sub-objects
        for propel in list(typeel) :

            if propel.tag == _const_tag :
                
                self._parseConstant(propel, type, included)
                
            elif propel.tag == 'enum' :
                
                self._parseEnum(propel, type, included)
                
            elif propel.tag == 'attribute' :
                
                self._parseAttr(propel, type)

            elif propel.tag == 'method' :
                
                self._parseMeth(propel, type)

            elif propel.tag == 'xtc-config' :
                
                # every attribute must have a name
                cfgname = propel.get('name')
                if not cfgname: raise ValueError('xtc-config element missing name')
                
                # find type for this name
                cfg = pkg.lookup(cfgname, Type)
                if not cfg: raise ValueError('unknown xtc-config name: '+cfgname)
                
                # add it to the list of config classes
                type.xtcConfig.append(cfg)
                
            elif propel.tag == 'tag' :

                _setTag(type, propel)

        # constructors need to be parsed last as they may depend on other types
        for propel in list(typeel) :
            if propel.tag == 'ctor' :
                self._parseCtor(propel, type)


        # calculate offsets for the data members
        type.calcOffsets()
    

    def _parseH5Type(self, typeel, pkg, included):
        """Method which parses definition of h5schema"""


        # every type must have a name
        schemaname = typeel.get('name')
        if not schemaname: raise ValueError('h5schema element missing name')
        logging.debug("XmlReader._parseH5Type: new schema: %s", schemaname)

        # find corresponding pstype
        pstype = pkg.lookup(schemaname, Type)
        if not pstype: raise ValueError('h5schema element: cannot find pstype '+schemaname)

        # make new type object
        type = H5Type(schemaname,
                      package = pkg,
                      pstype = pstype,
                      version = int(typeel.get('version')),
                      included = included,
                      location = self.location[-1] )
        pstype.h5schemas.append(type)

        # loop over sub-elements
        for propel in list(typeel) :

            if propel.tag == "dataset" :
                
                self._parseH5Dataset(propel, type, pstype)

            elif propel.tag == 'tag' :

                _setTag(type, propel)

            elif propel.tag == 'enum-map' :

                self._parseH5EnumMap(propel, type, pstype)

                
    def _parseH5EnumMap(self, elem, h5type, pstype):
        '''Method which parses enum remapping'''
        
        # get the name of the enum type
        enum_name = elem.get('enum')
        if not enum_name: raise ValueError('h5 enum-map element missing enum name')
        
        # find enum type
        enum_type = pstype.lookup(enum_name, Enum)
        if not enum_type: raise ValueError('enum-map element specifies unknown enum name: '+enum_name)
        
        constants = enum_type.constants()
        
        for el in list(elem) :

            if el.tag == "remap" :
                psname = el.get('psname')
                h5name = el.get('h5name')

                # check that name is defined
                match = [c for c in constants if c.name == psname] + [None]
                match = match[0]
                if match is None: raise ValueError('remap element specifies unknown enum constant name: '+psname)

                h5type.enum_map.setdefault(enum_name, {})[psname] = h5name
        
    def _parseH5Dataset(self, dselem, h5type, pstype):
        """Method which parses definition of h5 dataset"""

        # dataset must have a name
        dsname = dselem.get('name')
        if not dsname: raise ValueError('h5 dataset element missing name')

        logging.debug("XmlReader._parseH5Dataset: new dataset: %s", dsname)

        dstype = dselem.get('type')
        if dstype:
            dstype = pstype.lookup(dstype, (Type, Enum))
            if not dstype: raise ValueError('dataset element has unknown type '+dselem.get('type'))

        # make dataset
        ds = H5Dataset(name = dsname, 
                       parent = h5type, 
                       pstype = pstype,
                       type = dstype,
                       method = dselem.get('method'),
                       rank = int(dselem.get('rank', -1)),
                       schema_version = int(dselem.get('schema_version', 0)))
        h5type.datasets.append(ds)

        # loop over sub-elements
        for propel in list(dselem) :

            if propel.tag == "attribute" :
                
                self._parseH5Attribute(propel, ds, pstype)
                                
            elif propel.tag == 'tag' :

                _setTag(ds, propel)

    def _parseH5Attribute(self, elem, ds, pstype):
        """Method which parses definition of h5 attribute"""

        # attribute must have a name
        aname = elem.get('name')
        if not aname: raise ValueError('h5 attribute element is missing name')

        logging.debug("XmlReader._parseH5Attribute: new attribute: %s", aname)

        atype = elem.get('type')
        if atype:
            atype = pstype.lookup(atype, (Type, Enum))
            if not atype: raise ValueError('attribute element has unknown type '+elem.get('type'))

        # make attribute
        attr = H5Attribute(name = aname, 
                           parent = ds,
                           type = atype,
                           method = elem.get('method') or aname,
                           rank = int(elem.get('rank', -1)),
                           shape = elem.get('shape'),
                           schema_version = int(elem.get('schema_version', 0)))
        ds.attributes.append(attr)

        # loop over sub-elements
        for propel in list(elem) :

            if propel.tag == 'tag' :

                _setTag(attr, propel)

    def _parseAttr(self, attrel, type):
    
        # every attribute must have a name
        attrname = attrel.get('name')
        if not attrname: raise ValueError('attribute element missing name')
        
        # find type object
        atypename = attrel.get('type')
        if not atypename: raise ValueError('attribute element missing type')
        atype = type.lookup(atypename, (Type, Enum))
        if not atype: raise ValueError('attribute element has unknown type '+atypename)

        # get offset, make a number from it if possible
        attroffset = attrel.get('offset')
        if attroffset and attroffset.isdigit(): attroffset = int(attroffset)

        # create attribute
        attr = Attribute( attrname,
                          type = atype,
                          parent = type, 
                          shape = attrel.get('shape'), 
                          comment = (attrel.text or '').strip(), 
                          offset = attroffset,
                          tags = self._tags(attrel),
                          access = attrel.get('access'),
                          shape_method = attrel.get('shape_method'),
                          accessor_name = attrel.get('accessor') )
        logging.debug("XmlReader._parseAttr: new attribute: %s", attr)

        # accessor method for it
        accessor = attrel.get('accessor')
        if accessor :
            rank = 0
            if attr.shape: rank = len(attr.shape.dims)
            method = Method(accessor, 
                            attribute = attr, 
                            parent = type, 
                            type = atype,
                            rank = rank,
                            access = attr.access,
                            comment = attr.comment)
            attr.accessor = method
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
                              comment = (bitfel.text or "").strip() )
                logging.debug("XmlReader._parseAttr: new bitfield: %s", bf)
                bfoff += size
                
                accessor = bitfel.get('accessor')
                if accessor :
                    method = Method(accessor, 
                                    bitfield = bf, 
                                    parent = type, 
                                    type = bftype,
                                    access = bitfel.get('access'),
                                    comment = bf.comment)
                    bf.accessor = method
                    logging.debug("XmlReader._parseAttr: new method: %s", method)

    def _tags(self, elem):

        tags = {}
        for subel in list(elem) :
            if subel.tag == 'tag' :    
                # every tag must have a name
                tagname = subel.get('name')
                if not tagname: raise ValueError('tag element missing name')
                tags[tagname] = subel.get('value')
        return tags

    def _parseCtor(self, ctorel, parent):
    

        # may have arguments defined as <arg name='argname' type='typename' dest='attrname' method='meth_name' expr='expr'/> 
        # build a list of CtorArg objects
        args = []
        for argel in list(ctorel) :
            if argel.tag == "arg" :
                
                # name and optional type
                argname = argel.get('name')
                atype = None
                if argel.get('type'):
                    atype = parent.lookup(argel.get('type'), (Type, Enum))
                    if not atype: raise ValueError('argument element has unknown type '+argel.get('type'))
                
                # destination must be given
                dest_name = argel.get('dest')
                if not dest_name: raise ValueError('argument element missing dest attribute')
                if dest_name.lower() == 'none':
                    dest = None
                    if not atype: raise ValueError('argument without destination needs type: ' + argname)
                else:
                    dest = [d for d in parent.attributes_and_bitfields() if d.name == dest_name]
                    if not dest: raise ValueError('argument element defines unknown destination ' + dest_name)
                    dest = dest[0]
                
                # optional method
                meth = None
                if argel.get('method'):
                    meth = parent.lookup(argel.get('method'), Method)
                    if not meth: raise ValueError('argument element specifies unknown method name ' + argel.get('method'))
                elif dest:
                    meth = dest.accessor
                    if not meth: raise ValueError('argument element needs method as attribute does not have accessor. type = %s, arg = %s' % (parent.name, argname))
                    
                # optional expression
                expr = argel.get('expr')
                
                args.append(CtorArg(argname, dest, atype, meth, expr))

        # can also specify initialization values for some attributes
        # like <attr-init dest="attrname" value="value"/>
        attr_init = []
        for attrel in list(ctorel) :
            if attrel.tag == "attr-init" :
                
                # destination must be given
                dest_name = argel.get('dest')
                if not dest_name: raise ValueError('attr-init element missing dest attribute')
                dest = [d for d in parent.attributes_and_bitfields() if d.name == dest_name]
                if not dest: raise ValueError('attr-init element defines unknown destination ' + dest_name)
                dest = dest[0]

                value = argel.get('value')
                if not value: raise ValueError('attr-init element must provide value')
                
                attr_init.append(CtorInit(dest, value))


        ctor = Constructor(parent, 
                           args = args,
                           attr_init = attr_init,
                           access = ctorel.get('access', 'public'),
                           tags = self._tags(ctorel),
                           comment = (ctorel.text or "").strip())
        logging.debug("XmlReader._parseCtor: new constructor: %s", ctor)

                        
    def _parseMeth(self, methel, type):
    
        # every method must have a name
        name = methel.get('name')
        if not name: raise ValueError('method element missing name')
        
        # find type object
        mtype = None
        typename = methel.get('type')
        if typename:
            mtype = type.lookup(typename, (Type, Enum))
            if not mtype: raise ValueError('method element has unknown type '+typename)
        rank = int(methel.get('rank', 0))

        args = []
        for argel in list(methel) :
            if argel.tag == "arg" :
                argname = argel.get('name')
                typename = argel.get('type')
                atype = type.lookup(typename, (Type, Enum))
                if not atype: raise ValueError('argument element has unknown type '+typename)
                args.append((argname, atype))

        expr = {}
        for exprel in list(methel) :
            if exprel.tag == "expr" :
                lang = exprel.get('lang', "Any")
                value = exprel.get('value')
                expr[lang] = value

        code = {}
        for exprel in list(methel) :
            if exprel.tag == "code" :
                lang = exprel.get('lang', "Any")
                body = exprel.get('body')
                code[lang] = body

        method = Method(name, 
                        parent = type, 
                        type = mtype,
                        rank = rank,
                        args = args,
                        expr = expr,
                        code = code,
                        access = methel.get('access', 'public'),
                        tags = self._tags(methel),
                        comment = (methel.text or "").strip())
        logging.debug("XmlReader._parseMeth: new method: %s", method)

                        
    def _parseConstant(self, constel, parent, included):

        # every constant must have a name and value
        cname = constel.get('name')
        if not cname: raise ValueError('const element missing name')
        cval = constel.get('value')
        if not cval: raise ValueError('const element missing value')
        Constant(cname, cval, parent, included=included, comment = (constel.text or "").strip())
            
    def _parseEnum(self, enumel, parent, included):
        
        enum = Enum(enumel.get('name'), 
                    parent, 
                    base=enumel.get('base', 'int32_t'), 
                    included=included, 
                    comment = (enumel.text or "").strip())
        for econst in list(enumel):
            if econst.tag != _enum_const_tag : raise ValueError('expecting %s tag'%_enum_const_tag)
            Constant(econst.get('name'), econst.get('value'), enum, comment = (econst.text or "").strip())

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
        
        tags = {'basic': 1, 'external': 2, 'value-type': 3}
        Type("char", size=1, align=1, package=ns, tags=tags)
        Type("int8_t", size=1, align=1, package=ns, tags=tags)
        Type("uint8_t", size=1, align=1, package=ns, tags=tags)
        Type("int16_t", size=2, align=2, package=ns, tags=tags)
        Type("uint16_t", size=2, align=2, package=ns, tags=tags)
        Type("int32_t", size=4, align=4, package=ns, tags=tags)
        Type("uint32_t", size=4, align=4, package=ns, tags=tags)
        Type("int64_t", size=8, align=8, package=ns, tags=tags)
        Type("uint64_t", size=8, align=8, package=ns, tags=tags)
        Type("float", size=4, align=4, package=ns, tags=tags)
        Type("double", size=8, align=8, package=ns, tags=tags)

        tags = {'basic': 1, 'external': 2, 'c++-name': 'const char*'}
        Type("string", size=0, align=0, package=ns, tags=tags)
    
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
