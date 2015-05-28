#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module HddlReader...
#
#------------------------------------------------------------------------

"""Parser for DLL files.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Andy Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import os.path
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
from psddl.Constructor import Constructor, CtorArg, CtorInit
from psddl.Enum import Enum
from psddl.ExprVal import ExprVal
from psddl.Method import Method
from psddl.Namespace import Namespace
from psddl.Package import Package
from psddl.Type import Type
from psddl.H5Type import H5Type
from psddl.H5Dataset import H5Dataset
from psddl.H5Attribute import H5Attribute
from psddl.HddlYacc import HddlYacc
from psddl.HddlYacc import QID

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# list of integer types    
_intTypes = ["int8_t", "uint8_t",  "int16_t", "uint16_t", 
             "int32_t", "uint32_t", "int64_t", "uint64_t",
             ]

class _error(SyntaxError):
    def __init__(self, filename, lineno, message):
        msg = "{0}:{1}: {2}".format(filename, lineno, message)
        SyntaxError.__init__(self, msg)
#         self.filename = filename
#         self.lineno = lineno

# mapping of the HDDL tag names to internal tag names
_tagmap = {
    'no_sizeof':    'no-sizeof',
    'cpp_name':     'c++-name',
    'value_type':   'value-type',
    'config_type':  'config-type',
}

def _tags(decl):
    ''' 
    Makes dictionary of tags from declaration. Filters out 'doc' tags.
    If tag has more than one argument then tag value will be a tuple of 
    arguments, otherwise it will be a value of first argument (or None)
    '''
    tags = {}
    for tag in decl['tags']:
        name = tag['name']
        name = _tagmap.get(name, name)
        args = tag['args']
        if name == 'doc': continue
        if args is None: 
            tags[name] = None
        elif len(args) == 1:
            tags[name] = args[0]
        else:
            tags[name] = args
    return tags


def _tagval(decl, tagname, default = None):
    ''' 
    Get tag values for a declaration, returns a list or None.
    If tag appears multiple times or has multiple arguments they are all
    merged into a single list. Tags without arguments are skipped.
    
    [[tag(a), tag(b)]] [[tag]] -> [a, b]
    [[tag]] -> []
    [[tag(a, b), tag(c)]] -> [a, b, c]
    '''
    values = None
    for tag in decl['tags']:
        if tag['name'] == tagname:
            if tag['args'] is not None:
                if values is None: values = []
                values += tag['args']
    if values is None: values = default
    return values


def _doc(decl):
    ''' 
    extract doc string from a declaration, this is done by merging all 'doc' tags together.
    '''
    return '\n'.join(_tagval(decl, 'doc', []))

def _constExprToString(decl):
    ''' convert parsed constant expression into string representation '''
    op = decl['op']     # opname, string
    lhs = decl['left']  # None or dict
    rhs = decl['right'] # dict or IDENTIFIER or QID
    if op is None:
        # number or identifier
        return str(rhs)
    elif lhs is None:
        if op == '(':
            return '(' + _constExprToString(rhs) + ')'
        else:
            # unary op
            return op + _constExprToString(rhs)
    else:
        # binary op
        if op == 'LSHIFT': 
            op = '<<'
        elif op == 'RSHIFT': 
            op = '>>'
        return _constExprToString(lhs) + op + _constExprToString(rhs)

def _cmpFiles(f1, f2):
    """Brute force file compare"""
    c1 = file(f1).read()
    c2 = file(f2).read()
    return c1 == c2

def _lineno(decl):
    ''' Return line number for a declaration '''
    return decl['pos'][0][0]

def _access(decl):
    ''' Return line number for a declaration '''
    for tag in decl['tags']:
        tagname = tag['name']
        if tagname in ['public', 'protected', 'private']: return tagname
    return 'public'

def _hasDevelTag(decl):
    for tag in decl['tags']:
        if tag['name'] == 'devel': 
            return True
    return False

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class HddlReader ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, ddlfiles, inc_dir, parseDevel=False ) :
        self.files = ddlfiles
        self.inc_dir = inc_dir
        self.parseDevelTypes = parseDevel

        # list of all files already processed (or being processed)
        # each item is a tuple (name, data)
        self.processed = []
        
        # stack (LIFO) of file names currently in processing
        self.location = []  

        # list of devel types encountered
        self.develTypes = []
    
    #-------------------
    #  Public methods --
    #-------------------

    def _isDevelType(self, typename, pkg):
        for develType in self.develTypes:
            if develType['typeName']==typename and pkg.fullName()==develType['pkgName']:
                return True
        return False

    def _processed(self, file):
        ''' 
        Check if file is already processed, just compare file data 
        byte-by-byte with the data from files already processed 
        '''
        data = open(file).read()
        for f, d in self.processed:
            if data == d:
                return True
        return False

    def read( self ) :
        '''
        Read all files and build a model
        '''

        # model is the global namespace
        model = Package('')
        self._initTypes(model)

        for file in self.files:
            if self._processed(file): continue 
            logging.debug("HddlReader.read: opening file %s", file)
            self._readFile( file, model, False )

        # return the model
        return model


    def _readFile( self, file, model, included ) :
        """Read one file and parse its contents.
        
          @param file     name of the file to read
          @param model    model instance (global namespace)
          @param included  if true then we are processing included file
        """

        # opne file and read its data
        data = open(file).read()

        # remember current file name 
        self.location.append(file)
        self.processed.append((file, data))

        parser = HddlYacc(debug=0)
        
        try:

            # parse data, build tree
            tree = parser.parse(data, file)
    
            # parse all includes first
            for include in tree['includes']:
                self._parseInclude(include, model)
    
            # top level namespace is a kind of package because it can 
            # contain all the same stuff but it does not have a parent
            parent = None
            self._parsePackage(tree, model, parent, included)
                
        except EOFError, ex:
            raise
        except SyntaxError, ex:
            print >>sys.stderr, ex
            for loc in reversed(self.location[:-1]):
                print >>sys.stderr, "    included from:", loc
            # special type of exception will stop processing
            raise EOFError
                

        del self.location[-1]


    def _parseInclude(self, indict, model):
        '''
        process include statement
        '''

        # check tag names
        self._checktags(indict, ['headers'])

        file = indict['name']
        headers = _tagval(indict, 'headers', [])

        logging.debug("HddlReader._parseInclude: locating include file %s", file)
        path = self._findInclude(file)
        if path is None:
            msg = "Cannot locate include file '{0}'".format(file)
            raise _error(self.location[-1], _lineno(indict), msg)
        logging.debug("HddlReader._parseInclude: found file %s", path)

        # if this file was processed already just skip it
        if self._processed(path):
            logging.debug("HddlReader._parseInclude: file %s was already included", file)
            return

        # if the include file is in the list of regular files then process it as regular file
        included = True
        for f in self.files:
            if not self._processed(f) and _cmpFiles(path, f) :
                included = False
                break

        # remember all includes
        model.use.append(dict(file=file, cpp_headers=headers))

        # get data from file
        logging.debug("HddlReader._parseInclude: reading file %s", path)
        self._readFile( path, model, included )



    def _parsePackage(self, pkgdict, model, parent, included):
        ''' Parse package definition
          
          @param pkgdict dictionary with package declaration
          @param model   full model instance
          @param parent  parent package or None when parsing top-level namespace
          @param included  true if in included file
        '''
        
        pkgname = pkgdict['name']  # QID instance, not a string

        # check tag names
        self._checktags(pkgdict, ['external', 'cpp_name', 'doc'])

        # make package object if it does not exist yet
        if parent:
            obj = parent.localName(pkgname)
            if obj is None:
                pkg = Package(pkgname, parent, 
                              comment = _doc(pkgdict),
                              tags = _tags(pkgdict))
            else:
                # make sure that existing name is a package
                if not isinstance(obj, Package):
                    msg = "while defining new package - package '{0}' already contains name '{1}' which is not a package".format(pkg.name, pkgname)
                    raise _error(self.location[-1], _lineno(pkgdict), msg)
                pkg = obj
        else:
            # it's top-level package, we fill model itself
            pkg = model

        for decl in pkgdict['declarations']:

            # package children can be types, constants or enums
            
            if decl['decl'] == 'package' :

                self._parsePackage(decl, model, pkg, included)

            elif decl['decl'] == 'type' :

                self._parseType(decl, pkg, included)

            elif decl['decl'] == 'h5schema' :

                self._parseH5Type(decl, pkg, included)

            elif decl['decl'] == 'const' :

                self._parseConstant(decl, pkg, included)

            elif decl['decl'] == 'enum' :

                self._parseEnum(decl, pkg, included)

            else:
                
                msg = "Package '{0}' contains unexpected declaration: {1}".format(pkgname, decl['decl'])
                raise _error(self.location[-1], _lineno(decl), msg)


    def _parseType(self, typedict, pkg, included):

        # check tag names
        self._checktags(typedict, ['value_type', 'devel', 'config_type', 'config', 'pack', 'no_sizeof', 'external', 'type_id', 'cpp_name', 'doc'])
        if _hasDevelTag(typedict):
            self.develTypes.append({'pkgName':pkg.fullName(), 'typeName':typedict['name']})
            if not self.parseDevelTypes: return
        # every type must have a name
        typename = typedict['name']
        
        # check the name
        if pkg.localName(typename):
            msg = "Name '{0}' is already defined in package {1}".format(typename, pkg.name)
            raise _error(self.location[-1], _lineno(typedict), msg)

        base = typedict['base']
        if base: 
            base = pkg.lookup(str(base), Type)
            if not base:
                msg = "Failed to resolve name of a base type '{0}'".format(typedict['base'])
                raise _error(self.location[-1], _lineno(typedict), msg)

        # check type_id tag
        xtc_type_id, xtc_version = self._getTypeId(typedict)

        # make new type object
        type = Type(typename,
                    version = xtc_version,
                    type_id = xtc_type_id,
                    levels = [],
                    pack = self._getIntTag(typedict, 'pack'),
                    base = base,
                    xtcConfig = self._getConfigTypes(typedict, pkg),
                    comment = _doc(typedict),
                    tags = _tags(typedict),
                    package = pkg,
                    included = included,
                    location = self.location[-1] )

        # first parse enums and constants
        for decl in typedict['declarations']:

            if decl['decl'] == 'const':
                
                self._parseConstant(decl, type, included)
                
            elif decl['decl'] == 'enum':
                
                self._parseEnum(decl, type, included)
                

        # next do members and methods as they may depend on other types (enums)
        for decl in typedict['declarations']:
            
            if decl['decl'] == 'member' :
                
                self._parseAttr(decl, type)

            elif decl['decl'] == 'method' :
                
                self._parseMeth(decl, type)

        # constructors need to be parsed last as they may depend on other types (methods and enums)
        for decl in typedict['declarations']:
            
            if decl['decl'] == 'ctor':
                
                self._parseCtor(decl, type)

        # calculate offsets for the data members
        type.calcOffsets()
    

    def _parseH5Type(self, schemadict, pkg, included):
        """Method which parses definition of h5schema"""

        # check tag names
        self._checktags(schemadict, ['version', 'embedded', 'default', 'external', 'doc'])

        # every type must have a name
        schemaname = schemadict['name']

        # find corresponding pstype
        pstype = pkg.lookup(schemaname, Type)
        if not pstype: 
            msg = "Failed to lookup name of a type '{0}', check that type exist and include its definition.".format(schemaname)
            raise _error(self.location[-1], _lineno(schemadict), msg)

        version = self._getIntTag(schemadict, 'version')
        if version is None: version = 0

        # make new type object
        type = H5Type(schemaname,
                      package = pkg,
                      pstype = pstype,
                      version = version,
                      tags = _tags(schemadict),
                      included = included,
                      location = self.location[-1] )
        pstype.h5schemas.append(type)

        # loop over sub-elements
        for decl in schemadict['declarations']:

            if decl['decl'] == "h5ds" :
                
                self._parseH5Dataset(decl, type, pstype)

            elif decl['decl'] == 'enum_remap' :

                self._parseH5EnumMap(decl, type, pstype)

                
    def _parseH5EnumMap(self, edecl, h5type, pstype):
        '''Method which parses enum remapping'''
        
        # check tag names
        self._checktags(edecl, ['doc'])

        enum_name = edecl['name']
        
        # find enum type
        enum_type = pstype.lookup(enum_name, Enum)
        if not enum_type: 
            msg = "Failed to resolve name of enum type '{0}'".format(enum_name)
            raise _error(self.location[-1], _lineno(edecl), msg)

        # constants defined in psana type        
        constants = enum_type.constants()
        
        for remap in edecl['remaps']:
            
            psname = remap['from']
            h5name = remap['to']

            # check that name is defined
            match = [c for c in constants if c.name == psname] + [None]
            match = match[0]
            if match is None: 
                msg = "Failed to resolve enum constant name '{0}'".format(psname)
                raise _error(self.location[-1], _lineno(remap), msg)

            h5type.enum_map.setdefault(enum_name, {})[psname] = h5name
        
    def _parseH5Dataset(self, dsdecl, h5type, pstype):
        """Method which parses definition of h5 dataset"""

        # check tag names
        self._checktags(dsdecl, ['method', 'vlen', 'external', 'zero_dims', 'doc'])

        dsname = dsdecl['name']

        logging.debug("HddlReader._parseH5Dataset: new dataset: %s", dsname)

        dstype = dsdecl['type']
        if dstype:
            dstype = pstype.lookup(dstype, (Type, Enum))
            if not dstype: 
                msg = "Failed to resolve dataset type name '{0}'".format(dsdecl['type'])
                raise _error(self.location[-1], _lineno(dsdecl), msg)

        # it may specify method name via tags, argument may be QID, we need string
        method = _tagval(dsdecl, 'method', [None])[0]
        if method: method = str(method)

        # shape can be specified as a rank only (number or None)
        rank = dsdecl['shape']
        if rank is None: rank = -1
        
        # optional schema version in tags
        schema_version = self._getIntTag(dsdecl, 'schema_version', 0)

        # make dataset
        ds = H5Dataset(name = dsname, 
                       parent = h5type, 
                       pstype = pstype,
                       type = dstype,
                       method = method,
                       rank = rank,
                       schema_version = schema_version,
                       tags = _tags(dsdecl))
        h5type.datasets.append(ds)

        # loop over sub-elements
        if dsdecl['attributes'] is not None:
            for adecl in dsdecl['attributes']:
                self._parseH5Attribute(adecl, ds, pstype)

    def _parseH5Attribute(self, adecl, ds, pstype):
        """Method which parses definition of h5 attribute"""

        # check tag names
        self._checktags(adecl, ['external', 'method', 'vlen', 'doc'])

        # attribute must have a name
        aname = adecl['name']

        logging.debug("HddlReader._parseH5Attribute: new attribute: %s", aname)

        atype = adecl['type']
        if atype:
            atype = pstype.lookup(str(atype), (Type, Enum))
            if not atype: 
                msg = "Failed to resolve attribute type name '{0}'".format(adecl['type'])
                raise _error(self.location[-1], _lineno(adecl), msg)

        # it may specify method name via tags, argument may be QID, we need string
        method = _tagval(adecl, 'method', [None])[0]
        if method: method = str(method)

        # shape can be specified as a rank only or as list of dimensions expressions (strings for now)
        rank = -1
        shape = None
        shapedecl = adecl['shape']
        if shapedecl is None:
            # nothing
            pass
        elif isinstance(shapedecl, int):
            # just a rank
            rank = shapedecl
        elif isinstance(shapedecl, list):
            # list of dimension expressions (strings)
            shape = ','.join(shapedecl)
            rank = len(shapedecl)
        else:
            msg = "Unexpected shape or rank declaration for attribute '{0}'".format(aname)
            raise _error(self.location[-1], _lineno(adecl), msg)

        # optional schema version in tags
        schema_version = self._getIntTag(adecl, 'schema_version', 0)

        # make attribute
        attr = H5Attribute(name = aname, 
                           parent = ds,
                           type = atype,
                           method = method or aname,
                           rank = rank,
                           shape = shape,
                           schema_version = schema_version,
                           tags = _tags(adecl))
        ds.attributes.append(attr)



    def _parseAttr(self, adecl, type):
        ''' Parse defintion of a single data member '''
        
        # check tag names
        self._checktags(adecl, ['public', 'private', 'protected', 'shape_method', 'doc'])

        # every attribute must have a name
        attrname = adecl['name']
        
        # find type object
        atypename = str(adecl['type'])
        atype = type.lookup(atypename, (Type, Enum))
        if not atype: 
            msg = "Failed to resolve member type name '{0}'".format(atypename)
            raise _error(self.location[-1], _lineno(adecl), msg)

        # shape can be specified as a rank only or as list of dimensions expressions (strings for now)
        shape = None
        if adecl['shape'] is not None:
            shape = ','.join(adecl['shape'])

        # it may specify shape_method via tags, argument may be QID, we need string
        shape_method = _tagval(adecl, 'shape_method', [None])[0]
        if shape_method: shape_method = str(shape_method)

        accessor = adecl['method']

        # create attribute
        attr = Attribute( attrname,
                          type = atype,
                          parent = type, 
                          shape = shape, 
                          comment = _doc(adecl), 
                          tags = _tags(adecl),
                          shape_method = shape_method,
                          accessor_name = accessor )
        logging.debug("HddlReader._parseAttr: new attribute: %s", attr)

        # accessor method for it
        if accessor :
            rank = 0
            if adecl['shape']: rank = len(adecl['shape'])
            method = Method(accessor, 
                            attribute = attr, 
                            parent = type, 
                            type = atype,
                            rank = rank,
                            access = _access(adecl),
                            comment = attr.comment)
            attr.accessor = method
            logging.debug("HddlReader._parseAttr: new method: %s", method)

        # get bitfields
        bfoff = 0
        for bfdecl in (adecl['bitfields'] or []):
        
            # check tag names
            self._checktags(bfdecl, ['public', 'private', 'protected', 'doc'])

            size = bfdecl['size']
            bftypename = str(bfdecl['type'])
            bftype = type.lookup(bftypename, (Type, Enum))
            if not bftype: 
                msg = "Failed to resolve bitfield type name '{0}'".format(bftypename)
                raise _error(self.location[-1], _lineno(bfdecl), msg)

            bf = Bitfield(bfdecl['name'], 
                          offset = bfoff, 
                          size = size,
                          parent = attr,
                          type = bftype,
                          comment = _doc(bfdecl))
            logging.debug("HddlReader._parseAttr: new bitfield: %s", bf)
            bfoff += size
            
            accessor = bfdecl['method']
            if accessor :
                method = Method(accessor, 
                                bitfield = bf, 
                                parent = type, 
                                type = bftype,
                                access = _access(bfdecl),
                                comment = bf.comment)
                bf.accessor = method
                logging.debug("HddlReader._parseAttr: new method: %s", method)

    def _parseCtor(self, ctordecl, parent):

        # check tag names
        self._checktags(ctordecl, ['auto', 'inline', 'force_definition', 'external', 'doc'])

    
        # can specify initialization values for some attributes
        attr_init = []
        init2dest = {}
        for initdecl in ctordecl['inits']:

            # resolve destination, must be a member or bitfield
            destname = initdecl['dest']
            dest = [d for d in parent.attributes_and_bitfields() if d.name == destname]
            if not dest: 
                msg = "Failed to resolve destination name '{0}' in constructor initializers list.".format(destname)
                raise _error(self.location[-1], _lineno(ctordecl), msg)
            dest = dest[0]

            # convert expression back to string form
            expr = _constExprToString(initdecl['expr'])
            
            attr_init.append(CtorInit(dest, expr))
            
            # remember it, may be needed later
            init2dest[expr] = dest


        # collect arguments
        args = []
        for argdecl in ctordecl['args']:

            # check tag names
            self._checktags(argdecl, ['method', 'doc'])

            # name and optional type
            argname = argdecl['name']
            atype = None
            if argdecl['type']:
                atype = parent.lookup(str(argdecl['type']), (Type, Enum))
                if not atype: 
                    msg = "Failed to resolve argument type name '{0}' for constructor argument '{1}'".format(argdecl['type'], argname)
                    raise _error(self.location[-1], _lineno(ctordecl), msg)
            
            # destination can be given
            dest = None
            if argdecl['dest']:
                dest = [d for d in parent.attributes_and_bitfields() if d.name == argdecl['dest']]
                if not dest: 
                    msg = "Failed to resolve destination name '{0}' for constructor argument '{1}'".format(argdecl['dest'], argname)
                    raise _error(self.location[-1], _lineno(ctordecl), msg)
                dest = dest[0]
            if not dest:
                # destination can also be given in initializer list, try to find init expression 
                # which is just the name of the argument
                dest = init2dest.get(argname)

            # shape can be specified as a rank only (number or None)
            rank = argdecl['rank']
            if rank is None: rank = -1
            
            # it may specify method via tags
            meth = _tagval(argdecl, 'method', [None])[0]
            if meth:
                meth = parent.lookup(str(meth), Method)
                if not meth: 
                    msg = "Failed to resolve method name '{0}' for constructor argument '{1}'".format(meth, argname)
                    raise _error(self.location[-1], _lineno(ctordecl), msg)
            elif dest:
                meth = dest.accessor
            
#             if not meth: 
#                 msg = "Constructor argument requires to have either a method name ([[method()]]) or destination with accessor method, for constructor argument '{0}'".format(argname)
#                 raise _error(self.location[-1], _lineno(ctordecl), msg)

            # rank is not used yet
            args.append(CtorArg(argname, dest, atype, meth, None))


        ctor = Constructor(parent, 
                           args = args,
                           attr_init = attr_init,
                           tags = _tags(ctordecl),
                           comment = _doc(ctordecl))
        logging.debug("HddlReader._parseCtor: new constructor: %s", ctor)

                        
    def _parseMeth(self, methdecl, type):
    
        # check tag names
        self._checktags(methdecl, ['inline', 'external', 'language', 'doc'])

        # every method must have a name
        name = methdecl['name']
        
        # find type object
        mtype = None
        typename = str(methdecl['type'])
        if typename != 'void':
            mtype = type.lookup(typename, (Type, Enum))
            if not mtype:
                msg = "Failed to resolve method return type '{0}'".format(typename)
                raise _error(self.location[-1], _lineno(methdecl), msg)

        args = []
        for argdecl in methdecl['args']:
            argname = argdecl['name']
            typename = str(argdecl['type'])
            atype = type.lookup(typename, (Type, Enum))
            if not atype: 
                msg = "Failed to resolve argument type name '{0}' for method argument '{1}'".format(typename, argname)
                raise _error(self.location[-1], _lineno(methdecl), msg)
            arank = argdecl['rank']
            args.append((argname, atype))

        codes = {}
        if methdecl['bodies']:
            for i, codeblock in enumerate(methdecl['bodies']):

                # check tag names
                self._checktags(codeblock, ['language', 'doc'])
                
                lang = _tagval(codeblock, 'language')
                if lang is None and i == 0:
                    # for first code block tags are merged with method tags
                    lang = _tagval(methdecl, 'language')
                if lang: lang = lang[0]
                if not lang: lang = 'Any'
                codes[lang] = codeblock['code']

        method = Method(name, 
                        parent = type, 
                        type = mtype,
                        rank = methdecl['rank'],
                        args = args,
                        expr = {},
                        code = codes,
                        tags = _tags(methdecl),
                        comment = _doc(methdecl))
        logging.debug("HddlReader._parseMeth: new method: %s", method)

                        
    def _parseConstant(self, decl, parent, included):
        ''' Parse constant declaration '''

        # check tag names
        self._checktags(decl, ['doc'])

        cname = decl['name']  # IDENTIFIER
        ctype = decl['type']  # IDENTIFIER
        cval = decl['value']  # const_expr
        
        # check type
        if ctype not in _intTypes:
            msg = "Constant has unexpected type '{0}', only integer types are supported now.".format(ctype)
            raise _error(self.location[-1], _lineno(decl), msg)
        
        # convert it back to string
        cval = decl['value_str']
        Constant(cname, cval, parent, included=included, comment = _doc(decl))
            
    def _parseEnum(self, decl, parent, included):
        ''' Parse enum declaration '''
        
        # check tag names
        self._checktags(decl, ['doc'])

        enum = Enum(decl['name'],
                    parent, 
                    base=decl.get('type', 'int32_t'), 
                    included=included, 
                    comment = _doc(decl))

        for cdecl in decl['constants']:
            Constant(cdecl['name'], cdecl['value_str'], enum, comment = _doc(cdecl))

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

    def _getTypeId(self, typedict):
        ''' Get type_id and version number from tags '''

        tags = [tag for tag in typedict['tags'] if tag['name'] == 'type_id']
        if len(tags) > 1:
            msg = "More than one type_id tags defined for type '{0}'".format(typedict['name'])
            raise _error(self.location[-1], _lineno(typedict), msg)
        if tags:
            tag = tags[0]
            args = tag['args']
            if args is None or len(args) != 2:
                msg = "type_id tag requires exactly two arguments"
                raise _error(self.location[-1], _lineno(tag), msg)
            
            if not isinstance(args[0], QID):
                msg = "first argument to type_id() must be an identifier"
                raise _error(self.location[-1], _lineno(tag), msg)
            if not isinstance(args[1], int):
                msg = "second argument to type_id() must be a number"
                raise _error(self.location[-1], _lineno(tag), msg)
            
            return str(args[0]), args[1]
            
        return None, None

     
    def _getIntTag(self, decl, tagname, default = None):
        ''' Get integer value from tags '''

        tags = [tag for tag in decl['tags'] if tag['name'] == tagname]
        if len(tags) > 1:
            msg = "More than one '{0}' tags defined in declaration of '{1}'".format(tagname, decl['name'])
            raise _error(self.location[-1], _lineno(decl), msg)
        if tags:
            tag = tags[0]
            args = tag['args']
            if args is None or len(args) != 1:
                msg = "{0}() tag requires exactly one argument".format(tagname)
                raise _error(self.location[-1], _lineno(tag), msg)
            
            if not isinstance(args[0], int):
                msg = "argument to {0}() must be a number".format(tagname)
                raise _error(self.location[-1], _lineno(tag), msg)

            return args[0]
            
        return default


    def _checktags(self, decl, allowed):
        ''' 
        check all tags against the list of allowed tags
        '''
        for tag in decl['tags']:
            if tag['name'] not in allowed:
                msg = "Unexpected tag name: {0}".format(tag['name'])
                raise _error(self.location[-1], _lineno(tag), msg)

    
    def _getConfigTypes(self, typedict, pkg):
        ''' Get values of config() tags as a list of config type objects '''

        cfgtypes = []
        tags = [tag for tag in typedict['tags'] if tag['name'] == 'config']
        for tag in tags:
            args = tag['args']
            if not args:
                msg = "config() tag requires one or more arguments"
                raise _error(self.location[-1], _lineno(tag), msg)
            for cfg in args:
                if not isinstance(cfg, QID):
                    msg = "arguments to config() tag must be qualified identifiers"
                    raise _error(self.location[-1], _lineno(tag), msg)
                cfgtype = pkg.lookup(str(cfg), Type)
                if not cfgtype:
                    if not self.parseDevelTypes and self._isDevelType(str(cfg), pkg):
                        print >>sys.stderr, "Warning: %s type=%s, DEVEL type=%s in config list is being omitted" % (pkg, typedict['name'], cfg)
                        continue
                    msg = "Failed to resolve name of a config type '{0}'".format(cfg)
                    raise _error(self.location[-1], _lineno(tag), msg)
                cfgtypes.append(cfgtype)
        return cfgtypes
     
     
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )

