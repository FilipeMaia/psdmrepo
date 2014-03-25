#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module DdlHdf5DataHelpers...
#
#------------------------------------------------------------------------

"""Bunch of helper classes for implementation of the HDF5 backend.

This software was developed for the LCLS project.  If you use all or 
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
import os
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import jinja2 as ji
from psddl.Enum import Enum
from psddl.Template import Template as T
from psddl.TemplateLoader import TemplateLoader

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# jinja environment
_jenv = ji.Environment(loader=TemplateLoader(), trim_blocks=True,
                       line_statement_prefix='$', line_comment_prefix='$$')

def _TEMPL(template):
    return _jenv.get_template('hdf5.tmpl?'+template)

_log = logging.getLogger("DdlHdf5DataHelpers")

def _dsFactory(schema, ds, psana_ns):
    '''Factory method for dataset types'''
    if ds.attributes: return DatasetCompound(schema, ds, psana_ns)
    return DatasetRegular(schema, ds, psana_ns)

def _interpolate(expr, typeobj):
    
    expr = expr.replace('{xtc-config}.', 'm_cfg->')
    expr = expr.replace('@config.', 'm_cfg->')
    expr = expr.replace('{type}.', typeobj.name+"::")
    expr = expr.replace('@type.', typeobj.name+"::")
    expr = expr.replace('{self}.', "this->")
    expr = expr.replace('@self.', "this->")
    return expr

def _typedecl(type):
    typename = type.fullName('C++')
    if not type.basic : typename = "const "+typename+'&'
    return typename

def _argdecl(name, type):    
    return _typedecl(type) + ' ' + name

def _types(type):
    """Generator for the type list of the type plus all it bases"""
    if type:
        for t in _types(type.base): yield t
        yield type

def _h5ds_typename(attr):
    '''Return type name for a dataset structure of the attribute'''
    aschema = attr.h5schema()
    if aschema: return aschema.datasets[0].classNameNs()

#------------------------
# Exported definitions --
#------------------------

class DSAttribute(object):

    def __init__(self, attr, psana_ns):
        '''Constructor takes an instance of H5Attribute class'''
        self.attr = attr
        self.psana_ns = psana_ns

    def storage_decl(self):
        '''Some attributes need special storage inside The schema class, this method 
        returns a list of declarations for this storage'''  

        if not self.attr.type.basic:
            typename = self.attr.type.fullName('C++', self.psana_ns)
            dsName = self.attr.parent.name
            attrName = self.attr.name
            if self.attr.type.value_type:
                if self.attr.rank == 0:
                    return [T("mutable $typename m_ds_storage_${dsName}_${attrName};")(locals())]
                else:
                    rank = self.attr.rank
                    return [T("mutable ndarray<const $typename, $rank> m_ds_storage_${dsName}_${attrName};")(locals())]
            else:
                if self.attr.rank == 0:
                    return [T("mutable boost::shared_ptr<$typename> m_ds_storage_${dsName}_${attrName};")(locals())]
                else:
                    attr_class = self.attr.h5schema().className()
                    rank = self.attr.rank
                    return [T("mutable ndarray<$attr_class, $rank> m_ds_storage_${dsName}_${attrName};")(locals())]

        # by default don't need any special storage
        return []
    
    def ds_attr_decl(self):
        '''Returns list of declarations for this attribute in a dataset structure'''

        # base type
        attr = self.attr
        aname = attr.name
        if not attr.stor_type.basic:
            atype = _h5ds_typename(attr)
        else:
            atype = attr.stor_type.name

        decls = []
        if attr.rank == 0:
            decls += [T('$atype $aname;')(locals())]
        else:
            if atype == 'char':
                if attr.sizeIsVlen():
                    decls += [T('$atype* $aname;')(locals())]
                else:
                    size = attr.shape.size()
                    decls += [T('$atype $aname[$size];')(locals())]
            else:
                if attr.sizeIsVlen(): 
                    decls += [T('size_t vlen_$aname;')(locals()), T('$atype* $aname;')(locals())]
                elif attr.sizeIsConst():
                    dims = attr.shape.decl()
                    decls += [T('$atype $aname$dims;')(locals())]
                else:
                    decls += [T('$atype* $aname;')(locals())]
                    
        if attr.external:
            # need to declare special initializer method for these
            decls += [T('void init_attr_$name();')[attr]]
            
        return decls


    def ds_attr_init(self):
        '''Returns list of initializers for attribute in dataset constructor'''

        # base type
        attr = self.attr
        if not attr.stor_type.basic:
            atype = _h5ds_typename(attr)
        else:
            atype = attr.stor_type.name

        # initializer
        dst = attr.name
        if not attr.external:
            if attr.rank == 0:
                src = 'psanaobj.' + attr.method + '()'
                return [T('$dst($src)')(locals())]
            else:
                if atype == 'char':
                    if attr.sizeIsVlen():
                        return [T('$dst(0)')(locals())]
                else:
                    if attr.sizeIsVlen():
                        return [T('vlen_$dst(0)')(locals()), T('$dst(0)')(locals())]
                    elif not attr.sizeIsConst():
                        return [T('$dst(0)')(locals())]
        return []
                        
    def ds_attr_initcode(self):
        '''Returns list of statements to initialize attribute in dataset constructor'''

        attr = self.attr

        if attr.external:
            return [T("  init_attr_$name();")[attr]]

        # base type
        if not attr.stor_type.basic:
            atype = _h5ds_typename(attr)
        else:
            atype = attr.stor_type.name

        # initializer
        src_method = attr.method
        dst = attr.name
        dst_type = atype
        if attr.rank > 0:
            dst_size = str(attr.shape.size())
            if atype == 'char':
                if attr.sizeIsVlen():
                    return [_TEMPL('malloc_copy_string').render(locals())]
                else:
                    return [_TEMPL('copy_string').render(locals())]
            elif attr.type.value_type:
                # array of copyable elements
                if attr.sizeIsVlen():
                    return [_TEMPL('copy_vlen_data').render(locals())]
                elif attr.sizeIsConst():
                    dst = dst + '[0]' * (len(attr.shape.dims)-1)
                    return [_TEMPL('copy_data').render(locals())]
                else:
                    return [_TEMPL('malloc_copy_data').render(locals())]
            else:
                # array of abstract elements, templates for now work only for rank=1
                if attr.rank > 1: raise TypeError("Rank > 1 is not supported for arrays of abstract classes")
                if attr.sizeIsVlen():
                    return [_TEMPL('copy_abs_vlen_data').render(locals())]
                elif attr.sizeIsConst():
                    return [_TEMPL('copy_abs_data').render(locals())]
                else:
                    return [_TEMPL('malloc_copy_abs_data').render(locals())]
            
        return []


class DatasetCompound(object):
    '''Helper class for datasets which also need to define a new compound data type'''
    
    def __init__(self, schema, ds, psana_ns):
        '''Arguments must have H5Type and H5Dataset type'''
        self.schema = schema
        self.ds = ds
        self.pstype = schema.pstype
        self.pstypename = self.pstype.fullName('C++', psana_ns)
        self.psana_ns = psana_ns
        self.attributes = [DSAttribute(attr, psana_ns) for attr in self.ds.attributes]
        
    def ds_read_decl(self):
        '''
        Returns the list of declarations for the schema class to support reading of dataset data.
        '''
        dsClassName = self.ds.classNameNs()
        dsName = self.ds.name
        return [T('mutable boost::shared_ptr<$dsClassName> m_ds_$dsName;')(locals()),
                 T('void read_ds_$dsName() const;')(locals())]

    def ds_read_impl(self):
        '''Returns implementation of the method(s) for dataset reading'''
        dsClassName = self.ds.classNameNs()
        schema = self.schema
        type = schema.pstype
        ds = self.ds
        return [_TEMPL('read_compound_ds_method').render(locals())]

    def make_ds_impl(self):
        '''Returns piece of code implementing dataset creation code'''
        return _TEMPL('make_compound_ds').render(ds=self.ds)
        
    def ds_write_impl(self):
        '''Returns piece of code implementing writing of the data to a dataset'''
        return _TEMPL('write_compound_ds').render(ds=self.ds)

    def ds_decltype(self):
        '''Return declaration type of the dataset variable'''
        dsClassName = self.ds.classNameNs()
        return T('boost::shared_ptr<$dsClassName>')(locals())

    def genDs(self, inc, cpp):

        if 'external' in self.ds.tags:
            _log.debug("_genDs: skip dataset - external")
            return

        hds = self

        print >>cpp, self._genH5TypeFunc("stored")
        print >>cpp, self._genH5TypeFunc("native")

        # if schema contains single dataset and corresponding data type is a value type
        # then add conversion function from this dataset class to a data type
        try:
            pstype = self.schema.pstype
            if len(self.schema.datasets) == 1 and pstype.value_type:
                
                # find a constructor with some arguments
                ctors = [ctor for ctor in pstype.ctors if (ctor.args or 'auto' in ctor.tags)]
                if len(ctors) == 1:
                    ctor = ctors[0]
    
                    # map ctor parameters to dataset attributes
                    dsattrs = []
                    for arg in ctor.args:
                        if not arg.method: raise TypeError("Attribute " + arg.dest.name + " does not have access method")
                        attr = [dsattr.name for dsattr in self.ds.attributes if dsattr.method == arg.method.name]
                        if not attr: raise TypeError("Failed to find HDF5 attributes for constructor arguments")
                        attr = attr[0]
                        if isinstance(arg.type, Enum): attr = "{0}({1})".format(arg.type.fullName('C++', self.psana_ns), attr)
                        dsattrs.append(attr)

                    conversion = pstype.fullName('C++', self.psana_ns)
                    cvt_args = ', '.join(dsattrs)

        except Exception, ex:
            # if we fail just ignore it
            _log.debug('_genDs: exception for conv operator: %s', ex)

        # attribute names which declare pointers
        pointers = [attr.name for attr in self.ds.attributes if attr.rank > 0 and not attr.sizeIsConst()]
        vlen_pointers = [attr.name for attr in self.ds.attributes if attr.rank > 0 and attr.type.name != 'char' and attr.sizeIsVlen()]

        print >>inc, _TEMPL('compound_dataset_decl').render(locals())

        # generate constructor and destructor
        print >>cpp, _TEMPL('compound_dataset_ctor_dtor').render(locals())

    def _genH5TypeFunc(self, func):
        """
        Generate native_type()/stored_type() static method for dataset class.
        """

        ns = self.schema.nsName()
        className = self.ds.className()

        attributes = [self._genH5TypeFuncAttr(attr, func) for attr in self.ds.attributes]
        return _TEMPL('compound_dataset_h5type_method').render(locals())


    def _genH5TypeFuncAttr(self, attr, func):
        '''
        Generate attribute data for type-definition function.
        For a given attribute returns dictionary with the following keys:
        'name'  - attribute name (string)
        'type'  - expression which results in a attribute HDF5 type (string)
        'type_decl' - optional string which produces declarations used by type
        '''

        _log.debug("_genH5TypeFuncAttr: attr = %s", attr)
        _log.debug("_genH5TypeFuncAttr: attr.sizeIsVlen() = %s attr.sizeIsConst() = %s", attr.sizeIsVlen(), attr.sizeIsConst())

        attr_name = attr.name
        attr_member = attr.name
        if attr.rank > 0 and attr.type.name != 'char' and attr.sizeIsVlen():
            attr_member = 'vlen_'+attr.name

        if attr.type.basic:
            
            if isinstance(attr.type, Enum):
                
                typename = attr.type.parent.fullName('C++', self.psana_ns)
                constants = [dict(name=c.name, h5name=self.schema.enumConstName(attr.type.name, c.name), type=typename) for c in attr.type.constants()]
                type = '_enum_type_' + attr.name
                enum_base = attr.stor_type.name
                type_decl = _TEMPL('h5type_definition_enum').render(locals())
                if attr.rank > 0:
                    baseType = type
                    type = '_array_type_' + attr.name
                    rank = -1 if attr.sizeIsVlen() else attr.rank
                    if attr.shape is None:
                        raise ValueError('attribute %s is missing shape definition' % attr.name)
                    shape = attr.shape.cs_dims()
                    type_decl += _TEMPL('h5type_definition_array').render(locals())
                return dict(name=attr_name, member=attr_member, type=type, type_decl=type_decl)
            
            elif attr.type.name == 'char':
                
                type_name = attr.type.name
                size = ""
                if attr.rank > 0:
                    type_name = 'const char*'
                    if not attr.sizeIsVlen():
                        size = str(attr.shape.size())
                return dict(name=attr.name, member=attr_member, type=T("hdf5pp::TypeTraits<$type_name>::${func}_type($size)")(locals()))

            else:
                
                type_name = attr.type.name
                type_decl = None
                type=T("hdf5pp::TypeTraits<$type_name>::${func}_type()")(locals())
                if attr.rank > 0:
                    baseType = type
                    type = '_array_type_' + attr.name
                    rank = -1 if attr.sizeIsVlen() else attr.rank
                    if attr.shape is None:
                        raise ValueError('attribute %s is missing shape definition' % attr.name)
                    shape = attr.shape.cs_dims()
                    type_decl = _TEMPL('h5type_definition_array').render(locals())
                return dict(name=attr_name, member=attr_member, type_decl=type_decl, type=type)

        else:

            # for non-basic type (like composite types) find corresponding h5 schema,
            # if it has only one dataset then use it here
            aschema = attr.h5schema()
            if not aschema:
                raise ValueError('No schema found for attribute %s' % attr.name)
            if len(aschema.datasets) != 1:
                raise ValueError('Attribute schema has number of datasets != 1: %d for attr %s of type %s' % (len(aschema.datasets), attr.name, attr.type.name))

            attr_type_name = _h5ds_typename(attr)

            type = T("hdf5pp::TypeTraits<${attr_type_name}>::${func}_type()")(locals())
            type_decl = None
            if attr.rank > 0:
                baseType = type
                type = '_array_type_' + attr.name
                rank = -1 if attr.sizeIsVlen() else attr.rank
                shape = attr.shape.cs_dims()
                type_decl = _TEMPL('h5type_definition_array').render(locals())

            return dict(name=attr_name, member=attr_member, type_decl=type_decl, type=type)


class DatasetRegular(object):
    '''Helper class for datasets which do not need compound data type'''
    
    def __init__(self, schema, ds, psana_ns):
        '''Arguments must have H5Type and H5Dataset type'''
        self.schema = schema
        self.ds = ds
        self.pstype = schema.pstype
        self.pstypename = self.pstype.fullName('C++', psana_ns)
        self.psana_ns = psana_ns
        self.attributes = []

    def _attr_typename(self):
        '''Returns type name for the attribute'''
        if self.ds.type.value_type:
            return self.ds.type.fullName('C++', self.psana_ns)
        else:
            # find a schema for attribute
            aschema = self.ds.h5schema()
            if not aschema: raise ValueError('No schema found for dataset %s' % self.ds.name)
            if len(aschema.datasets) != 1:
                raise ValueError('Attribute schema has number of datasets != 1: %d for attr %s of type %s' % (len(aschema.datasets), self.ds.name, self.ds.type.name))

            return T("${name}_v${version}")[aschema]

    def _attr_dsname(self):
        '''Returns dataset type name for the attribute'''
        if not self.ds.type.basic:
            aschema = self.ds.h5schema()
            if not aschema: raise ValueError('No schema found for dataset %s' % self.ds.name)
            return aschema.datasets[0].classNameNs()

    def ds_read_decl(self):
        _log.debug("ds_read_decl: ds=%s", self.ds)
        dsName = self.ds.name
        rank = self.ds.rank
        typename = self._attr_typename()
        if rank > 0:
            decls = [T("mutable ndarray<const $typename, $rank> m_ds_${dsName};")(locals())]
            decls += [T('void read_ds_$dsName() const;')(locals())]
        else:
            dsClassName = _h5ds_typename(self.ds)
            decls = [T('mutable boost::shared_ptr<$dsClassName> m_ds_$dsName;')(locals()),
                     T('void read_ds_$dsName() const;')(locals())]
            if self.ds.type.value_type:
                decls += [T("mutable $typename m_ds_storage_${dsName};")(locals())]
            else:
                decls += [T("mutable boost::shared_ptr<$typename> m_ds_storage_${dsName};")(locals())]
        return decls

    def ds_decltype(self):
        '''Return declaration type of the dataset variable'''
        rank = self.ds.rank
        if rank > 0:
            typename = self._attr_typename()
            return T("ndarray<const $typename, $rank>")(locals())
        else:
            dsClassName = _h5ds_typename(self.ds)
            return T('mutable boost::shared_ptr<$dsClassName>')(locals())

    def ds_read_impl(self):
        schema = self.schema
        type = schema.pstype
        ds = self.ds
        typename = self._attr_typename()

        if ds.rank > 0:
            if self.ds.type.basic:
                return [_TEMPL('read_array_ds_basic_method').render(locals())]
            elif self.ds.type.value_type:
                ds_struct = self._attr_dsname()
                return [_TEMPL('read_array_ds_udt_method').render(locals())]
            else:
                ds_struct = self._attr_dsname()
                return [_TEMPL('read_array_ds_abstract_method').render(locals())]
        else:
            dsClassName = _h5ds_typename(ds)
            if self.ds.type.value_type:
                return [_TEMPL('read_regular_ds_valuetype_method').render(locals())]
            else:
                return [_TEMPL('read_regular_ds_abstract_method').render(locals())]

    def make_ds_impl(self):
        '''Returns piece of code implementing dataset creation code'''
        ds = self.ds
        rank = ds.rank

        if ds.sizeIsVlen():
            # vlen array
            if ds.type.basic:
                ds_type = ds.type.name
            else:
                ds_type = self._attr_dsname()
            return _TEMPL('make_vlen_ds').render(locals())
        elif rank > 0:
            # non-vlen arrays
            if ds.type.basic:
                return _TEMPL('make_array_ds_basic').render(locals())
            elif ds.type.value_type:
                ds_struct = self._attr_dsname()
                return _TEMPL('make_array_ds_udt').render(locals())
            else:
                ds_struct = self._attr_dsname()
                # need a method which returns data shape
                shape_method = self.ds.shape_method
                if not shape_method: raise TypeError("shape method is required for dataset " + ds.name)
                return _TEMPL('make_array_ds_abstract').render(locals())
        else:
            ds_struct = self._attr_dsname()
            return _TEMPL('make_regular_ds').render(locals())
        
    def ds_write_impl(self):
        '''Returns piece of code implementing writing of the data to a dataset'''
        ds = self.ds
        rank = ds.rank

        if ds.sizeIsVlen():
            if ds.type.basic:
                ds_type = ds.type.name
                return _TEMPL('write_vlen_ds_basic').render(locals())
            elif ds.type.value_type:
                ds_type = self._attr_dsname()
                return _TEMPL('write_vlen_ds_udt').render(locals())
            else:
                ds_type = self._attr_dsname()
                # need a method which returns data shape
                shape_method = self.ds.shape_method
                if not shape_method: raise TypeError("shape method is required for dataset " + ds.name)
                return _TEMPL('write_vlen_ds_abstract').render(locals())
        elif rank > 0:
            zero_dims = 'zero_dims' in self.ds.tags
            if ds.type.basic:
                return _TEMPL('write_array_ds_basic').render(locals())
            elif ds.type.value_type:
                ds_struct = self._attr_dsname()
                return _TEMPL('write_array_ds_udt').render(locals())
            else:
                ds_struct = self._attr_dsname()
                # need a method which returns data shape
                shape_method = self.ds.shape_method
                if not shape_method: raise TypeError("shape method is required for dataset " + ds.name)
                return _TEMPL('write_array_ds_abstract').render(locals())
        else:
            ds_struct = self._attr_dsname()
            return _TEMPL('write_regular_ds').render(locals())

    def genDs(self, inc, cpp):

        pass

class SchemaType(object):
    '''
    Base type for schema types below.
    '''
    def __init__(self, schema, psana_ns):
        '''schema parameter is of type H5Type'''
        self.schema = schema
        self.pstype = schema.pstype
        self.pstypename = self.pstype.fullName('C++', psana_ns)
        self.psana_ns = psana_ns
        self.datasets = [_dsFactory(schema, ds, psana_ns) for ds in schema.datasets]
        

class SchemaValueType(SchemaType):
    '''
    All stuff needed for generation of code for value-types.
    '''
    def __init__(self, schema, psana_ns):
        '''schema parameter is of type H5Type'''
        SchemaType.__init__(self, schema, psana_ns)

    # generator for all HFD5 attributes
    def _schemaAttributes(self):
        for ds in self.schema.datasets:
            for dsattr in ds.attributes:
                yield ds, dsattr
        
    def genSchema(self, inc, cpp):
        """Generate code for value types"""

        if 'embedded' in self.schema.tags: return
        
        hschema = self
        _log.debug("_genValueType: type=%r", self.pstype)

        # find a constructor with some arguments
        ctors = [ctor for ctor in self.pstype.ctors if (ctor.args or 'auto' in ctor.tags)]
        if not ctors: raise TypeError("Type " + self.pstype.name + " does not have constructor defined")
        if len(ctors) > 1: raise TypeError("Type " + self.pstype.name + " has multiple constructors defined")
        ctor = ctors[0]

        # map ctor parameters to schema objects
        dsattrs = []
        _log.debug("_genValueType: ctor args=%r", ctor.args)
        for arg in ctor.args:
            if not arg.method: raise TypeError("Attribute " + arg.dest.name + " does not have access method")
            ds_attr = [(ds, dsattr) for ds, dsattr in self._schemaAttributes() if arg.method.name == dsattr.method]
            if not ds_attr:
                raise TypeError("Failed to find HDF5 attribute for constructor argument %s/%s in type %s" % (arg.name, arg.method.name, self.pstype.name))
            else:
                dsattrs += ds_attr[:1]

        args = []
        for ds, dsattr in dsattrs:
            expr = 'ds_{0}->{1}'.format(ds.name, dsattr.name)
            if isinstance(dsattr.type, Enum):
                expr = '{0}({1})'.format(dsattr.type.fullName('C++', self.psana_ns), expr)
            args.append(expr)

        print >>inc, _TEMPL('proxy_valtype_declaration').render(locals())
        print >>cpp, _TEMPL('proxy_valtype_impl_getTypedImpl').render(locals())


        print >>cpp, _TEMPL('schema_store_impl').render(locals())
        
class SchemaAbstractType(SchemaType):
    '''
    All stuff needed for generation of code for value-types.
    '''
    def __init__(self, schema, psana_ns):
        '''schema parameter is of type H5Type'''
        SchemaType.__init__(self, schema, psana_ns)

    def genSchema(self, inc, cpp):
        """Generate code for abstract types"""
        
        hschema = self
        className = '{0}_v{1}'.format(self.pstype.name, self.schema.version)
        
        _log.debug("_genAbsType: type=%s", repr(type))

        cpp_code = []

        # declarations for public methods 
        methods = []
        for t in _types(self.pstype):
            for meth in t.methods(): 
                if meth.access == 'public': 
                    decl, impl = self._genMethod(meth)
                    methods += decl
                    cpp_code += impl
            # generate _shape() methods for array attributes
            for attr in t.attributes() :
                decl, impl = self._genAttrShapeDecl(attr)
                methods += decl
                cpp_code += impl

        for ds in self.datasets:
            cpp_code += ds.ds_read_impl()

        # explicitely instantiate class with known config types
        for config in self.pstype.xtcConfig:
            cfgClassName = config.fullName('C++', self.psana_ns)
            cpp_code += [T("template class $className<$cfgClassName>;")(locals())]

        # may also provide a constructor which takes dataset data
        if len(self.datasets) == 1:
            ds = self.datasets[0]
            decltype = ds.ds_decltype()
            dsName = ds.ds.name
            dsCtorWithArg = T('(const ${decltype}& ds) : m_ds_${dsName}(ds) {}')(locals())

        print >>inc, _TEMPL('abstract_type_declaration').render(locals())
        for line in cpp_code:
            print >>cpp, line

        print >>cpp, _TEMPL('schema_store_impl').render(locals())
        

    def _genAttrShapeDecl(self, attr):

        if not attr.shape_method: return [], [] 
        if not attr.accessor: return [], []
        
        doc = "Method which returns the shape (dimensions) of the data returned by %s() method." % \
                attr.accessor.name
        
        # value-type arrays return ndarrays which do not need shape method
        if attr.type.value_type and attr.type.name != 'char': return [], []

        shape = [str(s or -1) for s in attr.shape.dims]

        body = _TEMPL('shape_method_impl').render(locals());
        body = _interpolate(body, attr.parent)

        return self._genMethodBody(attr.shape_method, "std::vector<int>", body, [], inline=False, doc=doc)

    def _method2ds(self, meth):
        """Map method to a dataset and attribute"""
        for ds in self.schema.datasets:
            for attr in ds.attributes:
                if attr.method == meth.name:
                    return (ds, attr)
            if ds.method == meth.name:
                return ds, None
        return (None, None)


    def _genMethod(self, meth):
        """Generate method declaration and definition, returns tuple of lists of strings"""


        if meth.name == "_sizeof" : return [], []

        _log.debug("_genMethod: meth: %s", meth)
        
        decl = []
        impl = []

        schema = self.schema
        type = schema.pstype
        psanatypename = type.fullName('C++', self.psana_ns)
        className = '{0}_v{1}'.format(type.name, schema.version)

        # determine dataset and attribute which holds data for this method
        # for non-compound datasets (datasets without attributes) attr will be None
        ds, attr = self._method2ds(meth)
        _log.debug("_genMethod: h5ds: %s, h5attr: %s, schema: %s", ds, attr, self.schema)

        if attr :
            
            # data is stored in a dataset
            args = []
            attr_type = attr.type.fullName('C++', self.psana_ns)
            ret_type = meth.type.fullName('C++', self.psana_ns)
            rank = attr.rank
            if attr.rank:
                if meth.type.name == 'char':
                    ret_type = "const char*"
                    args = [('i%d'%i, type.lookup('uint32_t')) for i in range(attr.rank-1)]
                elif meth.type.basic or meth.type.value_type:
                    attr_type = attr.stor_type.fullName('C++', self.psana_ns)
                    ret_type = T("ndarray<const $attr_type, $rank>")(locals())
                else:
                    args = [('i%d'%i, type.lookup('uint32_t')) for i in range(attr.rank)]
                    ret_type = T("const ${ret_type}&")(locals())
            elif not attr.type.basic:
                ret_type = T("const ${ret_type}&")(locals())
                
            meth_name = meth.name
            argdecl = ", ".join(["%s %s" % (atype.name, arg) for arg, atype in args])
            decl += [T("virtual $ret_type $meth_name($argdecl) const;")(locals())]
            
            if attr.rank == 0:
                
                if attr.type.basic:
                    
                    # simplest case, basic type, non-array
                    impl = _TEMPL('attr_access_method_basic').render(locals())

                elif attr.type.value_type:
                    
                    # non-array, but complex type, need to convert from HDF5 type to psana,
                    # store the result in member variable so that we can return reference to it
                    attr_type = attr.type.fullName('C++', self.psana_ns)
                    impl = _TEMPL('attr_access_method_valtype').render(locals())
                    
                else:
                    
                    # non-array, but complex type, need to convert from HDF5 type to psana,
                    # store the result in member variable so that we can return reference to it
                    aschema = attr.h5schema()
                    attr_type = _h5ds_typename(attr)
                    attr_class = "%s_v%d" % (aschema.pstype.fullName('C++'), aschema.version)
                    impl = _TEMPL('attr_access_method_abstype').render(locals())
               
            else:
                
                # non-zero rank, stored in a dataset

                if attr.type.name == 'char':
                    
                    # character array
                    # TODO: if rank is >1 then methods must provide arguments for all but first index,
                    # this is not implemented yet.
                    impl = _TEMPL('attr_access_method_array_char').render(locals())
                    
                elif attr.type.basic:
                    
                    # array of basic types, return ndarray, data is shared with the dataset
                    if attr.sizeIsVlen():
                        # VLEN array take dimension from VLEN size, currently means that only 1-dim VLEN arrays are supported
                        shape = T("m_ds_$dsName->vlen_$attrName")(dsName=ds.name, attrName=attr.name)
                    else:
                        shape = _interpolate(str(attr.shape), type)
                        attr_index = '[0]' * (len(attr.shape.dims)-1)
                    impl = _TEMPL('attr_access_method_array_basic').render(locals())
                    
                elif attr.type.value_type:
                    
                    # array of non-basic value type, have to convert
                    if attr.sizeIsVlen():
                        # VLEN array take dimension from VLEN size, currently means that only 1-dim VLEN arrays are supported
                        shape = T("m_ds_$dsName->vlen_$attrName")(dsName=ds.name, attrName=attr.name)
                    else:
                        shape = _interpolate(str(attr.shape), type)
                    impl = _TEMPL('attr_access_method_array_valtype').render(locals())
                    
                else:
                    
                    # array of non-basic abstract type, have to convert
                    if attr.sizeIsVlen():
                        # VLEN array take dimension from VLEN size, currently means that only 1-dim VLEN arrays are supported
                        shape = T("m_ds_$dsName->vlen_$attrName")(dsName=ds.name, attrName=attr.name)
                        data_size = shape
                    else:
                        shape = _interpolate(str(meth.attribute.shape), type)
                        data_size = _interpolate(str(meth.attribute.shape.size()), type)
                    arguse = ''.join(["[%s]" % arg for arg, _ in args])
                    attr_class = attr.h5schema().className()
                    attr_type = _h5ds_typename(attr)
                    impl = _TEMPL('attr_access_method_array_abstype').render(locals())

            return decl, [impl]
            
        elif ds:
            
            # non-compound dataset 

            args = []
            ds_type = ds.type.fullName('C++', self.psana_ns)
            ret_type = ds_type
            rank = ds.rank
            if rank:
                if ds.type.name == 'char':
                    ret_type = "const char*"
                    args = [('i%d'%i, type.lookup('uint32_t')) for i in range(rank-1)]
                elif ds.type.basic or ds.type.value_type:
                    ret_type = T("ndarray<const $ds_type, $rank>")(locals())
                else:
                    args = [('i%d'%i, type.lookup('uint32_t')) for i in range(rank)]
                    ret_type = T("const ${ret_type}&")(locals())
            elif not ds.type.basic:
                ret_type = T("const ${ret_type}&")(locals())
                
            meth_name = meth.name
            argdecl = ", ".join(["%s %s" % (atype.name, arg) for arg, atype in args])
            decl += [T("virtual $ret_type $meth_name($argdecl) const;")(locals())]
            
            if rank == 0:
                
                if ds.type.value_type:
                    
                    # non-array, value type, need to convert from HDF5 type to psana,
                    # store the result in member variable so that we can return reference to it
                    ds_type = ds.type.fullName('C++', self.psana_ns)
                    impl = _TEMPL('ds_access_method_valtype').render(locals())
                    
                else:
                    
                    # non-array, but complex type, need to convert from HDF5 type to psana,
                    # store the result in member variable so that we can return reference to it
                    aschema = ds.h5schema()
                    ds_class = "%s_v%d" % (aschema.pstype.fullName('C++'), aschema.version)
                    impl = _TEMPL('ds_access_method_abstype').render(locals())
               
               
            else:
                
                # non-zero rank, stored as a dataset

                if ds.type.value_type and not isinstance(ds.type, Enum):

                    # array of value type, have to convert
                    impl = _TEMPL('ds_access_method_array_valtype').render(locals())
                    
                else:
                    
                    # array of non-basic abstract type, have to convert
                    arguse = ''.join(["[%s]" % arg for arg, _ in args])
                    ds_class = ds.h5schema().className()
                    ds_type = _h5ds_typename(ds)
                    impl = _TEMPL('ds_access_method_array_abstype').render(locals())

            return decl, [impl]
            
        else:
            
            # data is not stored, if method defines body then use it, otherwise skip
            # definition but declare it anyway
            
            # if no return type given then it does not return anything
            rettype = meth.type
            if rettype is None:
                rettype = "void"
            else:
                rettype = rettype.fullName('C++', self.psana_ns)
                if meth.rank > 0:
                    rettype = "ndarray<const %s, %d>" % (rettype, meth.rank)

            # make method body
            body = meth.code.get("C++")
            if not body : body = meth.code.get("Any")
            if not body :
                expr = meth.expr.get("C++")
                if not expr : expr = meth.expr.get("Any")
                if expr:
                    body = expr
                    if rettype: body = "return %s;" % expr
                
            # config objects may be needed 
            cfgNeeded = False
            if body: 
                body = _interpolate(body, meth.parent)

            # default is not inline, can change with a tag
            inline = 'inline' in meth.tags
            
            return self._genMethodBody(meth.name, rettype, body, meth.args, inline, static=meth.static, doc=meth.comment)



    def _genMethodBody(self, methname, rettype, body, args=[], inline=False, static=False, doc=None):
        """ Generate method, both declaration and definition, given the body of the method.
        Returns tuple of lists of strings."""
        
        # make argument list
        argsspec = ', '.join([_argdecl(*arg) for arg in args])
        schema = self.schema
        type = schema.pstype

        return ([_TEMPL('method_declaration').render(locals())], [_TEMPL('method_definition').render(locals())])

        
def Schema(schema, psana_ns):
    '''Factory method that returns instance of one of the above classes,
    accepts instance of H5Type type.'''
    if schema.pstype.value_type:
        return SchemaValueType(schema, psana_ns)
    else:
        return SchemaAbstractType(schema, psana_ns)
    


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
