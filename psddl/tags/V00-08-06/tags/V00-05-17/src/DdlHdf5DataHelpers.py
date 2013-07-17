#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module DdlHdf5DataHelpers...
#
#------------------------------------------------------------------------

"""Bunch of helper classes for implementation of the HDF5 generator

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

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_ds_comp_decl_template = ji.Template('''
namespace {{ns}} {
struct {{className}} {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  {{className}}();
  ~{{className}}();

{% for attr in attributes %}
{% if attr.vlen %}
  size_t vlen_{{attr.name}};
{% endif %}
  {{attr.type}} {{attr.name}}; 
{% endfor %}

{% if conversion %}
  operator {{conversion}}() const { return {{conversion}}({{cvt_args}}); }
{% endif %}
};
}
''', trim_blocks=True)


_ds_comp_type_method_template = ji.Template('''
hdf5pp::Type {{ns}}_{{className}}_{{func}}_type()
{
  typedef {{ns}}::{{className}} DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
{% for attr in attributes %}
{% if attr.type_decl %}
{{attr.type_decl}}
{%- endif %}
  type.insert("{{attr.name}}", offsetof(DsType, {{attr.member}}), {{attr.type}});
{% endfor %}
  return type;
}

hdf5pp::Type {{ns}}::{{className}}::{{func}}_type()
{
  static hdf5pp::Type type = {{ns}}_{{className}}_{{func}}_type();
  return type;
}
''', trim_blocks=True)


_ds_ctor_dtor_template = ji.Template('''\
{{ns}}::{{className}}::{{className}}()
{
{% for attr in vlen_pointers %}
  this->vlen_{{attr}} = 0;
{% endfor %}
{% for attr in pointers %}
  this->{{attr}} = 0;
{% endfor %}
}
{{ns}}::{{className}}::~{{className}}()
{
{% for attr in pointers %}
  free(this->{{attr}});
{% endfor %}
}
''', trim_blocks=True)


_enum_h5type_definition = ji.Template('''\
  hdf5pp::EnumType<int32_t> {{type}} = hdf5pp::EnumType<int32_t>::enumType();
{% for c in constants %}
  {{type}}.insert("{{c.name}}", {{c.type}}::{{c.name}});
{% endfor %}
''', trim_blocks=True)

_array_h5type_definition = ji.Template('''\
{% if rank > 0 %}
  hsize_t {{type}}_shape[] = { {{shape}} };
  hdf5pp::ArrayType {{type}} = hdf5pp::ArrayType::arrayType({{baseType}}, {{rank}}, {{type}}_shape);
{% else %}
  hdf5pp::VlenType {{type}} = hdf5pp::VlenType::vlenType({{baseType}});
{% endif %}
''', trim_blocks=True)


_valtype_proxy_decl = ji.Template('''\
class Proxy_{{schema.name}}_v{{schema.version}} : public PSEvt::Proxy<{{psanatypename}}> {
public:
  typedef {{psanatypename}} PsanaType;

  Proxy_{{schema.name}}_v{{schema.version}}(hdf5pp::Group group, hsize_t idx) : m_group(group), m_idx(idx) {}
  virtual ~Proxy_{{schema.name}}_v{{schema.version}}() {}

protected:

  virtual boost::shared_ptr<PsanaType> getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<PsanaType> m_data;
};
''', trim_blocks=True)

_getTypedImpl_template = ji.Template('''\
boost::shared_ptr<{{psanatypename}}>
Proxy_{{schema.name}}_v{{schema.version}}::getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key)
{
  if (not m_data) {
{% for ds in schema.datasets %}
    boost::shared_ptr<{{ds.classNameNs()}}> ds_{{ds.name}} = hdf5pp::Utils::readGroup<{{ds.classNameNs()}}>(m_group, "{{ds.name}}", m_idx);
{% endfor %}
{% if 1 %}
    m_data.reset(new PsanaType({{args|join(', ')}}));
{% else %}
    m_data = boost::make_shared<PsanaType>({{args|join(', ')}});
{% endif %}
  }
  return m_data;
}

''', trim_blocks=True)


_shape_method_body_template = ji.Template('''\
  int shape[] = { {{shape|join(', ')}} };
  return std::vector<int>(shape, shape+{{shape|length}});
''', trim_blocks=True)


_abs_type_decl = ji.Template('''
{% set className = type.name ~ '_v' ~ schema.version %}

{% if type.xtcConfig %}
template <typename Config>
{% endif %}
class {{className}} : public {{psanatypename}} {
public:
  typedef {{psanatypename}} PsanaType;
  {{className}}() {}
{% if type.xtcConfig %}
  {{className}}(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}
{% else %}
  {{className}}(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
{% endif %}
{% if dsCtorWithArg %}
  {{className}}{{dsCtorWithArg}}
{% endif %}
  virtual ~{{className}}() {}
{% for meth in methods if meth %}
  {{meth}}
{% endfor %}
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
{% if type.xtcConfig %}
  boost::shared_ptr<Config> m_cfg;
{% endif %}
{% for decl in ds_read_decl if decl %}
  {{decl}}
{% endfor %}
};

''', trim_blocks=True)


_meth_decl = ji.Template('''\
{% if doc %}
/** {{doc}} */
{% endif %}
  {% if static %}static{% endif %}
{{rettype}} {{methname}}({{argsspec}}) 
{%- if not static %} const{% endif %}
{%- if inline and body %}
 { {{body}} }
{%- else %}
;
{% endif -%}
''', trim_blocks=True)


_meth_def = ji.Template('''\
{% set classname = type.name ~ '_v' ~ schema.version %}
{% if body and not inline -%}
{% if template %}
template <typename {{template}}>
{% endif %}
{{rettype}}
{{classname}}{% if template %}<{{template}}>{% endif %}::{{methname}}({{argsspec}})
{%- if not static %} const{% endif -%}
{ 
{{body}} 
}
{%- endif -%}
''', trim_blocks=True)

_read_comp_ds_impl_template = ji.Template('''\
{% set classname = type.name ~ '_v' ~ schema.version %}
{% if type.xtcConfig %}
{% set TMPL = '<Config>' %}
template <typename Config>
{% endif %}
void {{classname}}{{TMPL}}::read_ds_{{dsName}}() const {
  m_ds_{{dsName}} = hdf5pp::Utils::readGroup<{{dsClassName}}>(m_group, "{{dsName}}", m_idx);
}
''', trim_blocks=True)

_read_arr_ds_basic_impl_template = ji.Template('''\
{% set classname = type.name ~ '_v' ~ schema.version %}
{% if type.xtcConfig %}
{% set TMPL = '<Config>' %}
template <typename Config>
{% endif %}
void {{classname}}{{TMPL}}::read_ds_{{dsName}}() const {
  m_ds_{{dsName}} = hdf5pp::Utils::readNdarray<{{typename}}, {{rank}}>(m_group, "{{dsName}}", m_idx);
}
''', trim_blocks=True)


_read_arr_ds_udf_impl_template = ji.Template('''\
{% set classname = type.name ~ '_v' ~ schema.version %}
{% if type.xtcConfig %}
{% set TMPL = '<Config>' %}
template <typename Config>
{% endif %}
void {{classname}}{{TMPL}}::read_ds_{{dsName}}() const {
  ndarray<{{ds_struct}}, {{rank}}> arr = hdf5pp::Utils::readNdarray<{{ds_struct}}, {{rank}}>(m_group, "{{dsName}}", m_idx);
  ndarray<{{typename}}, {{rank}}> tmp(arr.shape());
  std::copy(arr.begin(), arr.end(), tmp.begin());
  m_ds_{{dsName}} = tmp;
}
''', trim_blocks=True)


_read_reg_ds_value_impl_template = ji.Template('''\
{% set classname = type.name ~ '_v' ~ schema.version %}
{% if type.xtcConfig %}
{% set TMPL = '<Config>' %}
template <typename Config>
{% endif %}
void {{classname}}{{TMPL}}::read_ds_{{dsName}}() const {
  m_ds_{{dsName}} = hdf5pp::Utils::readGroup<{{dsClassName}}>(m_group, "{{dsName}}", m_idx);
  m_ds_storage_{{dsName}} = *m_ds_{{dsName}};
}
''', trim_blocks=True)


_read_reg_ds_abs_impl_template = ji.Template('''\
{% set classname = type.name ~ '_v' ~ schema.version %}
{% if type.xtcConfig %}
{% set TMPL = '<Config>' %}
template <typename Config>
{% endif %}
void {{classname}}{{TMPL}}::read_ds_{{dsName}}() const {
  m_ds_{{dsName}} = hdf5pp::Utils::readGroup<{{dsClassName}}>(m_group, "{{dsName}}", m_idx);
  m_ds_storage_{{dsName}} = boost::make_shared<{{typename}}>(m_ds_{{dsName}});
}
''', trim_blocks=True)



_log = logging.getLogger("DdlHdf5DataHelpers")

def _isArrayDs(ds):
    '''ds type should be H5Dataset, returns true if dataset should have array type'''
    if not ds.attributes:
        if ds.rank > 0 and ds.type.name != 'char' and not ds.sizeIsConst() and not ds.sizeIsVlen():
            return True
    return False

def _dsFactory(schema, ds):
    if ds.attributes: return DatasetCompound(schema, ds)
    return DatasetRegular(schema, ds)

def _interpolate(expr, typeobj):
    
    expr = expr.replace('{xtc-config}.', 'm_cfg->')
    expr = expr.replace('{type}.', typeobj.name+"::")
    expr = expr.replace('{self}.', "this->")
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

class DatasetCompound(object):
    '''Helper class for datasets which also need to define a new compound data type'''
    
    def __init__(self, schema, ds):
        '''Arguments must have H5Type and H5Dataset type'''
        self.schema = schema
        self.ds = ds
        
    def ds_read_decl(self, psana_ns):
        dsClassName = self.ds.classNameNs()
        dsName = self.ds.name
        decls = [T('mutable boost::shared_ptr<$dsClassName> m_ds_$dsName;')(locals()),
                 T('void read_ds_$dsName() const;')(locals())]
        for attr in self.ds.attributes:
            if not attr.type.basic:
                typename = attr.type.fullName('C++', psana_ns)
                attrName = attr.name
                if attr.type.value_type:
                    if attr.rank == 0:
                        decls += [T("mutable $typename m_ds_storage_${dsName}_${attrName};")(locals())]
                    else:
                        rank = attr.rank
                        decls += [T("mutable ndarray<const $typename, $rank> m_ds_storage_${dsName}_${attrName};")(locals())]
                else:
                    if attr.rank == 0:
                        decls += [T("mutable boost::shared_ptr<$typename> m_ds_storage_${dsName}_${attrName};")(locals())]
                    else:
                        aschema = attr.h5schema()
                        attr_class = "%s_v%d" % (aschema.name, aschema.version)
                        rank = attr.rank
                        decls += [T("mutable ndarray<$attr_class, $rank> m_ds_storage_${dsName}_${attrName};")(locals())]
                        
        return decls

    def ds_read_impl(self, psana_ns):
        dsClassName = self.ds.classNameNs()
        schema = self.schema
        type = schema.pstype
        dsName = self.ds.name
        return [_read_comp_ds_impl_template.render(locals())]

    def ds_decltype(self, psana_ns):
        '''Return declaration type of the dataset variable'''
        dsClassName = self.ds.classNameNs()
        dsName = self.ds.name
        return T('boost::shared_ptr<$dsClassName>')(locals())

    def genDs(self, inc, cpp, psana_ns):

        if 'external' in self.ds.tags:
            _log.debug("_genDs: skip dataset - external")
            return

        ns = self.schema.nsName()
        className = self.ds.className()
        
        print >>cpp, self._genH5TypeFunc("stored", psana_ns)
        print >>cpp, self._genH5TypeFunc("native", psana_ns)

        attributes = []
        for attr in self.ds.attributes:
            dattr = dict(name = attr.name)
            
            _log.debug("_genDs: dataset attr: name=%s rank=%s shape=%s", attr.name, attr.rank, attr.shape)
            
            # base type
            if isinstance(attr.type, Enum):
                # enum types are mapped to uint32 for now, can use shorter
                # presentation if optimization is necessary
                dattr['type'] = "int32_t"
            elif not attr.type.basic:
                dattr['type'] = _h5ds_typename(attr)
            else:
                dattr['type'] = attr.type.name

            if attr.rank > 0:
                if dattr['type'] == 'char':
                    if attr.sizeIsVlen():
                        dattr['type'] = dattr['type'] +'*'
                    else:
                        dattr['name'] += '[' + str(attr.shape.size()) +']'
                else:
                    if attr.sizeIsVlen(): 
                        dattr['vlen'] = True
                        dattr['type'] = dattr['type'] +'*'
                    elif attr.sizeIsConst():
                        dattr['name'] += '[' + str(attr.shape.size()) +']'
                    else:
                        dattr['type'] = dattr['type'] +'*'
            attributes.append(dattr)


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
                        if isinstance(arg.type, Enum): attr = "{0}({1})".format(arg.type.fullName('C++', psana_ns), attr)
                        dsattrs.append(attr)

                    conversion = pstype.fullName('C++', psana_ns)
                    cvt_args = ', '.join(dsattrs)

        except Exception, ex:
            # if we fail just ignore it
            _log.debug('_genDs: exception for conv operator: %s', ex)

        print >>inc, _ds_comp_decl_template.render(locals())
        
        # generate constructor and destructor
        pointers = [attr.name for attr in self.ds.attributes if attr.rank > 0 and not attr.sizeIsConst()]
        vlen_pointers = [attr.name for attr in self.ds.attributes if attr.rank > 0 and attr.type.name != 'char' and attr.sizeIsVlen()]
        print >>cpp, _ds_ctor_dtor_template.render(locals())

    def _genH5TypeFunc(self, func, psana_ns):
        """
        Generate native_type()/stored_type() static method for dataset class.
        """

        ns = self.schema.nsName()
        className = self.ds.className()

        attributes = [self._genH5TypeFuncAttr(attr, func, psana_ns) for attr in self.ds.attributes]
        return _ds_comp_type_method_template.render(locals())


    def _genH5TypeFuncAttr(self, attr, func, psana_ns):
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
                
                typename = attr.type.parent.fullName('C++', psana_ns)
                constants = [dict(name=c.name, type=typename) for c in attr.type.constants()]
                type = '_enum_type_' + attr.name
                type_decl = _enum_h5type_definition.render(locals())
                if attr.rank > 0:
                    baseType = type
                    type = '_array_type_' + attr.name
                    rank = -1 if attr.sizeIsVlen() else attr.rank
                    shape = attr.shape.cs_dims()
                    type_decl += _array_h5type_definition.render(locals())
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
                    shape = attr.shape.cs_dims()
                    type_decl = _array_h5type_definition.render(locals())
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
                type_decl = _array_h5type_definition.render(locals())

            return dict(name=attr_name, member=attr_member, type_decl=type_decl, type=type)


class DatasetRegular(object):
    '''Helper class for datasets which do not need compound data type'''
    
    def __init__(self, schema, ds):
        '''Arguments must have H5Type and H5Dataset type'''
        self.schema = schema
        self.ds = ds

    def _attr_typename(self, psana_ns):
        '''Returns type name for the attribute'''
        if self.ds.type.value_type:
            return self.ds.type.fullName('C++', psana_ns)
        else:
            # find a schema for attribute
            aschema = self.ds.h5schema()
            if not aschema: raise ValueError('No schema found for dataset %s' % self.ds.name)
            if len(aschema.datasets) != 1:
                raise ValueError('Attribute schema has number of datasets != 1: %d for attr %s of type %s' % (len(aschema.datasets), self.name, self.ds.type.name))

            return T("${name}_v${version}")[aschema]

    def _attr_dsname(self):
        '''Returns dataset type name for the attribute'''
        if not self.ds.type.basic:
            aschema = self.ds.h5schema()
            if not aschema: raise ValueError('No schema found for dataset %s' % self.ds.name)
            return aschema.datasets[0].classNameNs()

    def ds_read_decl(self, psana_ns):
        _log.debug("ds_read_decl: ds=%s", self.ds)
        dsName = self.ds.name
        rank = self.ds.rank
        typename = self._attr_typename(psana_ns)
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

    def ds_decltype(self, psana_ns):
        '''Return declaration type of the dataset variable'''
        rank = self.ds.rank
        if rank > 0:
            typename = self._attr_typename(psana_ns)
            return T("ndarray<const $typename, $rank>")(locals())
        else:
            dsClassName = _h5ds_typename(self.ds)
            return T('mutable boost::shared_ptr<$dsClassName>')(locals())

    def ds_read_impl(self, psana_ns):
        schema = self.schema
        type = schema.pstype
        dsName = self.ds.name
        rank = self.ds.rank
        typename = self._attr_typename(psana_ns)

        if rank > 0:
            if self.ds.type.basic:
                return [_read_arr_ds_basic_impl_template.render(locals())]
            else:
                ds_struct = self._attr_dsname()
                return [_read_arr_ds_udf_impl_template.render(locals())]
        else:
            dsClassName = _h5ds_typename(self.ds)
            if self.ds.type.value_type:
                return [_read_reg_ds_value_impl_template.render(locals())]
            else:
                return [_read_reg_ds_abs_impl_template.render(locals())]

    def genDs(self, inc, cpp, psana_ns):

        pass

class SchemaType(object):
    '''
    Base type for schema types below.
    '''
    def __init__(self, schema):
        '''schema parameter is of type H5Type'''
        self.schema = schema
        self.datasets = [_dsFactory(schema, ds) for ds in schema.datasets]
        

class SchemaValueType(SchemaType):
    '''
    All stuff needed for generation of code for value-types.
    '''
    def __init__(self, schema):
        '''schema parameter is of type H5Type'''
        SchemaType.__init__(self, schema)

    # generator for all HFD5 attributes
    def _schemaAttributes(self):
        for ds in self.schema.datasets:
            for dsattr in ds.attributes:
                yield ds, dsattr
        
    def genSchema(self, inc, cpp, psana_ns):
        """Generate code for value types"""

        schema = self.schema
        type = schema.pstype
        
        if 'skip-proxy' in schema.tags: return
        
        _log.debug("_genValueType: type=%r", type)

        psanatypename = type.fullName('C++', psana_ns)

        # find a constructor with some arguments
        ctors = [ctor for ctor in type.ctors if (ctor.args or 'auto' in ctor.tags)]
        if not ctors: raise TypeError("Type " + type.name + " does not have constructor defined")
        if len(ctors) > 1: raise TypeError("Type " + type.name + " has multiple constructors defined")
        ctor = ctors[0]

        # map ctor parameters to schema objects
        dsattrs = []
        _log.debug("_genValueType: ctor args=%r", ctor.args)
        for arg in ctor.args:
            if not arg.method: raise TypeError("Attribute " + arg.dest.name + " does not have access method")
            ds_attr = [(ds, dsattr) for ds, dsattr in self._schemaAttributes() if arg.method.name == dsattr.method]
            if not ds_attr:
                raise TypeError("Failed to find HDF5 attribute for constructor argument %s/%s in type %s" % (arg.name, arg.method.name, type.name))
            else:
                dsattrs += ds_attr[:1]

        args = []
        for ds, dsattr in dsattrs:
            expr = 'ds_{0}->{1}'.format(ds.name, dsattr.name)
            if isinstance(dsattr.type, Enum):
                expr = '{0}({1})'.format(dsattr.type.fullName('C++', psana_ns), expr)
            args.append(expr)

        print >>inc, _valtype_proxy_decl.render(locals())
        print >>cpp, _getTypedImpl_template.render(locals())
        
class SchemaAbstractType(SchemaType):
    '''
    All stuff needed for generation of code for value-types.
    '''
    def __init__(self, schema):
        '''schema parameter is of type H5Type'''
        SchemaType.__init__(self, schema)
            

    def genSchema(self, inc, cpp, psana_ns):
        """Generate code for abstract types"""
        
        schema = self.schema
        type = schema.pstype
        psanatypename = type.fullName('C++', psana_ns)
        className = '{0}_v{1}'.format(type.name, schema.version)
        
        _log.debug("_genAbsType: type=%s", repr(type))

        cpp_code = []

        # declarations for public methods 
        methods = []
        for t in _types(self.schema.pstype):
            for meth in t.methods(): 
                if meth.access == 'public': 
                    decl, impl = self._genMethod(meth, psana_ns)
                    methods += decl
                    cpp_code += impl
            # generate _shape() methods for array attributes
            for attr in t.attributes() :
                decl, impl = self._genAttrShapeDecl(attr)
                methods += decl
                cpp_code += impl

        ds_read_decl = []
        for ds in self.datasets:
            ds_read_decl += ds.ds_read_decl(psana_ns)
            cpp_code += ds.ds_read_impl(psana_ns)

        # explicitely instantiate class with known config types
        for config in type.xtcConfig:
            cfgClassName = config.fullName('C++', psana_ns)
            cpp_code += [T("template class $className<$cfgClassName>;")(locals())]

        # may also provide a constructor which takes dataset data
        if len(self.datasets) == 1:
            ds = self.datasets[0]
            decltype = ds.ds_decltype(psana_ns)
            dsName = ds.ds.name
            dsCtorWithArg = T('(const ${decltype}& ds) : m_ds_${dsName}(ds) {}')(locals())

        print >>inc, _abs_type_decl.render(locals())
        for line in cpp_code:
            print >>cpp, line

    def _genAttrShapeDecl(self, attr):

        if not attr.shape_method: return [], [] 
        if not attr.accessor: return [], []
        
        doc = "Method which returns the shape (dimensions) of the data returned by %s() method." % \
                attr.accessor.name
        
        # value-type arrays return ndarrays which do not need shape method
        if attr.type.value_type and attr.type.name != 'char': return [], []

        shape = [str(s or -1) for s in attr.shape.dims]

        body = _shape_method_body_template.render(locals());
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


    def _genMethod(self, meth, psana_ns):
        """Generate method declaration and definition, returns tuple of lists of strings"""


        if meth.name == "_sizeof" : return [], []

        _log.debug("_genMethod: meth: %s", meth)
        
        decl = []
        impl = []

        schema = self.schema
        type = schema.pstype
        psanatypename = type.fullName('C++', psana_ns)
        className = '{0}_v{1}'.format(type.name, schema.version)

        # determine dataset and attribute which holds data for this method
        # for non-compound datasets (datasets without attributes) attr will be None
        ds, attr = self._method2ds(meth)
        _log.debug("_genMethod: h5ds: %s, h5attr: %s, schema: %s", ds, attr, self.schema)

        if attr :
            
            # data is stored in a dataset
            args = []
            attr_type = attr.type.fullName('C++', psana_ns)
            ret_type = meth.type.fullName('C++', psana_ns)
            rank = attr.rank
            if attr.rank:
                if meth.type.name == 'char':
                    ret_type = "const char*"
                    args = [('i%d'%i, type.lookup('uint32_t')) for i in range(attr.rank-1)]
                elif meth.type.basic or meth.type.value_type:
                    ret_type = T("ndarray<const $attr_type, $rank>")(locals())
                else:
                    args = [('i%d'%i, type.lookup('uint32_t')) for i in range(attr.rank)]
                    ret_type = T("const ${ret_type}&")(locals())
            elif not attr.type.basic:
                ret_type = T("const ${ret_type}&")(locals())
                
            meth_name = meth.name
            argdecl = ", ".join(["%s %s" % (type.name, arg) for arg, type in args])
            decl += [T("virtual $ret_type $meth_name($argdecl) const;")(locals())]
            if type.xtcConfig:
                impl += [T("template <typename Config>\n$ret_type $className<Config>::$meth_name($argdecl) const {")(locals())]
            else:
                impl += [T("$ret_type $className::$meth_name($argdecl) const {")(locals())]
            
            if attr.rank == 0:
                
                if attr.type.basic:
                    
                    # simpest case, basic type, non-array
                    impl += [T("  if (not m_ds_$name) read_ds_$name();")[ds]]
                    impl += [T("  return $type(m_ds_$name->$attr_name);")(name=ds.name, attr_name=attr.name, type=ret_type)]
                    
                elif attr.type.value_type:
                    
                    # non-array, but complex type, need to convert from HDF5 type to psana,
                    # store the result in member varibale so that we can return reference to it
                    dsName = ds.name
                    attrName = attr.name
                    memberName = T("m_ds_storage_${dsName}_${attrName}")(locals());
                    attr_type = attr.type.fullName('C++', psana_ns)
                    impl += [T("  if (not m_ds_$name) read_ds_$name();")[ds]]
                    impl += [T("  $memberName = $attr_type(m_ds_$dsName->$attrName);")(locals())]
                    impl += [T("  return $memberName;")(locals())]
                    
                else:
                    
                    # non-array, but complex type, need to convert from HDF5 type to psana,
                    # store the result in member varibale so that we can return reference to it
                    dsName = ds.name
                    attrName = attr.name
                    memberName = T("m_ds_storage_${dsName}_${attrName}")(locals());
                    aschema = attr.h5schema()
                    attr_type = _h5ds_typename(attr)
                    attr_class = "%s_v%d" % (aschema.pstype.fullName('C++'), aschema.version)
                    impl += [T("  if (not $memberName) {")(locals())]
                    impl += [T("    if (not m_ds_$name) read_ds_$name();")[ds]]
                    impl += [T("    boost::shared_ptr<$attr_type> tmp(m_ds_$dsName, &m_ds_$dsName->$attrName);")(locals())]
                    impl += [T("    $memberName = boost::make_shared<$attr_class>(tmp);")(locals())]
                    impl += [  "  }"]
                    impl += [T("  return *$memberName;")(locals())]
               
               
            else:
                
                # non-zero rank, stored in a dataset

                if attr.type.name == 'char':
                    
                    # character array
                    # TODO: if rank is >1 then methods must provide arguments for all but first index,
                    # this is not implemented yet
                    impl += [T("  if (not m_ds_$name) read_ds_$name();")[ds]]
                    impl += [T("  return ($type)(m_ds_$name->$attr_name);")(name=ds.name, attr_name=attr.name, type=ret_type)]
                    
                elif attr.type.basic:
                    
                    # array of bacis types, return ndarray, data is shared with the dataset
                    dsName = ds.name
                    attrName = attr.name
                    if attr.sizeIsVlen():
                        # VLEN array take dimension from VLEN size, currently means that only 1-dim VLEN arrays are supported
                        shape = T("m_ds_$dsName->vlen_$attrName")(locals())
                    else:
                        shape = _interpolate(str(attr.shape), type)
                    impl += [T("  if (not m_ds_$name) read_ds_$name();")[ds]]
                    impl += [T("  boost::shared_ptr<$attr_type> ptr(m_ds_$dsName, m_ds_$dsName->$attrName);")(locals())]
                    impl += [T("  return make_ndarray(ptr, $shape);")(locals())]
                    
                elif attr.type.value_type:
                    
                    # array of non-basic value type, have to convert
                    dsName = ds.name
                    attrName = attr.name
                    if attr.sizeIsVlen():
                        # VLEN array take dimension from VLEN size, currently means that only 1-dim VLEN arrays are supported
                        shape = T("m_ds_$dsName->vlen_$attrName")(locals())
                        data_size = shape
                    else:
                        shape = _interpolate(str(attr.shape), type)
                        data_size = _interpolate(str(attr.shape.size()), type)
                    memberName = T("m_ds_storage_${dsName}_${attrName}")(locals());
                    impl += [T("  if (not m_ds_$name) read_ds_$name();")[ds]]
                    impl += [T("  if ($memberName.empty()) {")(locals())]
                    impl += [T("    unsigned shape[] = {$shape};")(locals())]
                    impl += [T("    ndarray<$attr_type, $rank> tmparr(shape);")(locals())]
                    impl += [T("    std::copy(m_ds_$dsName->$attrName, m_ds_$dsName->$attrName+$data_size, tmparr.begin());")(locals())]
                    impl += [T("    $memberName = tmparr;")(locals())]
                    impl += [T("  }")(locals())]
                    impl += [T("  return $memberName;")(locals())]
                    
                else:
                    
                    # array of non-basic abstract type, have to convert
                    dsName = ds.name
                    attrName = attr.name
                    if attr.sizeIsVlen():
                        # VLEN array take dimension from VLEN size, currently means that only 1-dim VLEN arrays are supported
                        shape = T("m_ds_$dsName->vlen_$attrName")(locals())
                        data_size = shape
                    else:
                        shape = _interpolate(str(meth.attribute.shape), type)
                        data_size = _interpolate(str(meth.attribute.shape.size()), type)
                    memberName = T("m_ds_storage_${dsName}_${attrName}")(locals());
                    arguse = ''.join(["[%s]" % arg for arg, type in args])
                    aschema = attr.h5schema()
                    attr_class = "%s_v%d" % (aschema.name, aschema.version)
                    attr_type = _h5ds_typename(attr)
                    impl += [T("  if (not m_ds_$name) read_ds_$name();")[ds]]
                    impl += [T("  if ($memberName.empty()) {")(locals())]
                    impl += [T("    unsigned shape[] = {$shape};")(locals())]
                    impl += [T("    ndarray<$attr_class, $rank> tmparr(shape);")(locals())]
                    impl += [T("    for (int i = 0; i != $data_size; ++ i) {")(locals())]
                    impl += [T("      boost::shared_ptr<$attr_type> ptr(m_ds_$dsName, &m_ds_$dsName->$attrName[i]);")(locals())]
                    impl += [T("      tmparr.begin()[i] = $attr_class(ptr);")(locals())]
                    impl += [T("    }")(locals())]
                    impl += [T("    $memberName = tmparr;")(locals())]
                    impl += [T("  }")(locals())]
                    impl += [T("  return $memberName$arguse;")(locals())]

            impl += ["}"]

            return decl, impl
            
        elif ds:
            
            # non-compound dataset,  

            args = []
            ds_type = ds.type.fullName('C++', psana_ns)
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
            argdecl = ", ".join(["%s %s" % (type.name, arg) for arg, type in args])
            decl += [T("virtual $ret_type $meth_name($argdecl) const;")(locals())]
            if type.xtcConfig:
                impl += [T("template <typename Config>\n$ret_type $className<Config>::$meth_name($argdecl) const {")(locals())]
            else:
                impl += [T("$ret_type $className::$meth_name($argdecl) const {")(locals())]
            
            if rank == 0:
                
                if ds.type.value_type:
                    
                    # non-array, value type, need to convert from HDF5 type to psana,
                    # store the result in member varibale so that we can return reference to it
                    dsName = ds.name
                    memberName = T("m_ds_storage_${dsName}")(locals());
                    ds_type = ds.type.fullName('C++', psana_ns)
                    impl += [T("  if (not m_ds_$dsName) read_ds_$dsName();")(locals())]
                    impl += [T("  $memberName = $ds_type(*m_ds_$dsName);")(locals())]
                    impl += [T("  return $memberName;")(locals())]
                    
                else:
                    
                    # non-array, but complex type, need to convert from HDF5 type to psana,
                    # store the result in member varibale so that we can return reference to it
                    dsName = ds.name
                    memberName = T("m_ds_storage_${dsName}")(locals());
                    aschema = ds.h5schema()
                    ds_type = _h5ds_typename(ds)
                    ds_class = "%s_v%d" % (aschema.pstype.fullName('C++'), aschema.version)
                    impl += [T("  if (not $memberName) {")(locals())]
                    impl += [T("    if (not m_ds_$dsName) read_ds_$dsName();")(locals())]
                    impl += [T("    $memberName = boost::make_shared<$ds_class>(m_ds_$dsName);")(locals())]
                    impl += [  "  }"]
                    impl += [T("  return *$memberName;")(locals())]
               
               
            else:
                
                # non-zero rank, stored as a dataset

                if ds.type.value_type:
                    
                    # array of value type, have to convert
                    dsName = ds.name
                    impl += [T("  if (m_ds_$dsName.empty()) read_ds_$dsName();")(locals())]
                    impl += [T("  return m_ds_$dsName;")(locals())]
                    
                else:
                    
                    # array of non-basic abstract type, have to convert
                    dsName = ds.name
                    memberName = T("m_ds_storage_${dsName}")(locals());
                    arguse = ''.join(["[%s]" % arg for arg, type in args])
                    aschema = ds.h5schema()
                    ds_class = "%s_v%d" % (aschema.name, aschema.version)
                    ds_type = _h5ds_typename(ds)
                    impl += [T("  if (m_ds_$dsName.empty()) read_ds_$dsName();")(locals())]
                    impl += [T("  if ($memberName.empty()) {")(locals())]
                    impl += [T("    ndarray<$ds_class, $rank> tmparr(m_ds_$dsName.shape());")(locals())]
                    impl += [T("    const unsigned size = m_ds_$dsName.size();")(locals())]
                    impl += [T("    for (unsigned i = 0; i != $data_size; ++ i) {")(locals())]
                    impl += [T("      boost::shared_ptr<$ds_type> ptr(m_ds_$dsName.data_ptr(), &m_ds_$dsName.begin()[i]);")(locals())]
                    impl += [T("      tmparr.begin()[i] = $ds_class(ptr);")(locals())]
                    impl += [T("    }")(locals())]
                    impl += [T("    $memberName = tmparr;")(locals())]
                    impl += [T("  }")(locals())]
                    impl += [T("  return $memberName$arguse;")(locals())]

            impl += ["}"]

            return decl, impl
            
        else:
            
            # data is not stored, if method defines body then use it, otherwise skip
            # definition but declare it anyway
            
            # if no return type given then it does not return anything
            rettype = meth.type
            if rettype is None:
                rettype = "void"
            else:
                rettype = rettype.fullName('C++', psana_ns)
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
            
            template = "Config" if type.xtcConfig else None
            return self._genMethodBody(meth.name, rettype, body, meth.args, inline, static=meth.static, doc=meth.comment, template=template)



    def _genMethodBody(self, methname, rettype, body, args=[], inline=False, static=False, doc=None, template=None):
        """ Generate method, both declaration and definition, given the body of the method.
        Returns tuple of lists of strings."""
        
        # make argument list
        argsspec = ', '.join([_argdecl(*arg) for arg in args])
        schema = self.schema
        type = schema.pstype

        return ([_meth_decl.render(locals())], [_meth_def.render(locals())])

        
def Schema(schema):
    '''Factory method that returns instance of one of the above classes,
    accepts instance of H5Type type.'''
    if schema.pstype.value_type:
        return SchemaValueType(schema)
    else:
        return SchemaAbstractType(schema)
    


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
