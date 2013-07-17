#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module DdlHdf5Data...
#
#------------------------------------------------------------------------

"""DDL parser which generates C++ code for HDF5 data classes.

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

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import jinja2 as ji
from psddl.Attribute import Attribute
from psddl.Enum import Enum
from psddl.Package import Package
from psddl.Type import Type
from psddl.H5Type import H5Type
from psddl.H5Dataset import H5Dataset
from psddl.H5Attribute import H5Attribute
from psddl.Template import Template as T

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_ds_decl_template = ji.Template('''
namespace {{ns}} {
struct {{dsClassName}} {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  {{dsClassName}}();
  ~{{dsClassName}}();

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

_ds_type_method_template = ji.Template('''
hdf5pp::Type {{ns}}_{{className}}_{{func}}_type()
{
  typedef {{ns}}::{{className}} DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
{% for attr in attributes %}
{% if attr.type_decl %}
  {{attr.type_decl}}
{%- endif %}
  type.insert("{{attr.name}}", offsetof(DsType, {{attr.name}}), {{attr.type}});
{% endfor %}
  return type;
}

hdf5pp::Type {{ns}}::{{className}}::{{func}}_type()
{
  static hdf5pp::Type type = {{ns}}_{{className}}_{{func}}_type();
  return type;
}
''', trim_blocks=True)

_enum_h5type_definition = ji.Template('''\
  hdf5pp::EnumType<int32_t> {{type}} = hdf5pp::EnumType<int32_t>::enumType();
{% for c in constants %}
  {{type}}.insert("{{c.name}}", {{c.type}}::{{c.name}});
{% endfor %}
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

_valtype_proxy_decl = ji.Template('''\
class {{proxyName}} : public PSEvt::Proxy<{{psanatypename}}> {
public:
  typedef {{psanatypename}} PsanaType;

  {{proxyName}}(hdf5pp::Group group, hsize_t idx) : m_group(group), m_idx(idx) {}
  virtual ~{{proxyName}}() {}

protected:

  virtual boost::shared_ptr<PsanaType> getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<PsanaType> m_data;
};
''', trim_blocks=True)

def _interpolate(expr, typeobj):
    
    expr = expr.replace('{xtc-config}.', 'm_cfg->')
    expr = expr.replace('{type}.', typeobj.name+"::")
    expr = expr.replace('{self}.', "this->")
    return expr

def _typename(type):
    
    return type.fullName('C++')

def _typedecl(type):
    typename = _typename(type)
    if not type.basic : typename = "const "+typename+'&'
    return typename

def _argdecl(name, type):    
    return _typedecl(type) + ' ' + name


def _schemas(pkg):
    '''generator function for all schemas inside a package'''
    for ns in pkg.namespaces() :
        if isinstance(ns, Package) :
            for schema in _schemas(ns): yield schema        
        elif isinstance(ns, Type) :
            for schema in ns.h5schemas: yield schema

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class DdlHdf5Data ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__(self, incname, cppname, backend_options, log):
        """Constructor
        
            @param incname  include file name
            @param cppname  source file name
        """
        self.incname = incname
        self.cppname = cppname
        self.incdirname = backend_options.get('gen-incdir', "")
        self.top_pkg = backend_options.get('top-package')
        self.psana_inc = backend_options.get('psana-inc', "psddl_psana")
        self.psana_ns = backend_options.get('psana-ns', "Psana")
        self.dump_schema = 'dump-schema' in backend_options

        self._log = log

        #include guard
        g = os.path.split(self.incname)[1]
        if self.top_pkg: g = self.top_pkg + '_' + g
        self.guard = g.replace('.', '_').upper()

    #-------------------
    #  Public methods --
    #-------------------

    def parseTree(self, model):
        
        if self.dump_schema: return self._dumpSchema(model)
        
        # open output files
        self.inc = file(self.incname, 'w')
        self.cpp = file(self.cppname, 'w')
        
        # include guard to header
        print >>self.inc, "#ifndef", self.guard 
        print >>self.inc, "#define", self.guard, "1"

        msg = "\n// *** Do not edit this file, it is auto-generated ***\n"
        print >>self.inc, msg
        print >>self.cpp, msg

        inc = os.path.join(self.incdirname, os.path.basename(self.incname))
        print >>self.cpp, "#include \"%s\"" % inc
        inc = os.path.join(self.psana_inc, os.path.basename(self.incname))
        print >>self.inc, "#include \"%s\"" % inc

        # add necessary includes
        print >>self.inc, "#include \"hdf5pp/Group.h\""
        print >>self.inc, "#include \"hdf5pp/Type.h\""
        print >>self.inc, "#include \"PSEvt/Proxy.h\""
        print >>self.cpp, "#include \"hdf5pp/CompoundType.h\""
        print >>self.cpp, "#include \"hdf5pp/EnumType.h\""
        print >>self.cpp, "#include \"hdf5pp/Utils.h\""
        print >>self.cpp, "#include \"PSEvt/DataProxy.h\""


        # headers for other included packages
        for use in model.use:
            path = use['file']
            headers = use['cpp_headers']
            if not headers:
                header = os.path.splitext(path)[0] + '.h'
                header = os.path.join(self.incdirname, os.path.basename(header))
                headers = [header]
            for header in headers:
                print >>self.inc, "#include \"%s\"" % header

        # headers for externally implemented schemas or datasets, headers for datasets are included
        # into cpp file (they are likely to be needed in inplementation of make_Class functions),
        # headers for external datasets are included into header.
        for pkg in model.packages():
            for schema in _schemas(pkg):
                if 'external' in schema.tags:
                    if schema.tags['external']: print >>self.cpp, "#include \"%s\"" % schema.tags['external']
                for ds in schema.datasets:
                    if 'external' in ds.tags:
                        if ds.tags['external']: print >>self.inc, "#include \"%s\"" % ds.tags['external']

        if self.top_pkg : 
            ns = "namespace %s {" % self.top_pkg
            print >>self.inc, ns
            print >>self.cpp, ns

        # loop over packages in the model
        for pkg in model.packages() :
            self._log.debug("parseTree: package=%s", repr(pkg))
            self._parsePackage(pkg)

        if self.top_pkg : 
            ns = "} // namespace %s" % self.top_pkg
            print >>self.inc, ns
            print >>self.cpp, ns

        # close include guard
        print >>self.inc, "#endif //", self.guard

        # close all files
        self.inc.close()
        self.cpp.close()


    def _parsePackage(self, pkg):
        
        if pkg.included:
            # for include packages we need to make sure that types from those 
            # packages get their schemas, they may be needed by our schemas  
            for ns in pkg.namespaces() :
                if isinstance(ns, Package) :
                    self._parsePackage(ns)
                elif isinstance(ns, Type) :
                    if not ns.h5schemas:
                        ns.h5schemas = [self._defaultSchema(ns)]
                        self._log.debug("_parsePackage: included type.h5schemas=%s", repr(ns.h5schemas))
            return

        self._log.info("Processing package %s", pkg.name)
        
        # open namespaces
        print >>self.inc, "namespace %s {" % pkg.name
        print >>self.cpp, "namespace %s {" % pkg.name

        # enums for constants
        for const in pkg.constants() :
            if not const.included :
                self._genConst(const)

        # regular enums
        for enum in pkg.enums() :
            if not enum.included :
                self._genEnum(enum)

        # loop over packages and types
        for ns in pkg.namespaces() :
            
            if isinstance(ns, Package) :
                
                self._parsePackage(ns)
            
            elif isinstance(ns, Type) :
    
                self._parseType(type = ns)

        # close namespaces
        print >>self.inc, "} // namespace %s" % pkg.name
        print >>self.cpp, "} // namespace %s" % pkg.name


    def _genConst(self, const):
        
        print >>self.inc, "  enum {\n    %s = %s /**< %s */\n  };" % \
                (const.name, const.value, const.comment)

    def _genEnum(self, enum):
        
        if enum.comment: print >>self.inc, "\n  /** %s */" % (enum.comment)
        print >>self.inc, "  enum %s {" % (enum.name or "",)
        for const in enum.constants() :
            val = ""
            if const.value is not None : val = " = " + const.value
            doc = ""
            if const.comment: doc = ' /**< %s */' % const.comment
            print >>self.inc, "    %s%s,%s" % (const.name, val, doc)
        print >>self.inc, "  };"

    def _parseType(self, type):

        self._log.debug("_parseType: type=%s", repr(type))
        self._log.trace("Processing type %s", type.name)

        # skip included types
        if type.included : return

        if not type.h5schemas:
            type.h5schemas = [self._defaultSchema(type)]
            self._log.debug("_parseType: type.h5schemas=%s", repr(type.h5schemas))

        for schema in type.h5schemas:
            self._genSchema(type, schema)

        psanatypename = type.fullName('C++', self.psana_ns)
        typename = type.name

        configs = type.xtcConfig or [None]

        for config in configs:
            
            cfgArgDecl = ''
            if config:
                cfgArgDecl = T(', const boost::shared_ptr<$cfgtypename>& cfg')(cfgtypename=config.fullName('C++', self.psana_ns))
            
            # make factory methods
            print >>self.inc, T("boost::shared_ptr<PSEvt::Proxy<$psanatypename> > make_$typename(int version, hdf5pp::Group group, hsize_t idx$cfgArgDecl);")(locals())
            print >>self.cpp, T("boost::shared_ptr<PSEvt::Proxy<$psanatypename> > make_$typename(int version, hdf5pp::Group group, hsize_t idx$cfgArgDecl) {")(locals())
            print >>self.cpp, "  switch (version) {"
            for schema in type.h5schemas:
                classname = T("${name}_v${version}")[schema]
                print >>self.cpp, T("  case $version:")[schema]
                if type.value_type:
                    proxytype = T('Proxy_${name}_v${version}')[schema]
                    print >>self.cpp, T("    return boost::make_shared<$proxytype>(group, idx);")(locals())
                else:
                    proxytype = T('PSEvt::DataProxy<$psanatypename> ')(locals())
                    if config:
                        cfgclassname = config.fullName('C++', self.psana_ns)
                        classname = T('$classname<$cfgclassname> ')(locals())
                        print >>self.cpp, T("    return boost::make_shared<$proxytype>(boost::make_shared<$classname>(group, idx, cfg));")(locals())
                    else:
                        print >>self.cpp, T("    return boost::make_shared<$proxytype>(boost::make_shared<$classname>(group, idx));")(locals())
            proxytype = T('PSEvt::DataProxy<$psanatypename> ')(locals())
            print >>self.cpp, T("  default:\n    return boost::make_shared<$proxytype>(boost::shared_ptr<$psanatypename>());")(locals())
            print >>self.cpp, "  }\n}"


    def _genSchema(self, type, schema):

        self._log.debug("_genSchema: %s", repr(schema))

        if 'external' in schema.tags:
            self._log.debug("_genSchema: skip schema - external")
            return

        for ds in schema.datasets:
            # generate datasets classes
            self._genDs(ds, schema)
            
        if type.value_type :
            self._genValueType(type, schema)
        else:
            self._genAbsType(type, schema)
            

    def _genDs(self, ds, schema):

        self._log.debug("_genDs: %s", repr(ds))
        self._log.debug("_genDs: schema %s", schema)

        if 'external' in ds.tags:
            self._log.debug("_genDs: skip dataset - external")
            return

        ns = schema.nsName()
        dsClassName = ds.className()

        self._genH5TypeFunc(ds, ns, dsClassName, "stored")
        self._genH5TypeFunc(ds, ns, dsClassName, "native")

        attributes = []
        for attr in ds.attributes:
            dattr = dict(name = attr.name)
            
            self._log.debug("_genDs: dataset attr: name=%s rank=%s shape=%s", attr.name, attr.rank, attr.shape)
            
            # base type
            if isinstance(attr.type, Enum):
                # enum types are mapped to uint32 for now, can use shorter
                # presentation if optimization is necessary
                dattr['type'] = "int32_t"
            elif not attr.type.basic:
                dattr['type'] = attr._h5ds_typename
            else:
                dattr['type'] = attr.type.name

            if attr.rank > 0:
                if dattr['type'] == 'char':
                    if attr.sizeIsConst():
                        dattr['name'] += '[' + str(attr.shape.size()) +']'
                    else:
                        dattr['vlen'] = True
                        dattr['type'] = dattr['type'] +'*'
                else:
                    if attr.sizeIsConst():
                        dattr['name'] += '[' + str(attr.shape.size()) +']'
                    else:
                        if attr.sizeIsVlen(): dattr['vlen'] = True
                        dattr['type'] = dattr['type'] +'*'
            attributes.append(dattr)


        # if schema contains single dataset and corresponding data type is a value type
        # then add conversion function from this dataset class to a data type
        try:
            if len(schema.datasets) == 1 and schema.pstype.value_type:
                
                # find a constructor with some arguments
                ctors = [ctor for ctor in schema.pstype.ctors if (ctor.args or 'auto' in ctor.tags)]
                if len(ctors) == 1:
                    ctor = ctors[0]
    
                    # map ctor parameters to dataset attributes
                    dsattrs = []
                    for arg in ctor.args:
                        if not arg.method: raise TypeError("Attribute " + arg.dest.name + " does not have access method")
                        attr = [dsattr.name for dsattr in ds.attributes if dsattr.method == arg.method.name]
                        if not attr: raise TypeError("Failed to find HDF5 attributes for constructor arguments")
                        attr = attr[0]
                        if isinstance(arg.type, Enum): attr = "%s(%s)" % (arg.type.fullName('C++', self.psana_ns), attr)
                        dsattrs.append(attr)

                    conversion = schema.pstype.fullName('C++', self.psana_ns)
                    cvt_args = ', '.join(dsattrs)

        except Exception, ex:
            # if we fail just ignore it
            self._log.debug('_genDs: exception for conv operator: %s', ex)

        print >>self.inc, _ds_decl_template.render(locals())
        
        
        # generate constructor and destructor
        print >>self.cpp, T('$ns::$dsClassName::$dsClassName()\n{')(locals())
        for attr in ds.attributes:
            if attr.rank > 0 and not attr.sizeIsConst():
                print >>self.cpp, T('  this->$attr = 0;')(attr = attr.name)
        print >>self.cpp, T('}')(locals())
        print >>self.cpp, T('$ns::$dsClassName::~$dsClassName()\n{')(locals())
        for attr in ds.attributes:
            if attr.rank > 0 and not attr.sizeIsConst():
                print >>self.cpp, T('  delete [] this->$attr;')(attr = attr.name)
        print >>self.cpp, T('}')(locals())
        

    def _genH5TypeFunc(self, ds, ns, className, func):
        """
        Generate native_type()/stored_type() static method for dataset class.
        """

        attributes = [self._genH5TypeFuncAttr(attr, func) for attr in ds.attributes]
        print >>self.cpp, _ds_type_method_template.render(locals())


    def _genH5TypeFuncAttr(self, attr, func):
        '''
        Generate attribute data for type-definition function.
        For a given attribute returns dictionary with the following keys:
        'name'  - attribute name (string)
        'type'  - expression which results in a attribute HDF5 type (string)
        'type_decl' - optional string which produces declarations used by type
        '''

        if attr.type.basic:
            
            if isinstance(attr.type, Enum):
                typename = attr.type.parent.fullName('C++', self.psana_ns)
                constants = [dict(name=c.name, type=typename) for c in attr.type.constants()]
                type = '_enum_type_' + attr.name
                return dict(name=attr.name, type=type, type_decl=_enum_h5type_definition.render(locals()))
            else:
                type_name = attr.type.name
                if attr.rank > 0 and type_name == 'char': 
                    type_name = 'const char*'
                return dict(name=attr.name, type=T("hdf5pp::TypeTraits<$type_name>::${func}_type()")(locals()))

        else:

            # for non-basic type (like composite types) find corresponding h5 schema,
            # if it has only one dataset then use it here
            aschema = [sch for sch in attr.type.h5schemas if sch.version == attr.schema_version]
            if not aschema:
                raise ValueError('No schema found for attribute %s' % attr.name)
            aschema = aschema[0]
            if len(aschema.datasets) != 1:
                raise ValueError('Attribute schema has number of datasets != 1: %d for attr %s of type %s' % (len(aschema.datasets), attr.name, attr.type.name))

            attr_type_name = '::'.join([aschema.nsName(), aschema.datasets[0].className()])
            attr_type_name = attr.type.parent.fullName('C++') + '::' + attr_type_name

            # remember it for use in other places
            attr._h5schema = aschema 
            attr._h5ds = aschema.datasets[0]
            attr._h5ds_typename = attr_type_name

            return dict(name=attr.name, type=T("hdf5pp::TypeTraits<${attr_type_name}>::${func}_type()")(locals()))


    def _genValueType(self, type, schema):
        """Generate code for value types"""
        
        # generator for all HFD5 attributes
        def _schemaAttributes(schema):
            for ds in schema.datasets:
                for dsattr in ds.attributes:
                    yield ds, dsattr

        
        self._log.debug("_genValueType: type=%r", type)

        proxyName = T("Proxy_${name}_v${version}")[schema]
        psanatypename = type.fullName('C++', self.psana_ns)

        print >>self.inc, _valtype_proxy_decl.render(locals())

        # implementation of getTypedImpl()
        print >>self.cpp, T("boost::shared_ptr<$psanatypename>")(locals())
        print >>self.cpp, T("$proxyName::getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key)\n{")(locals())
        
        # get all datasets
        for ds in schema.datasets:
            dsClassName = '::'.join([schema.nsName(), ds.className()])
            dsName = ds.name
            print >>self.cpp, T("  boost::shared_ptr<$dsClassName> ds_$dsName = hdf5pp::Utils::readGroup<$dsClassName>(m_group, \"$dsName\", m_idx);")(locals())

        # find a constructor with some arguments
        ctors = [ctor for ctor in type.ctors if (ctor.args or 'auto' in ctor.tags)]
        if not ctors: raise TypeError("Type " + type.name + " does not have constructor defined")
        if len(ctors) > 1: raise TypeError("Type " + type.name + " has multiple constructors defined")
        ctor = ctors[0]

        # map ctor parameters to schema objects
        dsattrs = []
        self._log.debug("_genValueType: ctor args=%r", ctor.args)
        for arg in ctor.args:
            if not arg.method: raise TypeError("Attribute " + arg.dest.name + " does not have access method")
            ds_attr = [(ds, dsattr) for ds, dsattr in _schemaAttributes(schema) if arg.method.name == dsattr.method]
            if not ds_attr:
                raise TypeError("Failed to find HDF5 attribute for constructor argument %s/%s in type %s" % (arg.name, arg.method.name, type.name))
            else:
                dsattrs += ds_attr[:1]

        args = ['ds_'+ds.name+'->'+dsattr.name for ds, dsattr in dsattrs]
        print >>self.cpp, "  return boost::make_shared<PsanaType>(%s);" % (', '.join(args),)

        print >>self.cpp, "}\n"

    def _genAbsType(self, type, schema):
        """Generate code for abstract types"""
        
        def _types(type):
            """Generator for the type list of the given type plus all it bases"""
            if type.base:
                for t in _types(type.base): yield t
            yield type
        
        self._log.debug("_genAbsType: type=%s", repr(type))

        className = T("${name}_v${version}")[schema]
        psanatypename = type.fullName('C++', self.psana_ns)

        print >>self.inc, '\n'
        if type.xtcConfig:
            print >>self.inc, T("template <typename Config>")(locals())
        print >>self.inc, T("class $className : public $psanatypename {\npublic:")(locals())

        print >>self.inc, T("  typedef $psanatypename PsanaType;")(locals())
        
        # constructor
        print >>self.inc, T("  $className() {}")(locals())
        if type.xtcConfig:
            print >>self.inc, T("  $className(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)")(locals())
            print >>self.inc, "    : m_group(group), m_idx(idx), m_cfg(cfg) {}"
        else:
            print >>self.inc, T("  $className(hdf5pp::Group group, hsize_t idx)")(locals())
            print >>self.inc, "    : m_group(group), m_idx(idx) {}"
        if len(schema.datasets) == 1:
            ds = schema.datasets[0]
            dsClassName = '::'.join([schema.nsName(), ds.className()])
            dsName = ds.name
            print >>self.inc, T("  $className(const boost::shared_ptr<$dsClassName>& ds) : m_ds_$dsName(ds) {}")(locals())

        # destructor
        print >>self.inc, T("  virtual ~$className() {}")(locals())

        # declarations for public methods 
        for t in _types(type):
            for meth in t.methods(): 
                if meth.access == 'public': self._genMethod(meth, type, schema, className)
            # generate _shape() methods for array attributes
            for attr in t.attributes() :
                self._genAttrShapeDecl(attr, className)


        print >>self.inc, "private:"

        print >>self.inc, "  mutable hdf5pp::Group m_group;"
        print >>self.inc, "  hsize_t m_idx;"
        if type.xtcConfig:
            print >>self.inc, "  boost::shared_ptr<Config> m_cfg;"


        for ds in schema.datasets:
            dsClassName = '::'.join([schema.nsName(), ds.className()])
            dsName = ds.name
            print >>self.inc, T("  mutable boost::shared_ptr<$dsClassName> m_ds_$dsName;")(locals())
            print >>self.inc, T("  void read_ds_$dsName() const;")(locals())
            if type.xtcConfig:
                print >>self.cpp, T("template <typename Config>\nvoid $className<Config>::read_ds_$dsName() const {")(locals())
            else:
                print >>self.cpp, T("void $className::read_ds_$dsName() const {")(locals())
            print >>self.cpp, T("  m_ds_$dsName = hdf5pp::Utils::readGroup<$dsClassName>(m_group, \"$dsName\", m_idx);")(locals())
            print >>self.cpp, "}"

            # user-defined data may need local storge as it is returned by reference
            for attr in ds.attributes:
                if not attr.type.basic:
                    typename = attr.type.fullName('C++', self.psana_ns)
                    attrName = attr.name
                    if attr.type.value_type:
                        if attr.rank == 0:
                            print >>self.inc, T("  mutable $typename m_ds_storage_${dsName}_${attrName};")(locals())
                        else:
                            rank = attr.rank
                            print >>self.inc, T("  mutable ndarray<const $typename, $rank> m_ds_storage_${dsName}_${attrName};")(locals())
                    else:
                        if attr.rank == 0:
                            print >>self.inc, T("  mutable boost::shared_ptr<$typename> m_ds_storage_${dsName}_${attrName};")(locals())
                        else:
                            attr_class = "%s_v%d" % (attr._h5schema.name, attr._h5schema.version)
                            rank = attr.rank
                            print >>self.inc, T("  mutable ndarray<$attr_class, $rank> m_ds_storage_${dsName}_${attrName};")(locals())

        # explicitely instantiate class with known config types
        for config in type.xtcConfig:
            cfgClassName = config.fullName('C++', self.psana_ns)
            print >>self.cpp, T("template class $className<$cfgClassName>;")(locals())

        # close class declaration
        print >>self.inc, "};\n"

    def _genAttrShapeDecl(self, attr, className):

        if not attr.shape_method: return 
        if not attr.accessor: return
        
        doc = "Method which returns the shape (dimensions) of the data returned by %s() method." % \
                attr.accessor.name
        
        # value-type arrays return ndarrays which do not need shape method
        if attr.type.value_type and attr.type.name != 'char': return

        shape = [str(s or -1) for s in attr.shape.dims]

        body = "  std::vector<int> shape;" 
        body += T("\n  shape.reserve($size);")(size=len(shape))
        for s in shape:
            body += T("\n  shape.push_back($dim);")(dim=s)
        body += "\n  return shape;"

        # guess if we need to pass cfg object to method
        cfgNeeded = body.find('{xtc-config}') >= 0
        body = _interpolate(body, attr.parent)

        self._genMethodBody(attr.shape_method, "std::vector<int>", className, body, [], inline=False, doc=doc)

    def _genMethod(self, meth, type, schema, className):
        """Generate method declaration and definition"""

        def _method2ds(method, schema):
            """Map method to a dataset and attribute"""
            for ds in schema.datasets:
                for attr in ds.attributes:
                    if attr.method == meth.name:
                        return (ds, attr)
            return (None, None)


        if meth.name == "_sizeof" : return

        self._log.debug("_genMethod: meth: %s", meth)
        

        ds, attr = _method2ds(meth, schema)
        self._log.debug("_genMethod: h5ds: %s, h5attr: %s, schema: %s", ds, attr, schema)

        if attr :
            
            
            # data is stored in a dataset
            args = []
            attr_type = attr.type.fullName('C++', self.psana_ns)
            ret_type = attr_type
            rank = attr.rank
            if attr.rank:
                if attr.type.name == 'char':
                    ret_type = "const char*"
                    args = [('i%d'%i, type.lookup('uint32_t')) for i in range(attr.rank-1)]
                elif attr.type.basic or attr.type.value_type:
                    ret_type = T("ndarray<const $attr_type, $rank>")(locals())
                else:
                    args = [('i%d'%i, type.lookup('uint32_t')) for i in range(attr.rank)]
                    ret_type = T("const ${ret_type}&")(locals())
            elif not attr.type.basic:
                ret_type = T("const ${ret_type}&")(locals())
                
            meth_name = meth.name
            argdecl = ", ".join(["%s %s" % (type.name, arg) for arg, type in args])
            print >>self.inc, T("  virtual $ret_type $meth_name($argdecl) const;")(locals())
            if type.xtcConfig:
                print >>self.cpp, T("template <typename Config>\n$ret_type $className<Config>::$meth_name($argdecl) const {")(locals())
            else:
                print >>self.cpp, T("$ret_type $className::$meth_name($argdecl) const {")(locals())
            
            if attr.rank == 0 and attr.type.basic:
                
                # simpest case, basic type, non-array
                print >>self.cpp, T("  if (not m_ds_$name.get()) read_ds_$name();")[ds]
                print >>self.cpp, T("  return $type(m_ds_$name->$attr_name);")(name=ds.name, attr_name=attr.name, type=ret_type)
                
            elif attr.rank == 0 and attr.type.value_type:
                
                # non-array, but complex type, need to convert from HDF5 type to psana,
                # store the result in member varibale so that we can return reference to it
                dsName = ds.name
                attrName = attr.name
                memberName = T("m_ds_storage_${dsName}_${attrName}")(locals());
                attr_type = attr.type.fullName('C++', self.psana_ns)
                print >>self.cpp, T("  if (not m_ds_$name.get()) read_ds_$name();")[ds]
                print >>self.cpp, T("  $memberName = $attr_type(m_ds_$dsName->$attrName);")(locals())
                print >>self.cpp, T("  return $memberName;")(locals())
                
            elif attr.rank == 0:
                
                # non-array, but complex type, need to convert from HDF5 type to psana,
                # store the result in member varibale so that we can return reference to it
                dsName = ds.name
                attrName = attr.name
                memberName = T("m_ds_storage_${dsName}_${attrName}")(locals());
                aschema = attr._h5schema
                attr_type = attr._h5ds_typename
                attr_class = "%s_v%d" % (aschema.name, aschema.version)
                print >>self.cpp, T("  if (not m_ds_$name.get()) {")[ds]
                print >>self.cpp, T("    read_ds_$name();")[ds]
                print >>self.cpp, T("    boost::shared_ptr<$attr_type> tmp(m_ds_$dsName, &m_ds_$dsName->$attrName);")(locals())
                print >>self.cpp, T("    $memberName = boost::make_shared<$attr_class>(tmp);")(locals())
                print >>self.cpp, "  }"
                print >>self.cpp, T("  return *$memberName;")(locals())
                
            elif attr.rank and attr.type.name == 'char':
                
                # character array
                # TODO: if rank is >1 then methods must provide arguments for all but first index,
                # this is not implemented yet
                print >>self.cpp, T("  if (not m_ds_$name.get()) read_ds_$name();")[ds]
                print >>self.cpp, T("  return ($type)(m_ds_$name->$attr_name);")(name=ds.name, attr_name=attr.name, type=ret_type)
                
            elif attr.rank and attr.type.basic:
                
                # array of bacis types, return ndarray, data is shared with the dataset
                dsName = ds.name
                attrName = attr.name
                if attr.sizeIsVlen():
                    # VLEN array take dimension from VLEN size, currently means that only 1-dim VLEN arrays are supported
                    shape = T("m_ds_$dsName->vlen_$attrName")(locals())
                else:
                    shape = _interpolate(str(meth.attribute.shape), type)
                print >>self.cpp, T("  if (not m_ds_$name.get()) read_ds_$name();")[ds]
                print >>self.cpp, T("  boost::shared_ptr<$attr_type> ptr(m_ds_$dsName, m_ds_$dsName->$attrName);")(locals())
                print >>self.cpp, T("  return make_ndarray(ptr, $shape);")(locals())
                
            elif attr.rank and attr.type.value_type:
                
                # array of non-basic value type, have to convert
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
                print >>self.cpp, T("  if (not m_ds_$name.get()) read_ds_$name();")[ds]
                print >>self.cpp, T("  if ($memberName.empty()) {")(locals())
                print >>self.cpp, T("    unsigned shape[] = {$shape};")(locals())
                print >>self.cpp, T("    ndarray<$attr_type, $rank> tmparr(shape);")(locals())
                print >>self.cpp, T("    std::copy(m_ds_$dsName->$attrName, m_ds_$dsName->$attrName+$data_size, tmparr.begin());")(locals())
                print >>self.cpp, T("    $memberName = tmparr;")(locals())
                print >>self.cpp, T("  }")(locals())
                print >>self.cpp, T("  return $memberName;")(locals())
                
            elif attr.rank:
                
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
                attr_class = "%s_v%d" % (attr._h5schema.name, attr._h5schema.version)
                attr_type = attr._h5ds_typename
                print >>self.cpp, T("  if (not m_ds_$name.get()) read_ds_$name();")[ds]
                print >>self.cpp, T("  if ($memberName.empty()) {")(locals())
                print >>self.cpp, T("    unsigned shape[] = {$shape};")(locals())
                print >>self.cpp, T("    ndarray<$attr_class, $rank> tmparr(shape);")(locals())
                print >>self.cpp, T("    for (int i = 0; i != $data_size; ++ i) {")(locals())
                print >>self.cpp, T("      boost::shared_ptr<$attr_type> ptr(m_ds_$dsName, &m_ds_$dsName->$attrName[i]);")(locals())
                print >>self.cpp, T("      tmparr.begin()[i] = $attr_class(ptr);")(locals())
                print >>self.cpp, T("    }")(locals())
                print >>self.cpp, T("    $memberName = tmparr;")(locals())
                print >>self.cpp, T("  }")(locals())
                print >>self.cpp, T("  return $memberName$arguse;")(locals())

            print >>self.cpp, "}"

            
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
            
            template = "Config" if type.xtcConfig else None
            self._genMethodBody(meth.name, rettype, className, body, meth.args, inline, static=meth.static, doc=meth.comment, template=template)

    def _genMethodBody(self, methname, rettype, classname, body, args=[], inline=False, static=False, doc=None, template=None):
        """ Generate method, both declaration and definition, given the body of the method"""
        
        # make argument list
        argsspec = ', '.join([_argdecl(*arg) for arg in args])

        print >>self.inc, _meth_decl.render(locals())
        print >>self.cpp, _meth_def.render(locals())



    def _defaultSchema(self, type):
        """Generate default schema for a types from type itself"""

        self._log.debug("_defaultSchema: type=%s", type)

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
            if (meth.rank == 0 and meth.type.value_type) or (meth.rank == 1 and meth.type.name == 'char'):
                if not ds:
                    dsname = 'data'
                    if "config-type" in type.tags: dsname = 'config' 
                    ds = H5Dataset(name=dsname, parent=schema, pstype=type)
                    schema.datasets.append(ds)
                attr = H5Attribute(name=meth.name, type=meth.type, rank=meth.rank, method=meth.name, parent=ds)
                ds.attributes.append(attr)
        if ds: self._log.debug("_defaultSchema: scalars dataset: %s", ds)

        # for non-array attributes of user-defined types create individual datasets
        for meth in methods:
            if meth.rank == 0 and not meth.type.value_type:
                # get/make that type schema
                if not meth.type.h5schemas:
                    meth.type.h5schemas = [self._defaultSchema(meth.type)]
                # find its schema v0
                mschema = [s for s in meth.type.h5schemas if s.version == 0]
                if not mschema: raise ValueError("cannot find schema V0 for type "+meth.type.name)
                mschema = mschema[0]
                if mschema: self._log.debug("_defaultSchema: sub-typedataset: %s", mschema)
                if len(mschema.datasets) != 1: raise ValueError("schema for sub-type "+type.name+"."+meth.type.name+" contains more than 1 dataset")
                # copy it into this schema
                schema.datasets.append(mschema.datasets[0])

        # for array attributes create individual datasets
        for meth in methods:
            if meth.rank > 1 or (meth.type.name != 'char' and meth.rank > 0):
                ds = H5Dataset(name=meth.name, parent=schema, pstype=type)
                schema.datasets.append(ds)
                attr = H5Attribute(name=meth.name, parent=ds, type=meth.type, rank=meth.rank, method=meth.name)
                ds.attributes.append(attr)

        return schema


    def _dumpSchema(self, model):
        '''
        Method which dumps hdf5 schema for all types in a model
        '''
        for pkg in model.packages():
            self._dumpPkgSchema(pkg)

    def _dumpPkgSchema(self, pkg, offset=1):

        if not pkg.included: print '%s<package name="%s">' % ("    "*offset, pkg.name)

        for ns in pkg.namespaces() :
            
            if isinstance(ns, Package) :
                self._dumpPkgSchema(ns, offset+1)
            elif isinstance(ns, Type) :
                self._dumpTypeSchema(ns, offset+1)


        if not pkg.included: print '%s</package>' % ("    "*offset,)

    def _dumpTypeSchema(self, type, offset=1):

        if not type.h5schemas:
            type.h5schemas = [self._defaultSchema(type)]
            
        if type.included: return
            
        for schema in type.h5schemas:
            
            print '%s<h5schema name="%s" version="%d">' % ("    "*(offset), schema.name, schema.version)
            
            for ds in schema.datasets:
            
                print '%s<dataset name="%s">' % ("    "*(offset+1), ds.name)
                
                for attr in ds.attributes:
                
                    rank = ""
                    meth = ""
                    if attr.rank: rank = ' rank="%d"' % attr.rank
                    if attr.method != attr.name: meth = ' method="%s"' % attr.method
                    print '%s<attribute name="%s"%s%s/>' % ("    "*(offset+2), attr.name, meth, rank)
                
                print '%s</dataset>' % ("    "*(offset+1),)
            
            print '%s</h5schema>' % ("    "*(offset),)
            

    #--------------------
    #  Private methods --
    #--------------------

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
