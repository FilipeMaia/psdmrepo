#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module DdlDumpHddl...
#
#------------------------------------------------------------------------

"""psddlc backend which dumps model in human-ddl format.

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

#---------------------------------
#  Imports of base class module --
#---------------------------------


#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.Package import Package
from psddl.Type import Type
from psddl.H5Type import H5Type
from psddl.Template import Template as T

#----------------------------------
# Local non-exported definitions --
#----------------------------------
def _fmttags(tags):
    if tags: return '\n  '.join(['[[{0}]]'.format(tag) for tag in tags])
    return ''

def _fmttags1(tags):
    if tags: return '[[{0}]]'.format(', '.join(tags))
    return ''

def _dims(dims):
    return ''.join(["[{0}]".format(str('*' if d is None else d)) for d in dims])

def _codesubs(expr):
    expr = expr.replace('{xtc-config}', '@config')
    expr = expr.replace('{type}.', '@class.')
    expr = expr.replace('{self}.', "@self.")
    return expr


#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class DdlDumpHddl ( object ) :

    @staticmethod
    def backendOptions():
        """ Returns the list of options supported by this backend, returned value is 
        either None or a list of triplets (name, type, description)"""
        return None

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, backend_options, log ) :
        '''Constructor
        
           @param backend_options  dictionary of options passed to backend
           @param log              message logger instance
        '''
        self.outname = backend_options['global:source']

        self._log = log


    #-------------------
    #  Public methods --
    #-------------------

    def parseTree ( self, model ) :
        
        # open output files
        self.out = file(self.outname, 'w')

        # headers for other included packages
        for use in model.use:
            headers = use['cpp_headers']
            fname = use['file']
            base, ext = os.path.splitext(fname)
            if ext == '.xml': fname = base
            if not headers:
                print >>self.out, '@include "{0}";'.format(fname)
            else:
                names = ', '.join(['"{0}"'.format(header) for header in headers])
                names = _fmttags(['headers({0})'.format(names)])
                print >>self.out, '@include "{0}" {1};'.format(fname, names)

        # loop over packages in the model
        for pkg in model.packages() :
            if not pkg.included :
                self._log.debug("parseTree: package=%s", repr(pkg))
                self._genPackage(pkg)

        # close all files
        self.out.close()


    def _genPackage(self, pkg):

        self._log.debug("_genPackage: pkg=%s", repr(pkg))

        tags = []
        if pkg.external: tags.append('external')
        if 'c++-name' in pkg.tags: tags.append('cpp_name("{0}")'.format(pkg.tags['c++-name']))
        tags = _fmttags(tags)

        # open package
        print >>self.out, T("@package $name $tags {")(name=pkg.name, tags=tags)

        # constants
        for const in pkg.constants():
            if not const.included:
                self._genConst(const)

        # regular enums
        for enum in pkg.enums() :
            if not enum.included :
                self._genEnum(enum)

        # loop over packages and types
        for ns in pkg.namespaces() :
            
            if isinstance(ns, Package) :
                
                self._genPackage(ns)
            
            elif isinstance(ns, Type) :
    
                self._genType(type = ns)
                
                for schema in ns.h5schemas:
                    self._genH5Schema(schema)

        # close package
        print >>self.out, T("} //- @package $name")[pkg]

    def _genConst(self, const):
        
        if const.comment:
            print >>self.out, T("  /* $comment */")[const]
        print >>self.out, T("  @const int32_t $name = $value;")[const]


    def _genEnum(self, enum):

        if not enum.name: return

        base = enum.base.name
        if enum.comment:
            print >>self.out, T("  /* $comment */")[enum]
        print >>self.out, T("  @enum $name ($base) {")(name=enum.name, base=base)
        
        for const in enum.constants() :
            val = ""
            if const.value is not None : val = " = " + const.value
            doc = ""
            if const.comment: doc = T(' /* $comment */')[const]
            print >>self.out, T("    $name$value,$doc")(name=const.name, value=val, doc=doc)
        print >>self.out, "  }"

    def _genType(self, type):

        self._log.debug("_genType: type=%s", repr(type))

        # skip included types
        if type.included : return

        print >>self.out, T("\n\n//------------------ $name ------------------")[type]

        tags = []
        if type.type_id: tags.append('type_id({0}, {1})'.format(type.type_id, type.version))
        if type.external: tags.append('external')
        if type.value_type: tags.append('value_type')
        if 'config-type' in type.tags: tags.append('config_type')
        if 'c++-name' in type.tags: tags.append('cpp_name("{0}")'.format(type.tags['c++-name']))
        if 'no-sizeof' in type.tags: tags.append('no_sizeof')
        if type.pack: tags.append('pack({0})'.format(type.pack))
        if type.xtcConfig:
            types = []
            for cfg in type.xtcConfig:
                if cfg.parent is not type.parent:
                    types.append(cfg.fullName())
                else:
                    types.append(cfg.name)
            tags.append('config({0})'.format(', '.join(types)))
        
        tags = _fmttags(tags)

        base = ""
        if type.base : base = T("($name)")[type.base]

        # start class decl
        if type.comment: print >>self.out, T("/* $comment */")[type]
        print >>self.out, T("@type $name$base")(name=type.name, base=base)
        if tags: print >>self.out, "  " + tags
        print >>self.out, "{"

        # constants
        for const in type.constants():
            self._genConst(const)
        if type.constants(): print >>self.out

        # regular enums
        for enum in type.enums() :
            self._genEnum(enum)
        if type.enums(): print >>self.out

        # attributes
        for attrib in type.attributes():
            self._genAttrib(attrib)
        
        # methods
        for meth in type.methods():
            self._genMethod(meth)
        
        # constructors
        for ctor in type.ctors:
            if ctor.args or ctor.attr_init or 'auto' in ctor.tags:
                # skip default non-auto constructors, they are always defined
                self._genCtor(ctor)
        if type.ctors: print >>self.out

        print >>self.out, "}"


    def _genAttrib(self, attr):

        # use full name if not in a global namespace and not in type's namespace (or in type itself)
        if attr.type.parent.name and not (attr.type.parent is attr.parent.parent or attr.type.parent is attr.parent):
            typename = attr.type.fullName()
        else:
            typename = attr.type.name
            
        comment = T("\t/* $comment */")[attr] if attr.comment else ''
        method = T(" -> $name")[attr.accessor] if attr.accessor else ''
        
        tags = []
        if attr._shape_method: tags.append('shape_method({0})'.format(attr._shape_method))
        if attr.accessor and attr.accessor.access is not None and attr.accessor.access  != 'public': tags.append(attr.accessor.access)
        tags = _fmttags(tags)
        if tags: tags = '  '+tags
        
        name = attr.name
        
        if not attr.bitfields:
            shape = ''
            if attr.shape:
                shape = _codesubs(_dims(attr.shape.dims))
            
            print >>self.out, T("  $typename $name$shape$method$tags;$comment")(locals())
        else:
            print >>self.out, T("  $typename $name$method$tags {$comment")(locals())
            
            for bf in attr.bitfields:

                tags = []
                if bf.accessor and bf.accessor.access is not None and bf.accessor.access != 'public': tags.append(bf.accessor.access)
                tags = _fmttags(tags)
                if tags: tags = '  '+tags
                
                typename = bf.type.name
                name = bf.name
                method = T(" -> $name")[bf.accessor] if bf.accessor else ''
                comment = T("\t/* $comment */")[bf] if bf.comment else ''
                size = bf.size

                print >>self.out, T("    $typename $name:$size$method$tags;$comment")(locals())
            
            print >>self.out, "  }"


    def _genMethod(self, meth):
        
        # accessor methods are defined by attributes
        if meth.attribute or meth.bitfield: return

        # sizeof is not declared
        if meth.name == "_sizeof": return

        # use full name if not in a global namespace and not in type's namespace (or in type itself)
        if meth.type is None:
            typename = 'void'
        elif meth.type.parent.name and not (meth.type.parent is meth.parent.parent or meth.type.parent is meth.parent):
            typename = meth.type.fullName()
        else:
            typename = meth.type.name
        if meth.rank: typename = typename + '[]'*meth.rank

        tags = []
        if 'inline' in meth.tags: tags.append('inline')
        tags = _fmttags1(tags)
        if tags: tags = '  '+tags

        args = []
        for arg in meth.args:
            args.append('{0} {1}'.format(arg[1].name, arg[0]))
        args = ', '.join(args)
        
        print >>self.out, ""
        if meth.comment: print >>self.out, T("  /* $comment */")[meth]
        name = meth.name

        if meth.code:
            print >>self.out, T("  $typename $name($args)$tags")(locals())
            count = 0
            for lang, code in meth.code.items():
                tags = _fmttags(['language("{0}")'.format(lang)])
                print >>self.out, T("  $tags @{")(locals())
                print >>self.out, _codesubs(code)
                sep = '' if count == (len(meth.code)-1) else ','
                print >>self.out, "  @}"+sep
                count += 1
        elif meth.expr:
            print >>self.out, T("  $typename $name($args)$tags")(locals())
            count = 0
            for lang, code in meth.expr.items():
                tags = _fmttags(['language("{0}")'.format(lang)])
                sep = '' if count == (len(meth.expr)-1) else ','
                code = _codesubs(code)
                print >>self.out, T("  $tags @{ return $code; @}$sep")(locals())
                count += 1
        else:
            print >>self.out, T("  $typename $name($args) [[external]]$tags;")(locals())



    def _genCtor(self, ctor):

        tags = []
        if 'auto' in ctor.tags: tags.append('auto')
        if 'inline' in ctor.tags: tags.append('inline')
        if 'force_definition' in ctor.tags: tags.append('force_definition')
        if 'external' in ctor.tags or \
            ('force_definition' not in ctor.tags and None in [arg.dest for arg in ctor.args]):
            tags.append('external')
        tags = _fmttags1(tags)
        if tags: tags = '  '+tags

        args = []
        arginits = []
        for arg in ctor.args:
            adecl = None
            ainit = None
            if arg.dest and arg.expr == arg.name:
                # argument itself is used to initialize destination
                if arg.type is arg.dest.type:
                    adecl = '{0} -> {1}'.format(arg.name, arg.dest.name)
                else:
                    adecl = '{0} {1} -> {2}'.format(arg.type.name, arg.name, arg.dest.name)
            elif arg.dest:
                # Expression is used to initialize destination, declare argument with a type and
                # define initialization expression
                adecl = '{0} {1}'.format(arg.type.name, arg.name)
                ainit = '{0}({1})'.format(arg.dest.name, arg.expr)
            else:
                # just declare argument
                adecl = '{0} {1}'.format(arg.type.name, arg.name)
                
            if arg.method and (not arg.dest or arg.dest.accessor is not arg.method):
                # add special method for this argument
                adecl = adecl + ' ' + _fmttags1(['method({0})'.format(arg.method.name)])

            args.append(adecl)
            if ainit: arginits.append(ainit)
            
        for arginit in ctor.attr_init:
            ainit = '{0}({1})'.format(arginit.dest.name, arginit.expr)
            arginits.append(ainit)
                
        args = ', '.join(args)
        arginits = ', '.join(arginits)
        
        print >>self.out, ""
        if ctor.comment: print >>self.out, T("  /* $comment */")[ctor]

        if 'auto' in ctor.tags: args = ''
        if arginits:
            print >>self.out, T("  @init($args)\n    $arginits$tags;")(locals())
        else:
            print >>self.out, T("  @init($args)$tags;")(locals())



    def _genH5Schema(self, schema):
        
        self._log.debug("_genH5Schema: schema=%s", repr(schema))
        
        # skip included types
        if schema.included : return

        print >>self.out, T("\n\n//------------------ $name ------------------")[schema]

        tags = []
        tags.append('version({0})'.format(schema.version))
        if 'external' in schema.tags: 
            if schema.tags['external']:
                tags.append('external("{0}")'.format(schema.tags['external']))
            else:
                tags.append('external')
        if 'skip-proxy' in schema.tags or 'embedded' in schema.tags: tags.append('embedded')
        if 'default' in schema.tags: tags.append('default')
        tags = _fmttags(tags)

        # start decl
        print >>self.out, T("@h5schema $name")[schema]
        if tags: print >>self.out, "  " + tags
        print >>self.out, "{"

        for enum_name, maps in schema.enum_map.items():
            # mappings for enum values
            print >>self.out, T("  @enum $name {")(name=enum_name)
            for psname, h5name in maps.items():
                print >>self.out, T("    $psname -> $h5name,")(locals())
            print >>self.out, T("  }")(name=enum_name)

        for ds in schema.datasets:

            tags = []
            if 'external' in ds.tags:
                if ds.tags['external']:
                    tags.append('external("{0}")'.format(ds.tags['external']))
                else:
                    tags.append('external')
            if 'vlen' in ds.tags: tags.append('vlen')
            if 'zero_dims' in ds.tags: tags.append('zero_dims')
            
            if ds.method:
                
                if ds.method != ds.name: tags.append('method({0})'.format(ds.method))
                tags = _fmttags1(tags)
                if tags: tags = ' '+tags

                type = ""
                if ds._type and (ds._method() is None or ds._method().type != ds._type): 
                    type = ds._type.name+' '

                shape = ''
                if ds._rank is not None and ds._rank > 0 and (ds._method() is None or ds._method().rank != ds._rank):
                    shape = '[]'*ds._rank

                print >>self.out, T("  @dataset $type$name$tags;")(type=type, name=ds.name, tags=tags)
                
            else:
                
                tags = _fmttags1(tags)
                if tags: tags = ' '+tags
                print >>self.out, T("  @dataset $name {")(name=ds.name)
                for attr in ds.attributes:

                    tags = []
                    if 'external' in attr.tags:
                        if attr.tags['external']:
                            tags.append('external("{0}")'.format(attr.tags['external']))
                        else:
                            tags.append('external')
                    if 'vlen' in attr.tags: tags.append('vlen')
                    if attr.method != attr.name: tags.append('method({0})'.format(attr.method))
                    tags = _fmttags1(tags)
                    if tags: tags = ' '+tags

                    type = ""
                    if attr._type and (attr._method() is None or attr._method().type != attr._type): 
                        type = attr._type.name+' '

                    shape = ''
                    if attr._shape and (attr._method() is None or attr._method().attribute is None or attr._method().attribute.shape != attr.shape):
                        shape = _dims(attr._shape.dims)
                    elif attr._rank is not None and attr._rank > 0 and (attr._method() is None or attr._method().rank != attr._rank):
                        shape = '[]'*attr._rank

                    print >>self.out, T("    @attribute $type$name$shape$tags;")(type=type, name=attr.name, tags=tags, shape=shape)
                    
                print >>self.out, "  }"

        print >>self.out, "}"



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
