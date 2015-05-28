#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CppTypeCodegen...
#
#------------------------------------------------------------------------

"""Class responsible for C++ code generation for Type object 

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
import logging
import types

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import jinja2 as ji
from psddl.Attribute import Attribute
from psddl.ExprVal import ExprVal
from psddl.Method import Method
from psddl.Enum import Enum
from psddl.Type import Type
from psddl.Template import Template as T
from psddl.TemplateLoader import TemplateLoader

#----------------------------------
# Local non-exported definitions --
#----------------------------------
# jinja environment
_jenv = ji.Environment(loader=TemplateLoader(), trim_blocks=True,
                       line_statement_prefix='$', line_comment_prefix='$$')

def _TEMPL(template):
    return _jenv.get_template('cppcodegen.tmpl?'+template)

def _interpolate(expr, typeobj):
    
    expr = expr.replace('{xtc-config}', 'cfg')
    expr = expr.replace('@config', 'cfg')
    expr = expr.replace('{type}.', typeobj.name+"::")
    expr = expr.replace('@type.', typeobj.name+"::")
    expr = expr.replace('{self}.', "this->")
    expr = expr.replace('@self.', "this->")
    return expr

def _hasconfig(str):
    return '{xtc-config}' in str or '@config' in str

def _typename(type):
    
    return type.fullName('C++')

def _typedecl(type):
    if isinstance(type, types.StringTypes): return type
    typename = _typename(type)
    if not type.basic : typename = "const "+typename+'&'
    return typename

def _argdecl(name, type):    
    return _typedecl(type) + ' ' + name

def _dims(dims):
    return ''.join(['[%s]'%d for d in dims])

def _dimargs(dims, type):
    int_type = type.lookup('uint32_t')
    return [('i%d'%i, int_type) for i in range(len(dims))]

def _dimexpr(dims):
    return ''.join(['[i%d]'%i for i in range(len(dims))])

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class CppTypeCodegen ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, inc, cpp, type, abstract=False, pdsdata=False ) :
        '''
        Parameters:
        inc    - file object for resulting include file
        cpp    - file object for resulting source file
        type   - instance of Type class
        abstract - set to true for interface types (psana non-value types)
        pdsdata  - set to true when called from pdsdata backend
        '''
        # define instance variables
        self._inc = inc
        self._cpp = cpp
        self._type = type
        self._abs = abstract
        self._pdsdata = pdsdata

    #-------------------
    #  Public methods --
    #-------------------

    def codegen(self):

        logging.debug("CppTypeCodegen.codegen: type=%s", repr(self._type))

        # class-level comment
        print >>self._inc, T("\n/** @class $name\n\n  $comment\n*/\n")[self._type]

        # declare config classes if needed
        for cfg in self._type.xtcConfig:
            # only add declaration for the types from the same namespace, declaring 
            # classes from other namespaces cannot be done at this point
            if cfg.parent is self._type.parent:
                print >>self._inc, T("class $name;")[cfg]

        # for non-abstract types C++ may need pack pragma
        needPragmaPack = self._pdsdata and self._type.pack
        if needPragmaPack : 
            print >>self._inc, T("#pragma pack(push,$pack)")[self._type]

        # base class
        base = ""
        if self._type.base : base = T(": public $name")[self._type.base]

        # start class declaration
        print >>self._inc, T("\nclass $name$base {")(name = self._type.name, base = base)
        access = "private"

        # enums for version and typeId
        access = self._access("public", access)
        if self._type.type_id is not None: 
            doc = '/**< XTC type ID value (from Pds::TypeId class) */'
            print >>self._inc, T("  enum { TypeId = Pds::TypeId::$type_id $doc };")(type_id=self._type.type_id, doc=doc)
        if self._type.version is not None: 
            doc = '/**< XTC type version number */'
            print >>self._inc, T("  enum { Version = $version $doc };")(version=self._type.version, doc=doc)

        # enums for constants
        access = self._access("public", access)
        for const in self._type.constants() :
            self._genConst(const)

        # regular enums
        access = self._access("public", access)
        for enum in self._type.enums() :
            self._genEnum(enum)

        if not self._abs:
            # constructors, first all declared explicitly
            defCtor = None
            for ctor in self._type.ctors :
                access = self._access(ctor.access, access)
                self._genCtor(ctor)
                if not ctor.args: defCtor = ctor
            if not defCtor:
                # generate default constructor anyway
                access = self._access('public', access)
                print >>self._inc, T("  $name() {}")[self._type]
            if not self._type.value_type:
                # non-value types also get copy constructor (possibly disabled), and disabled assignment
                _sizeof = self._type.lookup('_sizeof', Method)
                sizestr = str(self._type.size)
                if _sizeof is None or _hasconfig(sizestr):
                    access = self._access('private', access)
                    print >>self._inc, T("  $name(const $name&);")[self._type]
                    print >>self._inc, T("  $name& operator=(const $name&);")[self._type]
                else:
                    access = self._access('public', access)
                    print >>self._inc, _TEMPL('copy_ctor').render(classname=self._type.name)
                

        if self._abs:
            # need virtual destructor
            access = self._access("public", access)
            print >>self._inc, T("  virtual ~$name();")[self._type]
            print >>self._cpp, T("\n$name::~$name() {}\n")[self._type]

        # generate methods (for interfaces public methods only)
        for meth in self._type.methods(): 
            if not self._abs or meth.access == "public": 
                access = self._access(meth.access, access)
                self._genMethod(meth)

        # generate _shape() methods for array attributes
        for attr in self._type.attributes() :
            access = self._access("public", access)
            self._genAttrShapeDecl(attr)

        if not self._abs:
            # data members
            for attr in self._type.attributes() :
                access = self._access("private", access)
                self._genAttrDecl(attr)

        # close class declaration
        print >>self._inc, "};"

        # enum stream insertions
        for enum in self._type.enums() :
            self._genEnumPrint(enum)

        # close pragma pack
        if needPragmaPack : 
            print >>self._inc, "#pragma pack(pop)"

    def _access(self, newaccess, oldaccess):
        if newaccess != oldaccess:
            print >>self._inc, newaccess+":"
        return newaccess
        
    def _genConst(self, const):
        
        doc = ""
        if const.comment: doc = T("/**< $comment */ ")[const]
        print >>self._inc, T("  enum { $name = $value $doc};")(name=const.name, value=const.value, doc=doc)

    def _genEnum(self, enum):
        
        print >>self._inc, _TEMPL('enum_decl').render(locals())

    def _genEnumPrint(self, enum):
        
        if not enum.name: return
        
        print >>self._inc, _TEMPL('enum_print_decl').render(locals())
        print >>self._cpp, _TEMPL('enum_print_impl').render(locals())

    def _genAttrDecl(self, attr):
        """Generate attribute declaration"""
        
        logging.debug("_genAttrDecl: attr: %s", attr)
        
        doc = ""
        if attr.comment : doc = T("\t/**< $comment */")(comment = attr.comment.strip())
        
        if not attr.shape :
            if attr.isfixed():
                decl = T("  $type\t$name;$doc")(type=_typename(attr.stor_type), name=attr.name, doc=doc)
            else:
                decl = T("  //$type\t$name;")(type=_typename(attr.stor_type), name=attr.name)
        else:
            if attr.isfixed():
                dim = _interpolate(_dims(attr.shape.dims), attr.parent)
                decl = T("  $type\t$name$shape;$doc")(type=_typename(attr.stor_type), name=attr.name, shape=dim, doc=doc)
            else :
                dim = _interpolate(_dims(attr.shape.dims), attr.parent)
                decl = T("  //$type\t$name$shape;")(type=_typename(attr.stor_type), name=attr.name, shape=dim)
        print >>self._inc, decl


    def _genMethod(self, meth):
        """Generate method declaration and definition"""

        logging.debug("_genMethod: meth: %s", meth)
        
        if meth.attribute:
            
            # generate access method for a named attribute
            
            attr = meth.attribute
            args = []
            docstring = attr.comment
            
            if not attr.shape:
                
                # attribute is a regular non-array object, 
                # return value or reference depending on what type it is
                rettype = _typedecl(attr.type)
                body = self._bodyNonArray(attr)

            elif attr.type.name == 'char':
                
                # char array is actually a string
                rettype = "const char*"
                args = _dimargs(attr.shape.dims[:-1], self._type)
                body = self._bodyCharArrray(attr)
                
            elif attr.type.value_type :
                
                if self._pdsdata:
                    # in pdsdata we pass arguments defining sizes of unknown dimensions
                    int_type = self._type.lookup('uint32_t')
                    args = [('dim%d'%i, int_type) for i, dim in enumerate(attr.shape.dims) if dim is None]
                    
                rettype = "ndarray<const %s, %d>" % (_typename(attr.stor_type), len(attr.shape.dims))
                if self._pdsdata:
                    # for pdsdata only generate method which takes shared pointer to object owning the data
                    body = self._bodyNDArrray(attr, 'T')
                    args_shptr = [('owner', 'const boost::shared_ptr<T>&')] + args
                    docstring = attr.comment+"\n\n" if attr.comment else ""
                    docstring += "    Note: this overloaded method accepts shared pointer argument which must point to an object containing\n"\
                        "    this instance, the returned ndarray object can be used even after this instance disappears."
                    self._genMethodBody(meth.name, rettype, body, args=args_shptr, inline=True, doc=docstring, template='T')
                    
                if not self._abs:
                    docstring = attr.comment+"\n\n" if attr.comment else ""
                    docstring += "    Note: this method returns ndarray instance which does not control lifetime\n" \
                        "    of the data, do not use returned ndarray after this instance disappears."
                body = self._bodyNDArrray(attr)

            else:

                # array of any other types
                rettype = _typedecl(attr.type)
                args = _dimargs(attr.shape.dims, self._type)
                body = self._bodyAnyArrray(attr)


            self._genMethodBody(meth.name, rettype, body, args, inline=True, doc=docstring)

        elif meth.bitfield:

            # generate access method for bitfield

            bf = meth.bitfield
            body = T("return $expr;")(expr=bf.expr())

            self._genMethodBody(meth.name, _typename(meth.type), body, inline=True, doc=meth.comment)

        else:

            # explicitly declared method with optional expression
            
            if meth.name == "_sizeof" and self._abs : return
            
            # if no type given then it does not return anything
            type = meth.type
            if type is None:
                type = "void"
            else:
                type = _typename(type)
                if meth.rank > 0:
                    # for arrays of enum types we actually return array of integers
                    if isinstance(meth.type, Enum): type = _typename(meth.type.base)
                    type = "ndarray<const %s, %d>" % (type, meth.rank)

            # make method body
            body = meth.code.get("C++")
            if not body : body = meth.code.get("Any")
            if not body :
                expr = meth.expr.get("C++")
                if not expr : expr = meth.expr.get("Any")
                if expr:
                    body = expr
                    if type: body = "return %s;" % expr
                
            # default is not inline, can change with a tag
            inline = 'inline' in meth.tags
            
            self._genMethodBody(meth.name, type, body, args=meth.args, inline=inline, static=meth.static, doc=meth.comment)


    def _bodyNonArray(self, attr):
        """Makes method body for methods returning non-array attribute values"""

        return _TEMPL('body_non_array').render(locals())

    def _bodyCharArrray(self, attr):
        """Makes method body for methods returning char*"""

        if attr.isfixed():

            return T("return $name$dimexpr;")(name=attr.name, dimexpr=_dimexpr(attr.shape.dims[:-1]))
        
        else:
            
            body = T("typedef char atype$dims;")(dims=_dims(attr.shape.dims[1:]))
            body += T("\n  ptrdiff_t offset=$offset;")[attr]
            body += "\n  const atype* pchar = (const atype*)(((const char*)this)+offset);"
            body += T("\n  return pchar$dimexpr;")(dimexpr=_dimexpr(attr.shape.dims[:-1]))
            return body

    def _bodyNDArrray(self, attr, template=None):
        """Makes method body for methods returning ndarray"""

        if template:
            return _TEMPL('body_ndarray_shptr').render(locals())
        return _TEMPL('body_ndarray').render(locals())

    def _bodyAnyArrray(self, attr):
        """Makes method body for methods returning array (pointer)"""

        shape = ', '.join([str(s or 0) for s in attr.shape.dims])
        if attr.isfixed():

            return T("return $name$dimexpr;")(name=attr.name, dimexpr=_dimexpr(attr.shape.dims))

        elif attr.type.variable:
            
            # _sizeof may need config
            sizeofCfg = '@config' if _hasconfig(str(attr.type.size)) else ''
                
            typename = _typename(attr.type)
            body = T("const char* memptr = ((const char*)this)+$offset;")[attr]
            body += "\n  for (uint32_t i=0; i != i0; ++ i) {"
            body += T("\n    memptr += ((const $typename*)memptr)->_sizeof($sizeofCfg);")(locals())
            body += "\n  }"
            body += T("\n  return *(const $typename*)(memptr);")(locals())
            return body

        else:
            
            idxexpr = ExprVal('i0', self._type)
            for i in range(1,len(attr.shape.dims)):
                idxexpr = idxexpr*attr.shape.dims[i] + ExprVal('i%d'%i, self._type)
                
            # _sizeof may need config
            sizeofCfg = '@config' if _hasconfig(str(attr.type.size)) else ''

            typename = _typename(attr.type)
            body = T("ptrdiff_t offset=$offset;")[attr]
            body += T("\n  const $typename* memptr = (const $typename*)(((const char*)this)+offset);")(locals())
            body += T("\n  size_t memsize = memptr->_sizeof($sizeofCfg);")(locals())
            body += T("\n  return *(const $typename*)((const char*)memptr + ($idxexpr)*memsize);")(locals())
            return body


    def _genMethodBody(self, methname, rettype, body, args=[], inline=False, static=False, doc=None, template=None):
        """ Generate method, both declaration and definition, given the body of the method"""
        
        # guess if we need to pass cfg object to method
        cfgNeeded = body and _hasconfig(body)
        if body: body = _interpolate(body, self._type)

        configs = [None]
        if cfgNeeded and not self._abs: configs = self._type.xtcConfig
        for cfg in configs:

            cargs = []
            if cfg: cargs = [('cfg', cfg)]
            cargs += args

            # make argument list
            argsspec = ', '.join([_argdecl(*arg) for arg in cargs])
            abstract = self._abs and not static
    
            print >>self._inc, _TEMPL('method_decl').render(locals())
    
            if not abstract and body and not inline:
                # out-of-line method
                classname = self._type.name
                print >>self._cpp, _TEMPL('method_impl').render(locals())


    def _genCtor(self, ctor):

        # find attribute for a given name
        def name2attr(name):
            
            if not name: return None
            
            # try argument name
            attr = self._type.localName(name)
            if isinstance(attr, Attribute): return attr
            
            # look for bitfield names also
            for attr in self._type.attributes():
                for bf in attr.bitfields:
                    if bf.name == name: return bf
                    
            raise ValueError('No attrubute or bitfield with name %s defined in type %s' % (name, self._type.name))
            
        # map attributes to arguments
        attr2arg = {}
        for arg in ctor.args:
            attr2arg[arg.dest] = arg

        # argument list for declaration
        arglist = []
        for arg in ctor.args:
            tname = _typename(arg.type)
            if isinstance(arg.dest, Attribute) and arg.dest.shape:
                tname = "const "+tname+'*'
            elif not arg.type.basic : 
                tname = "const "+tname+'&'
            arglist.append(tname+' '+arg.name)

        # initialization list
        initlist = []
        if self._type.base:
            init = ", ".join([c.name for c in ctor.args if c.base])
            initlist.append(T("$base($init)")(base=self._type.base.name, init=init))
        for attr in self._type.attributes():
            arg = attr2arg.get(attr,"")
            if attr.shape or not attr.isfixed():
                init = ''
            elif arg :
                init = arg.expr
            elif attr.bitfields:
                # there may be arguments initializing individual bitfields
                bfinit = []
                for bf in attr.bitfields:
                    bfarg = attr2arg.get(bf)
                    if bfarg: 
                        bfinit.append(bf.assignExpr(bfarg.expr))
                    else:
                        for ctorInit in ctor.attr_init:
                            if ctorInit.dest.name == bf.name:
                                bfinit.append(bf.assignExpr(ctorInit.expr))
                init = '|'.join(bfinit)
            else:
                init = ''
                for ctorInit in ctor.attr_init:
                    if ctorInit.dest.name == attr.name:
                        init = ctorInit.expr
            if init: initlist.append(T("$attr($init)")(attr=attr.name, init=init))

        # do we need generate definition too?
        genDef = 'external' not in ctor.tags

        classname = self._type.name
        if not genDef:
            
            # simply a declaration
            print >>self._inc, _TEMPL('ctor_decl').render(locals())

        else:
            
            # initializers for arrays
            arrayinit = []
            for attr in self._type.attributes():
                arg = attr2arg.get(attr, None)
                if arg:
                    if attr.shape:
                        arrayinit.append((arg.name, str(attr.shape.size()), attr))
                    elif not attr.isfixed():
                        arrayinit.append((arg.name, None, attr))
            
            if 'inline' in ctor.tags:
                
                # inline the definition
                print >>self._inc, _interpolate(_TEMPL('ctor_decl_inline').render(locals()), self._type)

            else:
            
                # out-line the definition
                print >>self._inc, _TEMPL('ctor_decl').render(locals())
                print >>self._cpp, _interpolate(_TEMPL('ctor_impl').render(locals()), self._type)


    def _genAttrShapeDecl(self, attr):

        if not attr.shape_method: return 
        if not attr.accessor: return
        
        doc = "Method which returns the shape (dimensions) of the data returned by %s() method." % \
                attr.accessor.name
        
        # value-type arrays return ndarrays which do not need shape method
        if attr.type.value_type and attr.type.name != 'char': return

        shape = [str(s or -1) for s in attr.shape.dims]
        body = _TEMPL('body_shape').render(locals())

        self._genMethodBody(attr.shape_method, "std::vector<int>", body, inline=False, doc=doc)



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
