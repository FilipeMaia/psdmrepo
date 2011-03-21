#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CppTypeCodegen...
#
#------------------------------------------------------------------------

"""Calss responsible for C++ code generation for Type object 

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

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.Attribute import Attribute
from psddl.ExprVal import ExprVal
from psddl.Method import Method
from psddl.Type import Type

#----------------------------------
# Local non-exported definitions --
#----------------------------------

def _interpolate(expr, typeobj):
    
    expr = expr.replace('{xtc-config}', 'cfg')
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

def _dimargs(shape, type):
    if not shape : return [] 
    int_type = type.lookup('uint32_t')
    return [('i%d'%i, int_type) for i in range(len(shape.dims))]

def _dimexpr(shape):
    return ''.join(['[i%d]'%i for i in range(len(shape.dims))])

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
    def __init__ ( self, inc, cpp, type, abstract=False ) :

        # define instance variables
        self._inc = inc
        self._cpp = cpp
        self._type = type
        self._abs = abstract

    #-------------------
    #  Public methods --
    #-------------------

    def codegen(self):

        logging.debug("CppTypeCodegen.codegen: type=%s", repr(self._type))

        # class-level comment
        print >>self._inc, "\n/** Class: %s\n  %s\n*/\n" % (self._type.name, self._type.comment)

        # declare config classes if needed
        for cfg in self._type.xtcConfig:
            print >>self._inc, "class %s;" % cfg.name

        # for non-abstract types C++ may need pack pragma
        needPragmaPack = not self._abs and self._type.pack
        if needPragmaPack : 
            print >>self._inc, "#pragma pack(push,%s)" % self._type.pack

        # base class
        base = ""
        if self._type.base : base = ": public %s" % self._type.base.name

        # start class declaration
        print >>self._inc, "\nclass %s%s {" % (self._type.name, base)
        access = "private"

        # enums for version and typeId
        access = self._access("public", access)
        if self._type.version is not None: print >>self._inc, "  enum {Version = %s};" % self._type.version
        if self._type.type_id is not None: print >>self._inc, "  enum {TypeId = Pds::TypeId::%s};" % self._type.type_id

        # enums for constants
        access = self._access("public", access)
        for const in self._type.constants() :
            self._genConst(const)

        # regular enums
        access = self._access("public", access)
        for enum in self._type.enums() :
            self._genEnum(enum)

        if not self._abs:
            # constructor
            for ctor in self._type.ctors :
                access = self._access(ctor.access, access)
                self._genCtor(ctor)

        if self._abs:
            # need virtual destructor
            access = self._access("public", access)
            print >>self._inc, "  virtual ~%s();" % self._type.name
            print >>self._cpp, "\n%s::~%s() {}\n" % (self._type.name, self._type.name)

        if not self._abs:
            # all methods
            for meth in self._type.methods(): 
                access = self._access(meth.access or "public", access)
                self._genMethDecl(meth)
        else:
            # generate method declaration for public members without accessors
            for attr in self._type.attributes() :
                if attr.access == "public" and attr.accessor is None:
                    self._genPubAttrMethod(attr)

            # generate declaration for public methods only
            access = self._access("public", access)
            pub_meth = [meth for meth in self._type.methods() if meth.access == "public"]
            for meth in pub_meth: 
                self._genMethDecl(meth)

        # generate _shape() methods for array attributes
        for attr in self._type.attributes() :
            access = self._access("public", access)
            self._genAttrShapeDecl(attr)

        if not self._abs:
            # data members
            for attr in self._type.attributes() :
                access = self._access(attr.access or "private", access)
                self._genAttrDecl(attr)

        # close class declaration
        print >>self._inc, "};"

        # close pragma pack
        if needPragmaPack : 
            print >>self._inc, "#pragma pack(pop)"

    def _access(self, newaccess, oldaccess):
        if newaccess != oldaccess:
            print >>self._inc, newaccess+":"
        return newaccess
        
    def _genConst(self, const):
        
        print >>self._inc, "  enum {%s = %s};" % (const.name, const.value)

    def _genEnum(self, enum):
        
        print >>self._inc, "  enum %s {" % (enum.name or "",)
        for const in enum.constants() :
            val = ""
            if const.value is not None : val = " = " + const.value
            print >>self._inc, "    %s%s," % (const.name, val)
        print >>self._inc, "  };"

    def _genAttrDecl(self, attr):
        """Generate attribute declaration"""
        
        logging.debug("_genAttrDecl: attr: %s", attr)
        
        def _dims(shape):
            return ''.join(['[%s]'%d for d in shape.dims])
        
        if not attr.dimensions :
            if attr.isfixed():
                decl = "  %s\t%s;" % (_typename(attr.type), attr.name)
            else:
                decl = "  //%s\t%s;" % (_typename(attr.type), attr.name)
        else:
            if attr.isfixed():
                dim = _interpolate(_dims(attr.dimensions), attr.parent)
                decl = "  %s\t%s%s;" % (_typename(attr.type), attr.name, dim)
            else :
                dim = _interpolate(_dims(attr.dimensions), attr.parent)
                decl = "  //%s\t%s%s;" % (_typename(attr.type), attr.name, dim)
        if attr.comment : decl += "\t/* %s */" % attr.comment.strip()
        print >>self._inc, decl

    def _genPubAttrMethod(self, attr):
        """Generate virtual method declaration for accessing public attribute"""

        logging.debug("_genPubAttrMethod: attr: %s", attr)
        
        if not attr.isfixed():

            # attribute can only be access through calculated offset
            self._genAccessMethod(attr.name, attr)

        elif not attr.dimensions:
            
            # attribute is a regular non-array object, 
            # return value or reference depending on what type it is
            self._genMethodExpr(attr.name, _typedecl(attr.type), attr.name, inline=True)
                
        else:

            # attribute is an array object, return pointer for basic types,
            # or reference to elements for composite types
            if attr.type.basic:
                rettype = "const "+_typename(attr.type)+'*'
                expr = '&' + attr.name + '[0]'*len(attr.dimensions.dims)
                self._genMethodExpr(attr.name, rettype, expr, inline=True)
            else:
                rettype = _typedecl(attr.type)
                expr = attr.name + _dimexpr(attr.dimensions)
                self._genMethodExpr(attr.name, rettype, expr, args=_dimargs(attr.dimensions, self._type), inline=True)


    def _genMethDecl(self, meth):
        """Generate method declaration and definition"""

        logging.debug("_genMethDecl: meth: %s", meth)
        
        if meth.attribute:
            
            # generate access method for a named attribute
            
            attr = meth.attribute
                        
            if not attr.isfixed():

                # attribute can only be access through calculated offset
                self._genAccessMethod(meth.name, attr)

            elif not attr.dimensions:
                
                # attribute is a regular non-array object, 
                # return value or reference depending on what type it is
                self._genMethodExpr(meth.name, _typedecl(attr.type), attr.name, inline=True)
                    
            else:

                # attribute is an array object, return pointer for basic types,
                # or reference to elements for composite types
                if attr.type.basic:
                    rettype = "const "+_typename(attr.type)+'*'
                    expr = '&' + attr.name + '[0]'*len(attr.dimensions.dims)
                    self._genMethodExpr(meth.name, rettype, expr, inline=True)
                else:
                    rettype = _typedecl(attr.type)
                    expr = attr.name + _dimexpr(attr.dimensions)
                    self._genMethodExpr(meth.name, rettype, expr, args=_dimargs(attr.dimensions, self._type), inline=True)

        elif meth.bitfield:

            # generate access method for bitfield

            bf = meth.bitfield
            expr = bf.expr()
            cfgNeeded = expr.find('{xtc-config}') >= 0
            expr = _interpolate(expr, meth.parent)

            if cfgNeeded and not self._abs:

                if not meth.parent.xtcConfig :
                    raise ValueError('xtc-config is not defined')

                for cfg in meth.parent.xtcConfig:
                    args = [('cfg', cfg)]
                    self._genMethodExpr(meth.name, _typename(meth.type), expr, args)

            else:
                
                self._genMethodExpr(meth.name, _typename(meth.type), expr, inline=True)

        else:

            # explicitly declared method with optional expression
            
            if meth.name == "_sizeof" and self._abs : return
            
            expr = meth.expr.get("C++")
            if not expr : expr = meth.expr.get("Any")
            cfgNeeded = False
            if expr: 
                cfgNeeded = expr.find('{xtc-config}') >= 0
                expr = _interpolate(expr, meth.parent)

            # if no type given then it does not return anything
            type = meth.type
            if type is None:
                type = "void"
            else:
                type = _typename(type)

            # default is not inline, can change with a tag
            inline = 'inline' in meth.tags
            
            if cfgNeeded and not self._abs:

                if not meth.parent.xtcConfig :
                    raise ValueError('xtc-config is not defined')

                for cfg in meth.parent.xtcConfig:
                    args = [('cfg', cfg)] + meth.args
                    self._genMethodExpr(meth.name, type, expr, args, inline, static=meth.static)

            else:
                
                self._genMethodExpr(meth.name, type, expr, meth.args, inline, static=meth.static)


    def _genMethodExpr(self, methname, rettype, expr, args=[], inline=False, static=False):
        
        # make argument list
        argsspec = [_argdecl(*arg) for arg in args]
        argsspec = ', '.join(argsspec)

        if static:
            static = "static "
            const = ""
        else:
            static = ""
            const = "const"
        

        if self._abs and not static:
            # abstract method declaration
            print >>self._inc, "  virtual %s %s(%s) const = 0;" % (rettype, methname, argsspec)
        else:
            
            if not expr:

                # declaration only, implementation provided somewhere else
                print >>self._inc, "  %s%s %s(%s) %s;" % (static, rettype, methname, argsspec, const)

            else:
                
                if rettype == "void":
                    
                    if inline:
                        print >>self._inc, "  %s%s %s(%s) %s {%s;}" % (static, rettype, methname, argsspec, const, expr)
                    else:
                        print >>self._inc, "  %s%s %s(%s) %s;" % (static, rettype, methname, argsspec, const)
                        print >>self._cpp, "%s\n%s::%s(%s) %s {\n  %s;\n}" % \
                                (rettype, self._type.name, methname, argsspec, const, expr)

                else:
                    
                    if inline:
                        print >>self._inc, "  %s%s %s(%s) %s {return %s;}" % \
                                (static, rettype, methname, argsspec, const, expr)
                    else:
                        print >>self._inc, "  %s%s %s(%s) %s;" % (static, rettype, methname, argsspec, const)
                        print >>self._cpp, "%s\n%s::%s(%s) %s {\n  return %s;\n}" % \
                                (rettype, self._type.name, methname, argsspec, const, expr)

    def _genAccessMethod(self, methname, attr):
        
        logging.debug("_genAccessMethod: meth: %s", methname)

        offset = str(attr.offset)
        cfgNeeded = offset.find('{xtc-config}') >= 0
        offset = _interpolate(offset, attr.parent)
        logging.debug("_genAccessMethod: cfgNeeded: %s", cfgNeeded)

        args = _dimargs(attr.dimensions, attr.parent)
        rettype = _typedecl(attr.type)
        
        if self._abs:

            logging.debug("_genAccessMethod: self._abs")
            
            if attr.dimensions and attr.type.basic : 
                rettype = 'const '+rettype+'*'
                argstr = ""
            else:
                argstr = ', '.join([_argdecl(*x) for x in args])
            print >>self._inc, "  virtual %s %s(%s) const = 0;" % (rettype, methname, argstr)

        elif attr.type.basic:
            
            logging.debug("_genAccessMethod: attr.type.basic")

            configs = [None]
            if cfgNeeded : configs = attr.parent.xtcConfig
            for cfg in configs:
 
                args = ''
                if cfg : args = _argdecl('cfg', cfg)
            
                print >>self._inc, "  const %s* %s(%s) const {" % (rettype, methname, args)
                print >>self._inc, "    ptrdiff_t offset=%s;" % (offset,)
                print >>self._inc, "    return (const %s*)(((const char*)this)+offset);" % (rettype,)  
                print >>self._inc, "  }"

        else:
            
            idxexpr = ExprVal(0)
            if attr.dimensions : 
                idxexpr = ExprVal('i0', self._type)
                for i in range(1,len(attr.dimensions.dims)):
                    idxexpr = idxexpr*attr.dimensions.dims[i] + ExprVal('i%d'%i, self._type)
            
            configs = [None]
            if cfgNeeded : configs = attr.parent.xtcConfig
            for cfg in configs:

                if cfg :
                    argstr = ', '.join([_argdecl(*x) for x in [('cfg', cfg)] + args])
                else :
                    argstr = ', '.join([_argdecl(*x) for x in args])
                typename = _typename(attr.type)

                print >>self._inc, "  const %s& %s(%s) const {" % (typename, methname, argstr)
                print >>self._inc, "    ptrdiff_t offset=%s;" % (offset,)
                print >>self._inc, "    const %s* memptr = (const %s*)(((const char*)this)+offset);" % (typename,typename)
                print >>self._inc, "    return *(memptr + %s);" % (idxexpr,)
                print >>self._inc, "  }"


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
            
        args = ctor.args
        if not args :
            
            if 'auto' in ctor.tags:
                # make one argument per type attribute
                for attr in self._type.attributes():
                    name = "arg_"+attr.name
                    type = attr.type
                    dest = attr
                    args.append((name, type, dest))
        
        else:
            
            # convert destination names to attribute objects
            args = [(name, type, name2attr(dest)) for name, type, dest in args]

        # map attributes to arguments
        attr2arg = {}
        for name, type, dest in args:
            attr2arg[dest] = name

        # argument list for declaration
        arglist = []
        for argname, argtype, attr in args:
            if not argtype: argtype = attr.type
            tname = _typename(argtype)
            if isinstance(attr,Attribute) and attr.dimensions:
                tname = "const "+tname+'*'
            elif not attr.type.basic : 
                tname = "const "+tname+'&'
            arglist.append(tname+' '+argname)
        arglist = ", ".join(arglist)

        # initialization list
        initlist = []
        for attr in self._type.attributes():
            arg = attr2arg.get(attr,"")
            if attr.dimensions:
                init = ''
            elif arg :
                init = arg
            elif attr.bitfields:
                # there may be arguments initializing individual bitfields
                bfinit = []
                for bf in attr.bitfields:
                    bfarg = attr2arg.get(bf)
                    if bfarg: bfinit.append(bf.assignExpr(bfarg))
                init = '|'.join(bfinit)
            elif attr.name in ctor.attr_init:
                init = ctor.attr_init[attr.name]
            else:
                init = ""
            if init: initlist.append("%s(%s)" % (attr.name, init))

        # do we need generate definition too?
        if 'c++-definition' in ctor.tags:
            genDef = True
        elif 'no-c++-definition' in ctor.tags:
            genDef = False
        else:
            # generate definition only if all destinations are known
            genDef = None not in [dest for name, type, dest in args]

        if not genDef:
            
            # simply a declaration
            print >>self._inc, "  %s(%s);" % (self._type.name, arglist)

        elif 'inline' in ctor.tags:
            
            # inline the definition
            print >>self._inc, "  %s(%s)" % (self._type.name, arglist)
            if initlist: print >>self._inc, "    : %s" % (', '.join(initlist))
            print >>self._inc, "  {"
            for attr in self._type.attributes():
                arg = attr2arg.get(attr,"")
                if attr.dimensions and arg:
                    size = attr.dimensions.size()
                    first = '[0]' * (len(attr.dimensions.dims)-1)
                    print >>self._inc, "    std::copy(%s, %s+(%s), %s%s);" % (arg, arg, size, attr.name, first)
            print >>self._inc, "  }"

        else:
            
            # out-line the definition
            print >>self._inc, "  %s(%s);" % (self._type.name, arglist)

            print >>self._cpp, "%s::%s(%s)" % (self._type.name, self._type.name, arglist)
            if initlist: print >>self._cpp, "    : %s" % (', '.join(initlist))
            print >>self._cpp, "{"
            for attr in self._type.attributes():
                arg = attr2arg.get(attr,"")
                if attr.dimensions and arg:
                    size = attr.dimensions.size()
                    first = '[0]' * (len(attr.dimensions.dims)-1)
                    print >>self._inc, "  std::copy(%s, %s+(%s), %s%s);" % (arg, arg, size, attr.name, first)
            print >>self._cpp, "}"


    def _genAttrShapeDecl(self, attr):

        if not attr.shape_meth: return 

        if self._abs:
        
            print >>self._inc, "  virtual std::vector<int> %s() const = 0;" % (attr.shape_meth)
        
        else:

            shape = attr.dimensions.dims

            cfgNeeded = False        
            for s in shape:
                if '{xtc-config}' in str(s): 
                    cfgNeeded = True

            args = []
            if cfgNeeded:
                for cfg in attr.parent.xtcConfig:
                    args.append(_argdecl('cfg', cfg))
            else:
                args = ['']
                
            for arg in args:    
                print >>self._inc, "  std::vector<int> %s(%s) const;" % (attr.shape_meth, arg)
                print >>self._cpp, "std::vector<int> %s::%s(%s) const\n{" % (self._type.name, attr.shape_meth, arg)
                print >>self._cpp, "  std::vector<int> shape;" 
                print >>self._cpp, "  shape.reserve(%d);" % len(shape)
                for s in shape:
                    if s is None:
                        expr = "-1"
                    else :
                        expr = str(s)
                    print >>self._cpp, "  shape.push_back(%s);" % _interpolate(expr, self._type)
                print >>self._cpp, "  return shape;"
                print >>self._cpp, "}\n"


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
