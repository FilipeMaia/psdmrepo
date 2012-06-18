#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: PythonCodegen.py 3615 2012-05-22 17:50:05Z jbarrera@SLAC.STANFORD.EDU $
#
# Description:
#  Module PythonCodegen...
#
#------------------------------------------------------------------------

"""Calss responsible for C++ code generation for Type object 

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: PythonCodegen.py 3615 2012-05-22 17:50:05Z jbarrera@SLAC.STANFORD.EDU $

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 3615 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging
import re

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
from psddl.Template import Template as T

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

def _argdecl2(name, type):    
    return name

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
class PythonCodegen ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, inc, cpp, type, abstract=False, namespace_prefix="", pkg_name = "" ) :

        # define instance variables
        self._inc = inc
        self._cpp = cpp
        self._type = type
        self._abs = abstract
        self._namespace_prefix = namespace_prefix
        self._pkg_name = pkg_name

    #-------------------
    #  Public methods --
    #-------------------

    def codegen(self):

        logging.debug("PythonCodegen.codegen: type=%s", repr(self._type))

        # declare config classes if needed
        for cfg in self._type.xtcConfig:
            print >>self._inc, T("class $name;")[cfg]

        # base class
        base = ""

        # this class (class being generated)
        wrapped = self._type.name
        name = wrapped + "_Wrapper"

        # start class declaration
        print >>self._inc, T("\nclass $name$base {")(name = name, base = base)
        access = "private"

        # shared_ptr and C++ pointer to wrapped object
        print >>self._inc, T("  shared_ptr<$wrapped> _o;")(wrapped = wrapped)
        print >>self._inc, T("  $wrapped* o;")(wrapped = wrapped)

        # enums for version and typeId
        access = self._access("public", access)
        if self._type.type_id is not None: 
            print >>self._inc, T("  enum { TypeId = Pds::TypeId::$type_id };")(type_id=self._type.type_id)
        if self._type.version is not None: 
            print >>self._inc, T("  enum { Version = $version };")(version=self._type.version)

        # constructor
        access = self._access("public", access)
        print >>self._inc, T("  $name(shared_ptr<$wrapped> obj) : _o(obj), o(_o.get()) {}")(locals())
        print >>self._inc, T("  $name($wrapped* obj) : o(obj) {}")(locals())
        print >>self._cpp, T("\n#define _CLASS(n, policy) class_<n>(#n, no_init)\\")(locals())

        # generate methods (for public methods and abstract class methods only)
        for meth in self._type.methods(): 
            access = self._access("public", access)
            if not self._abs or meth.access == "public": self._genMethod(meth)

        # generate _shape() methods for array attributes
        for attr in self._type.attributes() :
            access = self._access("public", access)
            self._genAttrShapeDecl(attr)

        # close class declaration
        print >>self._inc, "};"
        prefix = self._namespace_prefix

        # export classes to Python via boost _class
        print >>self._cpp, ""
        if not self._abs:
            print >>self._cpp, T('  _CLASS($prefix$wrapped, return_value_policy<copy_const_reference>());')(locals())
        print >>self._cpp, T('  _CLASS($prefix$name, return_value_policy<return_by_value>());')(locals())
        if not self._abs:
            print >>self._cpp, T('  std_vector_class_($wrapped);')(locals())
        print >>self._cpp, T('  std_vector_class_($name);')(locals())
        print >>self._cpp, '#undef _CLASS';

        # define Getter clases for some types
        if re.match(r'.*(Data|DataDesc|Config|Element)V[1-9][0-9]*_Wrapper', name):
            print >>self._cpp, T('  ADD_GETTER($wrapped);')(locals())
        print >>self._cpp, ""

    def _access(self, newaccess, oldaccess):
        if newaccess != oldaccess:
            print >>self._inc, newaccess+":"
        return newaccess
        
    def _genAttrDecl(self, attr):
        """Generate attribute declaration"""
        
        logging.debug("_genAttrDecl: attr: %s", attr)
        
        doc = ""
        if attr.comment : doc = T("\t/**< $comment */")(comment = attr.comment.strip())
        
        if not attr.shape :
            if attr.isfixed():
                decl = T("  $type\t$name;$doc")(type=_typename(attr.type), name=attr.name, doc=doc)
            else:
                decl = T("  //$type\t$name;")(type=_typename(attr.type), name=attr.name)
        else:
            if attr.isfixed():
                dim = _interpolate(_dims(attr.shape.dims), attr.parent)
                decl = T("  $type\t$name$shape;$doc")(type=_typename(attr.type), name=attr.name, shape=dim, doc=doc)
            else :
                dim = _interpolate(_dims(attr.shape.dims), attr.parent)
                decl = T("  //$type\t$name$shape;")(type=_typename(attr.type), name=attr.name, shape=dim)
        print >>self._inc, decl


    def _genMethod(self, meth):
        """Generate method declaration and definition"""

        logging.debug("_genMethod: meth: %s", meth)
        
        if meth.attribute:
            
            # generate access method for a named attribute
            
            attr = meth.attribute
            args = []
                        
            if not attr.shape:
                
                # attribute is a regular non-array object, 
                # return value or reference depending on what type it is
                rettype = _typedecl(attr.type)

            elif attr.type.name == 'char':
                
                # char array is actually a string
                rettype = "const char*"
                args = _dimargs(attr.shape.dims[:-1], self._type)
                
            elif attr.type.value_type :
                
                # return ndarray
                rettype = "ndarray<%s, %d>" % (_typename(attr.type), len(attr.shape.dims))

            else:

                # array of any other types
                rettype = _typedecl(attr.type)
                args = _dimargs(attr.shape.dims, self._type)

            # guess if we need to pass cfg object to method
            cfgNeeded = False

            configs = [None]
            if cfgNeeded and not self._abs: configs = attr.parent.xtcConfig
            for cfg in configs:

                cargs = []
                if cfg: cargs = [('cfg', cfg)]

                self._genMethodBody(meth.name, rettype, cargs + args)

        elif meth.bitfield:

            # generate access method for bitfield

            bf = meth.bitfield
            expr = bf.expr()
            cfgNeeded = expr.find('{xtc-config}') >= 0
            expr = _interpolate(expr, meth.parent)

            configs = [None]
            if cfgNeeded and not self._abs: configs = meth.parent.xtcConfig
            for cfg in configs:

                args = []
                if cfg: args = [('cfg', cfg)]

                self._genMethodBody(meth.name, _typename(meth.type), args=[])

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
                    type = "ndarray<%s, %d>" % (type, meth.rank)

            # config objects may be needed 
            cfgNeeded = False

            configs = [None]
            if cfgNeeded and not self._abs: configs = meth.parent.xtcConfig
            for cfg in configs:

                args = []
                if cfg: args = [('cfg', cfg)]
                args += meth.args

                self._genMethodBody(meth.name, type, args)

    def _genMethodBody(self, methname, rettype, args=[]):
        """ Generate method, both declaration and definition, given the body of the method"""

        # make argument list
        argsspec = ', '.join([_argdecl(*arg) for arg in args])

        policy = ""
        args = ', '.join([_argdecl2(*arg) for arg in args])
        index = rettype.find("ndarray<")
        if index == 0:
            ctype_ndim = rettype[8:]
            index = ctype_ndim.rfind(">")
            if index != -1:
                ctype_ndim = ctype_ndim[:index]
            index = ctype_ndim.find(", ")
            ctype = ctype_ndim[:index]
            ndim = int(ctype_ndim[index + 2:])
            if ndim == 1 or "::" in ctype:
                if ndim > 1:
                    print "WARNING: cannot generate ndarray<%s, %d>, so generating one-dimensional vector<%s> instead" % (ctype, ndim, ctype)
                print >>self._inc, T("  vector<$ctype> $methname($argsspec) const { VEC_CONVERT(o->$methname($args), $ctype); }")(locals())
            else:
                print >>self._inc, T("  PyObject* $methname($argsspec) const { ND_CONVERT(o->$methname($args), $ctype, $ndim); }")(locals())
        elif "&" in rettype and "::" in rettype:
            if (self._pkg_name + "::") in rettype:
                type = rettype.replace("&", "");
                type = type.replace("const ", "")
                index = type.find("::")
                if index != -1:
                    type = type[2+index:] # remove "Namespace::"
                wrappertype = type + "_Wrapper"
                print >>self._inc, T("  const $wrappertype $methname($argsspec) const { return $wrappertype(($type*) &o->$methname($args)); }")(locals())
                policy = ", policy"
            else:
                type = rettype
                print >>self._inc, T("  $type $methname($argsspec) const { return o->$methname($args); }")(locals())
                policy = ", policy"
        else:
            print >>self._inc, T("  $rettype $methname($argsspec) const { return o->$methname($args); }")(locals())

        print >>self._cpp, T("    .def(\"$methname\", &n::$methname$policy)\\")(methname=methname, classname=self._type.name, policy=policy)

    def _genAttrShapeDecl(self, attr):

        if not attr.shape_method: return False
        if not attr.accessor: return False
        
        # value-type arrays return ndarrays which do not need shape method
        if attr.type.value_type and attr.type.name != 'char': return False

        shape = [str(s or "???") for s in attr.shape.dims]
        if len(shape) > 2:
            print "Error: shape has more than 2 elements: ", shape
            sys.exit(1)
        if len(shape) < 1:
            print "Error: shape has no elements! ", shape
            sys.exit(1)
        if len(shape) == 2:
            shape1 = shape[1].strip()
            if not (shape1 == 'MAX_STRING_SIZE' or re.match(r'MAX_[A-Z]+_STRING_SIZE', shape1)):
                print "Error: shape has 2 elements and second is not MAX_???_STRING_SIZE: '%s'" % shape1
                sys.exit(1)
            print "shape=", len(shape), shape

        print "// ZZZ shape=", shape
        print attr.shape_method
        self._genMethodBody(attr.shape_method, "vector<int>")

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
