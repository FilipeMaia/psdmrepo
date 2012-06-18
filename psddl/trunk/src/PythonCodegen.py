#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: PythonCodegen.py 3615 2012-05-22 17:50:05Z jbarrera@SLAC.STANFORD.EDU $
#
# Description:
#  Module PythonCodegen...
#
#------------------------------------------------------------------------

"""Class responsible for C++ code generation for Type object 

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
    def __init__ ( self, inc, cpp ):

        # define instance variables
        self.inc = inc
        self.cpp = cpp

    #-------------------
    #  Public methods --
    #-------------------

    def codegen(self, type, namespace_prefix, pkg_name):
        # type is abstract by default but can be reset with tag "value-type"
        abstract = not type.value_type

        self._type = type
        self._namespace_prefix = namespace_prefix
        self._pkg_name = pkg_name

        print "$$$ namespace_prefix=", namespace_prefix
        print "$$$ pkg_name=", pkg_name

        logging.debug("PythonCodegen.codegen: type=%s", repr(type))

        # declare config classes if needed
        for cfg in type.xtcConfig:
            print >>self.inc, T("class $name;")[cfg]

        # base class
        base = ""

        # this class (class being generated)
        wrapped = type.name
        name = wrapped + "_Wrapper"

        # start class declaration
        print >>self.inc, T("\nclass $name$base {")(name = name, base = base)
        access = "private"

        # shared_ptr and C++ pointer to wrapped object
        print >>self.inc, T("  shared_ptr<$wrapped> _o;")(wrapped = wrapped)
        print >>self.inc, T("  $wrapped* o;")(wrapped = wrapped)

        # enums for version and typeId
        access = self._access("public", access)
        if type.type_id is not None: 
            print >>self.inc, T("  enum { TypeId = Pds::TypeId::$type_id };")(type_id=type.type_id)
        if type.version is not None: 
            print >>self.inc, T("  enum { Version = $version };")(version=type.version)

        # constructor
        access = self._access("public", access)
        print >>self.inc, T("  $name(shared_ptr<$wrapped> obj) : _o(obj), o(_o.get()) {}")(locals())
        print >>self.inc, T("  $name($wrapped* obj) : o(obj) {}")(locals())
        print >>self.cpp, T("\n#define _CLASS(n, policy) class_<n>(#n, no_init)\\")(locals())

        # generate methods (for public methods and abstract class methods only)
        for meth in type.methods(): 
            access = self._access("public", access)
            if not abstract or meth.access == "public": self._genMethod(meth, type, abstract)

        # generate _shape() methods for array attributes
        for attr in type.attributes() :
            access = self._access("public", access)
            self._genAttrShapeDecl(type, attr)

        # close class declaration
        print >>self.inc, "};"
        prefix = self._namespace_prefix

        # export classes to Python via boost _class
        print >>self.cpp, ""
        if not abstract:
            print >>self.cpp, T('  _CLASS($prefix$wrapped, return_value_policy<copy_const_reference>());')(locals())
        print >>self.cpp, T('  _CLASS($prefix$name, return_value_policy<return_by_value>());')(locals())
        if not abstract:
            print >>self.cpp, T('  std_vector_class_($wrapped);')(locals())
        print >>self.cpp, T('  std_vector_class_($name);')(locals())
        print >>self.cpp, '#undef _CLASS';

        # define Getter clases for some types
        if re.match(r'.*(Data|DataDesc|Config|Element)V[1-9][0-9]*_Wrapper', name):
            print >>self.cpp, T('  ADD_GETTER($wrapped);')(locals())
        print >>self.cpp, ""

    def _access(self, newaccess, oldaccess):
        if newaccess != oldaccess:
            print >>self.inc, newaccess+":"
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
        print >>self.inc, decl


    def _genMethod(self, meth, type, abstract):
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
                args = _dimargs(attr.shape.dims[:-1], type)
                
            elif attr.type.value_type :
                
                # return ndarray
                rettype = "ndarray<%s, %d>" % (_typename(attr.type), len(attr.shape.dims))

            else:

                # array of any other types
                rettype = _typedecl(attr.type)
                args = _dimargs(attr.shape.dims, type)

            # guess if we need to pass cfg object to method
            cfgNeeded = False

            configs = [None]
            if cfgNeeded and not abstract: configs = attr.parent.xtcConfig
            for cfg in configs:

                cargs = []
                if cfg: cargs = [('cfg', cfg)]

                self._genMethodBody(type, meth.name, rettype, cargs + args)

        elif meth.bitfield:

            # generate access method for bitfield

            bf = meth.bitfield
            expr = bf.expr()
            cfgNeeded = expr.find('{xtc-config}') >= 0
            expr = _interpolate(expr, meth.parent)

            configs = [None]
            if cfgNeeded and not abstract: configs = meth.parent.xtcConfig
            for cfg in configs:

                args = []
                if cfg: args = [('cfg', cfg)]

                self._genMethodBody(type, meth.name, _typename(meth.type), args=[])

        else:

            # explicitly declared method with optional expression
            
            if meth.name == "_sizeof" and abstract : return
            
            # if no type given then it does not return anything
            method_type = meth.type
            if method_type is None:
                method_type = "void"
            else:
                method_type = _typename(method_type)
                if meth.rank > 0:
                    method_type = "ndarray<%s, %d>" % (method_type, meth.rank)

            # config objects may be needed 
            cfgNeeded = False

            configs = [None]
            if cfgNeeded and not abstract: configs = meth.parent.xtcConfig
            for cfg in configs:

                args = []
                if cfg: args = [('cfg', cfg)]
                args += meth.args

                self._genMethodBody(type, meth.name, method_type, args)

    def _genMethodBody(self, type, method_name, rettype, args=[]):
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
                print >>self.inc, T("  vector<$ctype> $method_name($argsspec) const { VEC_CONVERT(o->$method_name($args), $ctype); }")(locals())
            else:
                print >>self.inc, T("  PyObject* $method_name($argsspec) const { ND_CONVERT(o->$method_name($args), $ctype, $ndim); }")(locals())
        elif "&" in rettype and "::" in rettype:
            if (self._pkg_name + "::") in rettype:
                method_type = rettype.replace("&", "").replace("const ", "")
                index = method_type.find("::")
                if index != -1:
                    method_type = method_type[2+index:] # remove "Namespace::"
                wrappertype = method_type + "_Wrapper"
                print >>self.inc, T("  const $wrappertype $method_name($argsspec) const { return $wrappertype(($method_type*) &o->$method_name($args)); }")(locals())
                policy = ", policy"
            else:
                method_type = rettype
                print >>self.inc, T("  $method_type $method_name($argsspec) const { return o->$method_name($args); }")(locals())
                policy = ", policy"
        else:
            print >>self.inc, T("  $rettype $method_name($argsspec) const { return o->$method_name($args); }")(locals())

        print >>self.cpp, T("    .def(\"$method_name\", &n::$method_name$policy)\\")(method_name=method_name, classname=type.name, policy=policy)
        """
        if "_shape" in method_name:
            method_name = method_name.replace("_shape", "_size")
            print >>self.cpp, T("    .def(\"$method_name\", &n::$method_name$policy)\\")(method_name=method_name, classname=type.name, policy=policy)
        """

    def _genAttrShapeDecl(self, type, attr):
        if not attr.shape_method: return None
        if not attr.accessor: return None
        
        # value-type arrays return ndarrays which do not need shape method
        if attr.type.value_type and attr.type.name != 'char': return None

        dimensions = [str(s or "") for s in attr.shape.dims]
        if len(dimensions) < 1:
            print "Error: cannot generate '%s' method: shape has no dimensions!" % shape_method
            sys.exit(1)
        if len(dimensions) > 2:
            print "Error: cannot generate '%s' method: shape has more than 2 dimensions." % shape_method
            sys.exit(1)
        if len(dimensions) == 2:
            dimensions1 = dimensions[1].strip()
            if not (dimensions1 == 'MAX_STRING_SIZE' or re.match(r'MAX_[A-Z]+_STRING_SIZE', dimensions1)):
                print "Cannot generate '%s' method: shape has 2 dimensions and second is not a max string size" % shape_method
                sys.exit(1)

        self._genMethodBody(type, attr.shape_method, "vector<int>")

        # now generate _size() method if applicable.

        shape_method = attr.shape_method
        size_method = shape_method.replace("_shape", "_size")

        #print >> self.inc, T("  int ${size_method}() const { return ${shape_method}()[0]; }")(locals())

        method_name = attr.accessor.name
        #print >> self.inc, T("  list $method_name() { list l; const int n = ${method_name}_size(); for (int i = 0; i < n; i++) l.append(o->${method_name}(i)); return l; }")(locals())

        return method_name

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
