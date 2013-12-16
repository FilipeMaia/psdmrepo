#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module DdlPsanaDoc...
#
#------------------------------------------------------------------------

"""DDL back-end which generates documentation for PSANA interfaces 

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see DdlPsanaInterfaces

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
import cgi

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.Constant import Constant
from psddl.Package import Package
from psddl.Type import Type
from psddl.Template import Template as T

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_css_file = "psana.css"

_css = """
BODY,H1,H2,H3,H4,H5,H6,P,CENTER,TD,TH,UL,DL,DIV {
    font-family: Geneva, Arial, Helvetica, sans-serif;
}

BODY,TD {
       font-size: 90%;
}
H1 {
    text-align: center;
       font-size: 160%;
}
H2 {
       font-size: 120%;
}
H3 {
       font-size: 100%;
}


A {
       text-decoration: none;
       font-weight: bold;
       color: #000080;
}
A:visited {
       text-decoration: none;
       font-weight: bold;
       color: #000080
}
A:hover {
    text-decoration: none;
    background-color: #E0E0E0;
}

dt { 
margin: 2px;
padding: 2px;
}

.const {font-family:monospace;}
.code {font-family:monospace;}

.methrettype {text-align:right;}

div.def {
background-color: #E0E0E0;
margin: 0px;
padding: 2px;
line-height: 140%;
border: 1px solid #000000;
}

div.descr {
margin: 10px;
padding: 2px;
line-height: 140%;
border: 1px solid #000000;
}

"""

def _typename(type):
    
    return type.fullName('C++')

def _typedecl(type):
    typename = _typename(type)
    if not type.basic : typename = "const "+typename+'&'
    return typename

def _esc(s):
    if type(s) == type({}):
        return dict([(k, _esc(v)) for k, v in s.iteritems()])
    elif type(s) == type(""):
        return cgi.escape(s)
    else:
        return s

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class DdlPsanaDoc ( object ) :

    @staticmethod
    def backendOptions():
        """ Returns the list of options supported by this backend, returned value is 
        either None or a list of triplets (name, type, description)"""
        return [
            ('psana-inc', 'PATH', "directory for Psana includes, default: psddl_psana"),
            ]


    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, backend_options, log ) :
        '''Constructor
        
           @param backend_options  dictionary of options passed to backend
           @param log              message logger instance
        '''

        self.dir = backend_options['global:output-dir']
        self.top_pkg = backend_options.get('global:top-package')

        self.psana_inc = backend_options.get('psana-inc', "psddl_psana")

        self._log = log

    #-------------------
    #  Public methods --
    #-------------------

    def packages(self, ns):
        """returns sorted package list"""
        packages = ns.packages()[:]
        packages.sort(key=lambda pkg: pkg.fullName('C++',self.top_pkg))
        return packages

    def types(self, ns):
        """returns sorted types list"""
        types = ns.types()[:]
        types.sort(key=lambda t: t.fullName('C++',self.top_pkg))
        return types

    def methods(self, type):
        """returns sorted methods list"""
        methods = [m for m in type.methods() if m.name != '_sizeof']
        methods.sort(key=lambda t: t.name)
        return methods

    def printPackages(self, ns, out):

        packages = self.packages(ns)
        if packages:
            print >>out, '<h2>Packages/namespaces</h2><ul>'
            for pkg in packages :
                href = _esc(self._pkgFileName(pkg))
                name = _esc(pkg.fullName('C++',self.top_pkg))
                print >>out, T('<li><a href="$href">$name</a></li>')(locals())
            print >>out, '</ul>'

    def printTypes(self, ns, out):

        types = self.types(ns)
        if types:
            print >>out, '<h2>Types/classes</h2><ul>'
            for type in types :
                print >>out, T('<li>$ref</li>')(ref=self._typeRef(type))
            print >>out, '</ul>'

    def printConstants(self, ns, out):

        constants = ns.constants()[:]
        
        if isinstance(ns, Type):
            if ns.version is not None: 
                comment="XTC type version number"
                c = Constant("Version", ns.version, None, comment=comment)
                constants.insert(0, c)
            if ns.type_id is not None: 
                comment="XTC type ID value (from Pds::TypeId class)"
                c = Constant("TypeId", "Pds::TypeId::"+ns.type_id, None, comment=comment)
                constants.insert(0, c)
        
        if constants:
            print >>out, '<h2>Constants</h2>'
            for const in constants:
                print >>out, T('<div class="descr"><div class="def">$name = $value</div>$comment</div>')\
                        (_esc(const.__dict__))


    def printEnum(self, enum, out):
        
        
        print >>out, T('<div class="descr"><div class="def" id="enum_$name">')[enum]
        print >>out, T('Enumeration <font class="enumname">$name</font>')(name=_esc(enum.fullName('C++',self.top_pkg)))
        print >>out, '</div>'
        print >>out, T("<p>$comment</p>")(_esc(enum.__dict__))
        print >>out, "<p>Enumerators:<table>"
        for const in enum.constants() :
            print >>out, '<tr>'
            val = ""
            if const.value is not None : val = " = " + const.value
            print >>out, T('<td class="const">$name</td>')(_esc(const.__dict__))
            print >>out, T('<td class="const">$value</td>')(value=_esc(val))
            print >>out, T('<td>$comment</td>')(_esc(const.__dict__))
            print >>out, '</tr>'
        print >>out, "</table></p>"
        print >>out, '</div>'

    def printEnums(self, ns, out):

        enums = ns.enums()
        if enums:
            print >>out, '<h2>Enumeration types</h2><dl>'
            for enum in enums:
                self.printEnum(enum, out)
            print >>out, '</dl>'

    def parseTree ( self, model ) :
        
        # open output file
        out = file(os.path.join(self.dir, "index.html"), "w")
        self._htmlHeader(out, "Psana Data Interfaces Reference")
        print >>out, '<h1>Psana Data Interfaces Reference</h1>'
        
        self.printPackages(model, out)

        self.printConstants(model, out)

        self.printEnums(model, out)
        
        self.printTypes(model, out)

        # loop over packages in the model
        for pkg in self.packages(model) :
            self._log.debug("parseTree: package=%s", repr(pkg))
            self._parsePackage(pkg)

        self._htmlFooter(out)
        out.close()
        
        # write CSS
        out = file(os.path.join(self.dir, _css_file), "w")
        out.write(_css)
        out.close()

    def _parsePackage(self, pkg):

        pkgname = pkg.fullName('C++',self.top_pkg)
        filename = self._pkgFileName(pkg)

        # open output file
        out = file(os.path.join(self.dir, filename), "w")
        self._htmlHeader(out, T("Package $name Reference")(name=_esc(pkgname)))
        print >>out, T('<h1>Package $name Reference</h1>')(name=_esc(pkgname))

        print >>out, _esc(pkg.comment)


        self.printPackages(pkg, out)
            
        self.printConstants(pkg, out)

        self.printEnums(pkg, out)
        
        self.printTypes(pkg, out)

        # loop over packages and types
        for type in pkg.types() :
            self._parseType(type)

        self._htmlFooter(out)
        out.close()

    def _parseType(self, type):

        self._log.debug("_parseType: type=%s", repr(type))

        typename = type.fullName('C++',self.top_pkg)
        filename = self._typeFileName(type)

        # open output file
        out = file(os.path.join(self.dir, filename), "w")
        self._htmlHeader(out, T("Class $name Reference")(name=_esc(typename)))
        print >>out, T('<h1>Class $name Reference</h1>')(name=_esc(typename))

        if type.location:
            include = os.path.basename(type.location)
            include = os.path.splitext(include)[0] + '.h'
            repourl = T("https://pswww.slac.stanford.edu/trac/psdm/browser/psdm/$package/trunk/include/$header")\
                    (package=self.psana_inc, header=include)
            print >>out, T('<p>Include: <span class="code">#include "<a href="$href">$package/$header</a>"</span></p>')\
                    (href=repourl, package=_esc(self.psana_inc), header=_esc(include))

        if type.base:
            print >>out, T("<p>Base class: $base</p>")(base=self._typeRef(type.base))

        print >>out, _esc(type.comment)
            

        self.printConstants(type, out)

        self.printEnums(type, out)
        
        # build the list of all methods
        mlist = []
        for meth in self.methods(type):
            if meth.access == 'public':
                rettype = self._methReturnType(meth)
                args = self._methArgs(meth)
                mlist.append((meth.name, rettype, args, meth.comment))

        # X_shape methods
        for attr in type.attributes():
            if attr.shape_method and attr.accessor and (attr.type.name == 'char' or not attr.type.value_type):
                rettype = "std::vector<int>"
                args = ""
                descr = self._methShapeDescr(attr)
                mlist.append((attr.shape_method, rettype, args, descr))

        
        mlist.sort()
        
        if mlist:
            
            print >>out, '<h2>Member Functions</h2>'
            print >>out, '<div class="descr">'
            print >>out, '<table class="methods">'
            
            for meth in mlist:
                print >>out, self._methDecl(*meth)
            print >>out, '</table></div>'            

            print >>out, '<h2>Member Functions Reference</h2>'
            
            for meth in mlist:
                print >>out, T('<div class="descr"><div class="def" id="meth_$name">$decl</div>$descr</div>')\
                        (name=meth[0], decl=self._methDecl2(*meth), descr=_esc(meth[3]))
        

        self._htmlFooter(out)
        out.close()

    def _methDecl(self, name, rettype, args, descr):

        return T('<tr><td class="methrettype">$type</td><td class="methdecl">$name($args)</td></tr>')\
            (type=_esc(rettype), name=self._methRef(_esc(name)), args=_esc(args))

    def _methShapeDescr(self, attr):

        if attr.accessor:
            return """Method which returns the shape (dimensions) of the data returned by 
                %s() method.""" % self._methRef(_esc(attr.accessor.name))
        else:
            return """Method which returns the shape (dimensions) of the data member %s.""" % \
                _esc(attr.name)

    def _methDecl2(self, name, rettype, args, descr):
        return T('$type $name($args)')(type=_esc(rettype), name=_esc(name), args=_esc(args))

    def _methReturnType(self, meth):

        if meth.attribute:
            
            if not meth.attribute.shape:
                
                # attribute is a regular non-array object, 
                # return value or reference depending on what type it is
                typename = _typedecl(meth.attribute.type)

            elif meth.attribute.type.name == 'char':
                
                # char array is actually a string
                typename = "const char*"
                
            elif meth.attribute.type.value_type :
                
                # return ndarray
                typename = T("ndarray<const $type, $rank>")(type=_typename(meth.attribute.type), rank=len(meth.attribute.shape.dims))

            else:

                # array of any other types
                typename = _typedecl(meth.attribute.type)

        elif meth.type is None:
            
            typename = 'void'
            
        else:

            typename = _typedecl(meth.type)
            if meth.rank > 0:
                typename = T("ndarray<const $type, $rank>")(type=typename, rank=meth.rank)
            
        return typename

    def _methArgs(self, meth):

        args = []
        
        if meth.attribute:

            if meth.attribute.shape and not meth.type.basic:
                for i in range(len(meth.attribute.shape.dims)):
                    args.append('uint32_t i%d' % i)

        elif meth.args:
            
            self._log.debug("_methArgs: meth=%s args=%s", meth.name, meth.args)
            for arg in meth.args:
                args.append('%s %s' % (arg[1].name, arg[0]))
            
        return ', '.join(args)

    def _htmlHeader(self, f, title):
        
        print >>f, '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">'
        print >>f, '<html><head><meta http-equiv="Content-Type" content="text/html;charset=iso-8859-1">'
        print >>f, T('<title>$title</title>')(locals())
        print >>f, T('<link href="$href" rel="stylesheet" type="text/css">')(href=_css_file)
        print >>f, '</head><body>'

    def _htmlFooter(self, f):
        
        print >>f, '</body></html>'

    def _pkgFileName(self, pkg):

        pkgname = pkg.fullName(None,self.top_pkg)
        return "pkg."+pkgname+".html"

    def _typeFileName(self, type):

        typename = type.fullName(None,self.top_pkg)
        return "type."+typename+".html"

    def _pkgRef(self, pkg):

        return self._href(self._pkgFileName(pkg), _esc(pkg.fullName('C++',self.top_pkg)))

    def _typeRef(self, type):

        return self._href(self._typeFileName(type), _esc(type.fullName('C++',self.top_pkg)))

    def _methRef(self, methname):

        return self._href('#meth_'+methname, _esc(methname))

    def _href(self, href, name):

        return T('<a href="$href">$name</a>')(locals())



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
