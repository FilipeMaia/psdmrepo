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
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.Constant import Constant
from psddl.Package import Package
from psddl.Type import Type

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

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class DdlPsanaDoc ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, output_dir, backend_options ) :

        self.dir = output_dir
        self.top_pkg = backend_options.get('top-package')

        self.psana_inc = backend_options.get('psana-inc', "")

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
                print >>out, '<li><a href="%s">%s</a></li>' % (self._pkgFileName(pkg), pkg.fullName('C++',self.top_pkg))
            print >>out, '</ul>'

    def printTypes(self, ns, out):

        types = self.types(ns)
        if types:
            print >>out, '<h2>Types/classes</h2><ul>'
            for type in types :
                print >>out, '<li>%s</li>' % self._typeRef(type)
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
                print >>out, '<div class="descr"><div class="def">'
                print >>out, '%s = %s' % (const.name, const.value)
                print >>out, '</div>'
                print >>out, '%s' % const.comment
                print >>out, '</div>'


    def printEnum(self, enum, out):
        
        
        print >>out, '<div class="descr"><div class="def" id="enum_%s">' % enum.name
        print >>out, 'Enumeration <font class="enumname">%s</font>' % enum.fullName('C++',self.top_pkg)
        print >>out, '</div>'
        print >>out, "<p>%s</p>" % enum.comment
        print >>out, "<p>Enumerators:<table>"
        for const in enum.constants() :
            print >>out, '<tr>'
            val = ""
            if const.value is not None : val = " = " + const.value
            print >>out, '<td class="const">%s</td>' % const.name
            print >>out, '<td class="const">%s</td>' % val
            print >>out, '<td>%s</td>' % (const.comment)
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

        # loop over packages in the model
        for pkg in self.packages(model) :
            logging.debug("parseTree: package=%s", repr(pkg))
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
        self._htmlHeader(out, "Package %s Reference" % pkgname)
        print >>out, '<h1>Package %s Reference</h1>' % pkgname

        print >>out, pkg.comment


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

        logging.debug("_parseType: type=%s", repr(type))

        typename = type.fullName('C++',self.top_pkg)
        filename = self._typeFileName(type)

        # open output file
        out = file(os.path.join(self.dir, filename), "w")
        self._htmlHeader(out, "Class %s Reference" % typename)
        print >>out, '<h1>Class %s Reference</h1>' % typename

        if type.location:
            include = os.path.basename(type.location)
            include = os.path.splitext(include)[0] + '.h'
            print >>out, '<p>Include: <span class="code">#include "%s/%s"</span></p>' % (self.psana_inc, include)

        if type.base:
            print >>out, "<p>Base class: %s</p>" % self._typeRef(type.base)

        print >>out, type.comment
            

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
            if attr.shape_method:
                rettype = "std::vector&lt;int&gt;"
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
                print >>out, '<div class="descr">'
                print >>out, '<div class="def" id="meth_%s">' % meth[0]
                print >>out, self._methDecl2(*meth)
                print >>out, '</div>'
                print >>out, meth[3]
                print >>out, '</div>'
        

        self._htmlFooter(out)
        out.close()

    def _methDecl(self, name, rettype, args, descr):

        return '<tr><td class="methrettype">%s</td><td class="methdecl">%s(%s)</td></tr>' % \
            (rettype, self._methRef(name), args)

    def _methShapeDescr(self, attr):

        if attr.accessor:
            return """Method which returns the shape (dimensions) of the data returned by 
                %s() method.""" % self._methRef(attr.accessor.name)
        else:
            return """Method which returns the shape (dimensions) of the data member %s.""" % \
                attr.name

    def _methDecl2(self, name, rettype, args, descr):
        return '%s %s(%s)' % (rettype, name, args)

    def _methReturnType(self, meth):

        type = meth.type
        
        if meth.attribute:
            if meth.attribute.shape:
                # array, return type is either a pointer or reference
                typename = type.fullName('C++', self.top_pkg)
                if not type.basic : 
                    typename = "const "+self._typeRef(type)+'&'
                else:                
                    typename = "const "+typename+'*'
            else:
                # non-array
                typename = type.fullName('C++', self.top_pkg)
                if not type.basic : typename = "const "+self._typeRef(type)+'&'

        elif type is None:
            
            typename = 'void'
            
        else:

            typename = type.fullName('C++', self.top_pkg)
            if not type.basic : typename = "const "+self._typeRef(type)+'&'
            
        return typename

    def _methArgs(self, meth):

        type = meth.type
        args = []
        
        if meth.attribute:
            if meth.attribute.shape and not meth.type.basic:
                for i in range(len(meth.attribute.shape.dims)):
                    args.append('uint32_t i%d' % i)

            
        return ', '.join(args)

    def _htmlHeader(self, f, title):
        
        print >>f, '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">'
        print >>f, '<html><head><meta http-equiv="Content-Type" content="text/html;charset=iso-8859-1">'
        print >>f, '<title>%s</title>' % title
        print >>f, '<link href="%s" rel="stylesheet" type="text/css">' % _css_file
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

        return '<a href="%s">%s</a>' % (self._pkgFileName(pkg), pkg.fullName('C++',self.top_pkg))

    def _typeRef(self, type):

        return '<a href="%s">%s</a>' % (self._typeFileName(type), type.fullName('C++',self.top_pkg))

    def _methRef(self, methname):

        return '<a href="#meth_%s">%s</a>' % (methname, methname)



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
