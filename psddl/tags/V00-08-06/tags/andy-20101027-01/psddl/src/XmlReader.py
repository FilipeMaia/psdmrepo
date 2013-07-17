#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module XmlReader...
#
#------------------------------------------------------------------------

"""Class which validates DDL.

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
import elementtree.ElementTree as ET

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.Package import Package
from psddl.Attribute import Attribute
from psddl.Method import Method
from psddl.Type import Type
from psddl.TypeLib import TypeLib

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_const_tag = 'const'
_enum_tag = 'enum'
_enum_const_tag = 'enum-const'

def _findType(typename, pkg):
    
    typelib = TypeLib()
    
    w = typename.rsplit('.', 1)
    if len(w) > 1:
        pkgname = '.'.join(w[:-1])
        typename = w[-1]
        pkg = typelib.findPackage(pkgname)
        if not pkg : return None
        return typelib.findType(typename, pkg)
    else:
        return typelib.findType(typename, pkg)

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class XmlReader ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, xmlfile ) :
        
        self.file = xmlfile

    #-------------------
    #  Public methods --
    #-------------------

    def read( self ) :
        """Parse XML tree and return the list of packages"""

        typelib = TypeLib()

        model = []

        # read element tree from file
        et = ET.parse(self.file)
        
        # get root element
        root = et.getroot() 

        # root must be psddl
        if root.tag != 'psddl' :
            raise TypeError('Root element must be psddl and not '+root.tag)
        
        # scan its direct children
        for pkgel in list(root) :

            # all children must be packages
            if pkgel.tag != 'package' : 
                raise TypeError('Root element must be contain packages and not '+pkgel.tag)
            
            # package must have a name
            pkgname = pkgel.get('name')
            if not pkgname: raise ValueError('Package element missing name')

            # make package object
            pkg = Package(pkgname)
            model.append(pkg)
            typelib.addPackage(pkg)

            for subpkgel in list(pkgel) :

                # package children can be types, constants or enums
                if subpkgel.tag == 'pstype' :

                    self._parseType(subpkgel, pkg)

                elif subpkgel.tag == _const_tag :

                    self._parseConstant(subpkgel, pkg)

                elif subpkgel.tag == _enum_tag :

                    self._parseEnum(subpkgel, pkg)

                else:
                    
                    raise TypeError('Package contains unexpected element: '+subpkgel.tag)


        # return the model
        return model

    def _parseType(self, typeel, pkg):

        typelib = TypeLib()

        # every type must have a name
        typename = typeel.get('name')
        if not typename: raise ValueError('pstype element missing name')

        # make new type object
        type = Type(typename,
                    version = typeel.get('version'),
                    type_id = typeel.get('type_id'),
                    levels = typeel.get('levels', []),
                    object_size = typeel.get('object-size'),
                    total_size = typeel.get('total-size'),
                    pack = typeel.get('pack'),
                    comment = typeel.text.strip(),
                    package = pkg )
        pkg.types.append(type)
        typelib.addType(type)

        # get attributes
        for propel in list(typeel) :

            if propel.tag == _const_tag :
                
                self._parseConstant(propel, type)
                
                
            elif propel.tag == 'enum' :
                
                self._parseEnum(propel, type)
                
            elif propel.tag == 'attribute' :
        
                # every attribute must have a name
                attrname = propel.get('name')
                if not attrname: raise ValueError('attribute element missing name')
                
                # find type object
                atypename = propel.get('type')
                if not atypename: raise ValueError('attribute element missing type')
                atype = _findType(atypename, pkg)
                if not atype: raise ValueError('attribute element has unknown type '+atypename)

                # get offset, make a number from it if possible
                attroffset = propel.get('offset')
                if attroffset and attroffset.isdigit(): attroffset = int(attroffset)

                # create attribute
                attr = Attribute( attrname,
                                  type = atype,
                                  parent = type, 
                                  dimensions = propel.get('dimensions'), 
                                  comment = propel.text.strip(), 
                                  offset = attroffset,
                                  access = propel.get('access') )
                type.attributes.append(attr)

                # accessor method for it
                accessor = propel.get('accessor')
                if accessor :
                    method = Method(accessor, 
                                    attribute = attr, 
                                    parent = type, 
                                    type = None)
                    type.methods.append(method)

            elif propel.tag == 'xtc-config' :
                
                # every attribute must have a name
                cfgname = propel.get('name')
                if not cfgname: raise ValueError('xtc-config element missing name')
                
                # find type for this name
                cfg = _findType(cfgname, pkg)
                if not cfg: raise ValueError('unknown xtc-config name: '+cfgname)
                
                # add it to the list of config classes
                type.xtcConfig.append(cfgname)
                
            elif propel.tag == 'repeat' :
                
                # every attribute must have a name
                count = propel.get('count')
                if not count: raise ValueError('repeat element missing count attribute')
                
                # add it to the list of config classes
                type.repeat = count
                
                        
    def _parseConstant(self, constel, parent):

        # every constant must have a name and value
        cname = constel.get('name')
        if not cname: raise ValueError('const element missing name')
        cval = constel.get('value')
        if not cval: raise ValueError('const element missing value')
        parent.constants.append((cname, cval))
            
    def _parseEnum(self, enumel, parent):
            
        enums = []
        for econst in list(enumel):
            if econst.tag != _enum_const_tag : raise ValueError('expecting %s tag'%_enum_const_tag)
            enums.append((econst.get('name'), econst.get('value')))
        parent.enums.append((enumel.get('name'), enums))

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
