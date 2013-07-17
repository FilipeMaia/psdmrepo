#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module TypeLib...
#
#------------------------------------------------------------------------

"""Description of various types and their properties.

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

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.Type import Type

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class _TypeLib ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :

        self._typedefs = {}
        self._packages = {}

        self.addType(Type("char", size=1, align=1))
        self.addType(Type("int8_t", size=1, align=1))
        self.addType(Type("uint8_t", size=1, align=1))
        self.addType(Type("int16_t", size=2, align=2))
        self.addType(Type("uint16_t", size=2, align=2))
        self.addType(Type("int32_t", size=4, align=4))
        self.addType(Type("uint32_t", size=4, align=4))
        self.addType(Type("int64_t", size=8, align=8))
        self.addType(Type("float", size=4, align=4))
        self.addType(Type("double", size=8, align=8))

    #-------------------
    #  Public methods --
    #-------------------
    def addPackage(self, pkg):
        self._packages[pkg.name] = pkg

    def addType(self, type):
        pkgname = None 
        if type.package: pkgname = type.package.name
        self._typedefs[(type.name, pkgname)] = type

    def findPackage(self, name):
        return self._packages.get(name)

    def findType(self, typename, package):
        pkgname = None
        if package:
            # package can be package object or package name
            try:
                pkgname = package.name
            except:
                pkgname = package
        type = self._typedefs.get((typename, pkgname))
        if not type and pkgname:
            # try to search in global namespace too
            type = self._typedefs.get((typename, None))
        return type


_typeLib = _TypeLib()
def TypeLib() :
    return _typeLib

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
