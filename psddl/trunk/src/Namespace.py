#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Namespace...
#
#------------------------------------------------------------------------

"""Class representing namespace.

Namespace in DDL can be either package, type, or enum. All objects
are defined in exactly one namespace. 

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

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class Namespace ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, parent ) :
        self.name = name
        self._parent = parent
        self._children = {}
        self._ordered = []

        if parent: parent.add(self)

    def fullName(self):
        name = self.name
        if self._parent: 
            parent = self._parent.fullName() 
            if parent: name = parent + '.' + name
        return name

    def add(self, obj):
        """ Adds one more object to a namespace, name must not exist yet """

        logging.debug('adding name %s to namespace %s', obj.name, self.fullName())
        
        if self._children.get(obj.name) is not None :
            raise KeyError('name %s already defined in namespace %s' % (obj.name, self.fullName()))
        self._children[obj.name] = obj
        self._ordered.append(obj)

    def lookup(self, namestr, type=None):
        """ Implementation of the name lookup in the namespaces.
        Does recursive search of a name in this namespace and its children.
        If the name is not found then it's passed to parent namespace."""
        
        
        # split it at dots
        name = namestr.split('.')

        # find first level name in this namespace
        obj = self._children.get(name[0])
        
        if obj is None:

            # go ask parent
            if self._parent is not None : 
                obj = self._parent.lookup(namestr, type)
                
        else :
        
            for n in name[1:] :
    
                # it must be a namspace
                if not isinstance(obj, Namespace): 
                    obj = None
                    break
                
                obj = obj._children.get(n)

        # if specific type is requested then check object type
        if type is not None and not isinstance(obj, type) : obj = None
        
        return obj
        
    def packages(self):
        from psddl.Package import Package
        return self.__objects(Package)

    def types(self):
        from psddl.Type import Type
        return self.__objects(Type)

    def enums(self):
        from psddl.Enum import Enum
        return self.__objects(Enum)

    def constants(self):
        from psddl.Constant import Constant
        return self.__objects(Constant)

    def attributes(self):
        from psddl.Attribute import Attribute
        return self.__objects(Attribute)
    
    def methods(self):
        from psddl.Method import Method
        return self.__objects(Method)
    
    def __objects(self, type):
        """Get the list of objects oof given type defined in this namespace"""
        return [ch for ch in self._ordered if isinstance(ch, type)]
        

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
