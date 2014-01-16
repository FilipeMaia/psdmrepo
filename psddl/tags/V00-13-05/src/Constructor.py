#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Constructor...
#
#------------------------------------------------------------------------

"""Class describing type's constructor.


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
from psddl.Attribute import Attribute
from psddl.Bitfield import Bitfield

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------

#
# Class describing one constructor argument
#
class CtorArg(object):
    
    def __init__(self, name, dest, type = None, method = None, expr = None):
        '''
        name is a string
        dest must be an attribute or bitfield object
        type is argument type, if missing will be the same as dest type
        method is method object, if missing will be dest accessor
        expr is expression, if none then argument name is used
        '''
        self.name = name
        self.dest = dest
        self.type = type or (dest and dest.type)
        self.method = method or (dest and dest.accessor)
        self.expr = expr or name
        self.base = False   # true means that it is forwarded to base class

#
# Class describing one constructor initializer
#
class CtorInit(object):
    
    def __init__(self, dest, expr):
        '''
        dest must be an attribute or bitfield object
        expr is expression string
        '''
        self.dest = dest
        self.expr = expr

class Constructor ( object ) :
    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent, **kw ) :
        
        self.parent = parent

        self._args = kw.get('args', [])
        self.attr_init = kw.get('attr_init', [])
        self.comment = kw.get('comment')
        self.access = kw.get('access', "public")
        self.tags = kw.get('tags', {}).copy()
        
        self.parent.ctors.append(self)

        self._cargs = None

    #-------------------
    #  Public methods --
    #-------------------

    @property
    def args(self):
        '''
        Get the list of constructor arguments.
        
        Returns the list of CtorArg objects 
        '''
        
        if self._cargs is not None: return self._cargs
        
        # build a list of arguments to ctor
        if self._args :
           
            # use pre-defined argument list
            self._cargs = self._args
            
        else:

            # define new arg list            
            self._cargs = []
            
            if 'auto' in self.tags:
                
                # if base class exists then get the arguments from it first
                if self.parent.base:
                    bctor = [c for c in self.parent.base.ctors if 'auto' in c.tags]
                    if bctor:
                        self._cargs = bctor[0].args[:]
                        # update destination to signify base class
                        for arg in self._cargs:
                            arg.base = True
                
                # make one argument per attribute or bitfield with accessor
                for item in self.parent.attributes_and_bitfields():
                    if item.accessor:
                        name = "arg_"+item.name
                        self._cargs.append(CtorArg(name, item))

        return self._cargs
 

    def __str__(self):
        return "<%s(%s)>" % (self.parent.name, self.args)

    def __repr__(self):
        return "<%s(%s)>" % (self.parent.name, self.__dict__)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
