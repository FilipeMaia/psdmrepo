#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Template...
#
#------------------------------------------------------------------------

"""Brief one-line description of the module.

Following paragraphs provide detailed description of the module, its
contents and usage. This is a template module (or module template:)
which will be used by programmers to create new Python modules.
This is the "library module" as opposed to executable module. Library
modules provide class definitions or function definitions, but these
scripts cannot be run by themselves.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
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
#import PkgPackage.PkgModule
#from PkgPackage.PkgModule import PkgClass


#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

_OP_X = 'Y'

def _localMethod ( param ) :
    """Still have to describe what it is, even it is local."""
    pass


#------------------------
# Exported definitions --
#------------------------

# these definitions are visible to the clients

OP_ONE = 1             # comment it please
OP_NONE = None         # comment it please

def foo ( x ) :
    """Brief description of a foo().

    Long description - in fact everyone knows already what foo() does.
    """
    if x not in ( OP_ONE, OP_NONE ) :
        return None
    return (None,)


#---------------------
#  Class definition --
#---------------------
class Template ( object ) :
    """Brief description of a class.

    Full description of this class. The whole purpose of this class is 
    to serve as an example for LCLS users. It shows the structure of
    the code inside the class. Class can have class (static) variables, 
    which can be private or public. It is good idea to define constructor 
    for your class (in Python there is only one constructor). Put your 
    public methods after constructor, and private methods after public.

    @see BaseClass
    @see OtherClass
    """

    #--------------------
    #  Class variables --
    #--------------------
    publicStaticVariable = 0 
    __privateStaticVariable = "A string"

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, x, y ) :
        """Constructor.

        Explanation of what it does. So it does that and that, and also 
        that, but only if x is equal to that and y is not None.

        @param x   first parameter
        @param y   second parameter
        """

        # call ctor for base class explicitly
        

        # define instance variables
        self.__x = x                  # private 
        self._p = None                # "protected"
        self.y = y                    # public

    #-------------------
    #  Public methods --
    #-------------------

    def myMethod ( self, x, y ) :
        """Brief description of method.

        Explanation of what it does.
        @param x   first parameter
        @param y   second parameter
        @return    return value
        """
    	if self.__x > x :
            return self.y
        else:
            self._p = self.__myPrivateMethod ( y )
            return self._p

    #--------------------------------
    #  Static/class public methods --
    #--------------------------------

    @staticmethod
    def myStaticMethod ( x, y ) :
        """Brief description of method.

        Explanation of what it does.
        @param x   second parameter
        @param y   second parameter
        @return    return value
        """
        return x**2+y**2

    @classmethod
    def myClassMethod ( cls, x ) :
        """Brief description of method.

        Explanation of what it does.
        @param cls class object of this class or subclass
        @param x   second parameter
        @return    return value
        """
        return cls.x**2-x**2

    #--------------------
    #  Private methods --
    #--------------------

    def __myPrivateMethod ( self, y ) :
        """Brief description of method.

        Explanation of what it does.
        @param y   second parameter
        @return    return value
        """
        return y**2

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
