#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Template...
#
#------------------------------------------------------------------------

"""Wrapper for standard string.Template class

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

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
import string

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

# these definitions are visible to the clients

#---------------------
#  Class definition --
#---------------------
class Template(string.Template) :
    """
    Wrapper class for string.Template, it adds two methods to Template class
    which simplify code calling Template.substitute() method. The first method
    is __call__() which makes template instance callable; this method is 
    equivalent to substitute() and provides shorter notation::
    
      res = Template("$name -> $code")(name=name, code=code)
      res = Template("$name -> $code")({'name': name, 'code': code})
      
    Second method uses indexing notation to simplify substitution based on 
    object attributes::

      res = Template("$name -> $code")[object]
      
    In this case object must have attributes *name* and *code*. This is 
    equivalent to:

      res = Template("$name -> $code").substitute(object.__dict__)
      
    """

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, template) :
        string.Template.__init__(self, template)

    #-------------------
    #  Public methods --
    #-------------------

    def __call__(self, *args, **kws) :
        """
        self(*args, **kwargs) -> string
        
        Shorthand for substitute(*args, **kwargs), see string.Template documentation.
        """
        return self.substitute(*args, **kws)

    def __getitem__(self, obj) :
        """
        self[object] -> string
        
        Object must have attributes with names identical to identifiers
        used in template string.
        """
        return self.substitute(obj.__dict__)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
