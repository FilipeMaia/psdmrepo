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

@see RelatedModule

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


    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, template) :
        string.Template.__init__(self, template)

    #-------------------
    #  Public methods --
    #-------------------

    def __call__(self, *args, **kws) :
        return self.substitute(*args, **kws)

    def __getitem__(self, obj) :
        """Support for template[object]"""
        return self.substitute(obj.__dict__)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
