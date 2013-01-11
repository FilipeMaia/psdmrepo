#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module RequestWithMethod...
#
#------------------------------------------------------------------------

"""Special request class for urllib2 with support for HTTP method.

This software was developed for the LUSI project.  If you use all or 
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
import urllib2

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
class RequestWithMethod ( urllib2.Request ) :
    
    def __init__(self, method, *args, **kwargs):
        self.method = method
        urllib2.Request.__init__(self,*args, **kwargs)

    def get_method(self):
        return self.method

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
