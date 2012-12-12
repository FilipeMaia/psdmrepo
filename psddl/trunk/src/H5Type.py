#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module H5Type...
#
#------------------------------------------------------------------------

"""Class defining schema for single type in HDF5.

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

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class H5Type ( object ) :

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, **kw ) :
        
        self.name = name
        self.package = kw.get('package')         # parent Package object
        self.pstype = kw.get('pstype', None)     # corresponding Type object
        self.datasets = []                       # List of H5Dataset objects
        self.version = kw.get('version', 0)      # version number
        self.included = kw.get('included')
        self.location = kw.get('location')
        self.tags = kw.get('tags', {}).copy()
        
    #-------------------
    #  Public methods --
    #-------------------

    def __str__(self):
        
        return "<H5Type(name=%s, version=%s, datasets=%s)>" % (self.name, self.version, self.datasets)

    def __repr__(self):
        
        return "<H5Type(name=%s, version=%s, datasets=%s)>" % (self.name, self.version, self.datasets)


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
