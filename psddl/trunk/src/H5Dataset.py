#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module H5Dataset...
#
#------------------------------------------------------------------------

"""Class corresponding to HDF5 dataset in the schema

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
class H5Dataset ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, **kw ) :
        
        self.name = kw.get('name')          # dataset name
        self.pstype = kw.get('pstype', None)     # corresponding Type object
        self.attributes = []                # list of H5Attribute objects
        self.tags = kw.get('tags', {}).copy()

    #-------------------
    #  Public methods --
    #-------------------

    def __str__(self):
        
        return "<H5Dataset(name=%s, attributes=%s)>" % (self.name, self.attributes)

    def __repr__(self):
        
        return "<H5Dataset(name=%s, attributes=%s)>" % (self.name, self.attributes)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
