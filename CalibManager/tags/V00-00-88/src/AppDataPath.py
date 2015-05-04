#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module AppDataPath...
#
#------------------------------------------------------------------------

"""AppDataPath class represents a path to a file that can be found in
one of the $SIT_DATA locations.

@version $Id$

@author Andy Salnikov
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
import os

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

#---------------------
#  Class definition --
#---------------------
class AppDataPath ( object ) :
    """
    AppDataPath class represents a path to a file that can be found in one of the $SIT_DATA locations.
    """

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, relPath) :
        """Constructor takes relative file path."""
        self.m_path = ""
        
        dataPath = '../../data:' + os.getenv("SIT_DATA")
        if not dataPath: return
            
        for dir in dataPath.split(':'):
            path = os.path.join(dir, relPath)
            if os.path.exists(path):
                self.m_path = path
                break
            
    #-------------------
    #  Public methods --
    #-------------------

    def path(self) :
        """Returns path of the existing file or empty string"""
        return self.m_path

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
