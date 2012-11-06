#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module FileNameManager...
#
#------------------------------------------------------------------------

"""Dynamically generates the file names from the confoguration parameters

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

from ConfigParametersCorAna import confpars as cp
from Logger                 import logger

#-----------------------------

class FileNameManager :
    """Dynamically generates the file names from the confoguration parameters.
    """

    def __init__ (self) :
        """Constructor.
        @param fname  the file name for output log file
        """

    def path_xtc_pedestals(self) :
        return cp.in_dir_dark.value() + '/' + cp.in_file_dark.value()

    def path_psana_cfg_pedestals(self) :
        return cp.dir_work.value() + '/' + cp.fname_prefix.value() + 'pedestals.cfg'

#-----------------------------

fnm = FileNameManager ()

#-----------------------------

if __name__ == "__main__" :

    print 'path_xtc_pedestals()       : ', fnm.path_xtc_pedestals()
    print 'path_psana_cfg_pedestals() : ', fnm.path_psana_cfg_pedestals()
    print '',
    print '',
    print '',
    
    sys.exit ( 'End of test for FileNameManager' )

#-----------------------------
