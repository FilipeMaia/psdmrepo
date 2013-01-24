#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ViewResults...
#
#------------------------------------------------------------------------

"""First look at results

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

@version $Id:$

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
import numpy as np

from ConfigParametersCorAna   import confpars as cp
from Logger                   import logger
from FileNameManager          import fnm
import GlobalUtils            as     gu
from PlotImgSpe               import *

#-----------------------------

class ViewResults :
    """First look at results.
    """

    def __init__ (self, fname=None) :
        """
        @param fname the file name with results
        """

        self.setFileName(fname)

#-----------------------------

    def setFileName(self, fname=None) :
        if fname == None : self.fname = cp.res_fname.value()
        else :             self.fname = fname

#-----------------------------

    def get_cor_array_from_text_file(self) :
        logger.info('get_cor_array_from_text_file: ' + self.fname, __name__)
        #return np.loadtxt(fname, dtype=np.float32)


    def get_cor_array_from_binary_file(self) :
        logger.info('get_cor_array_from_binary_file: ' + self.fname, __name__)
        self.arr = np.fromfile(self.fname, dtype=np.float32)

        img_rows = cp.bat_img_rows.value()
        img_cols = cp.bat_img_cols.value()
        img_size = cp.bat_img_size.value()

        nptau = self.arr.shape[0]/cp.bat_img_size.value()/3
        self.arr.shape = (nptau, 3, img_rows, img_cols)
        logger.info('Set arr.shape = ' + str(self.arr.shape), __name__)
        return self.arr

#-----------------------------

    def get_list_of_tau_from_file(self, fname_tau) :
        #fname_tau = fnm.path_cora_merge_tau()
        logger.info('get_list_of_tau_from_file: ' + fname_tau, __name__)
        return np.loadtxt(fname_tau, dtype=np.uint16)

#-----------------------------

