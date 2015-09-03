#!/usr/bin/env python
#------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ex_cspad_img...
#
#------------------------------------------------------------------------

"""This module provides an examples of how to get and plot CSPad image

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see PyCSPadImage.CalibPars, PyCSPadImage.CSPADPixCoords

@version $Id: 2014-07-02$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#----------
#  Imports 
#----------
import os
import sys
import numpy as np

import PyCSPadImage.CalibPars          as calp
import PyCSPadImage.CSPADPixCoords     as pixc
import PyCSPadImage.GlobalMethods      as gm
import pyimgalgos.GlobalGraphics       as gg

#----------------------------------------------

def get_cspad_pixel_coordinate_index_arrays(path_calib='./', runnum=1) :
    """Uses cspad geometry calibration parameters and 
       returns flatten cspad iX and iY arrays (size=32*185*388) of pixel coordinates [in pix]
    """
    calibstore = calp.CalibPars( path=path_calib, run=runnum )
    pixcoords = pixc.CSPADPixCoords(calibstore)
    pixcoords.print_cspad_geometry_pars()
    iX,iY = pixcoords.get_cspad_pix_coordinate_arrays_pix()
    iX -= iX.min()
    iY -= iY.min()
    return iX.flatten(), iY.flatten()

#----------------------------------------------

def plot_image_from_index_arrays(iX, iY, W, amps=None) :
    """Gets flatten pixel coordinate X, Y index arrays and array W of weights of the same size and plots image 
    """
    image = gg.getImageFromIndexArrays(iX, iY, W=W.flatten())
    axis = gg.plotImageLarge(image, amp_range=amps, figsize=(12,11))
    gg.show()
 
#----------------------------------------------

if __name__ == "__main__" :

    if len(sys.argv) < 2 :
        fname = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-05-15/cspad-arr-cxid2714-r0023-lysozyme-rings.txt'
    else :
        fname = sys.argv[1]

    print 'Plot cspad image using ndarray from file:\n%s' % fname

    nda = gm.getCSPadArrayFromFile(fname, dtype=np.float32, shape=(32, 185, 388)) 

    path_calib = '/reg/d/psdm/xpp/xppa7714/calib/CsPad::CalibV1/XppGon.0:Cspad.0'
    runnum = 251
    amps   = (0,2200)

    iX, iY = get_cspad_pixel_coordinate_index_arrays(path_calib, runnum)

    plot_image_from_index_arrays(iX, iY, nda, amps)

    sys.exit ( 'End of %s' % sys.argv[0] )

#----------------------------------------------
