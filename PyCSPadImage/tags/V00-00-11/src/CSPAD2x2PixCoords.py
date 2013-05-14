#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CSPAD2x2PixCoords...
#
#------------------------------------------------------------------------

"""
This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 2013-02-01$

@author Mikhail S. Dubrovin
"""

#--------------------------------
#  Module's version from CVS --
#--------------------------------
__version__ = "$Revision: 4 $"
# $Source$
#--------------------------------

import sys
import math
import numpy as np
from time import time

#import matplotlib.pyplot as plt

from PixCoords2x1 import *
import GlobalGraphics as gg # For test purpose in main only

#------------------------------

class CSPAD2x2PixCoords (PixCoords2x1) :
    """Self-sufficient class for generation of CSPad2x2 pixel coordinate array without data base (WODB)"""

    sects = 2 # Total number of sections in quad
    xc_um_def    = np.array([198., 198.]) * PixCoords2x1.pixs # 109.92
    yc_um_def    = np.array([ 95., 308.]) * PixCoords2x1.pixs # 109.92
    tilt_deg_def = np.array([  0.,   0.])

##------------------------------

    def __init__ (sp, calib=None, xc_um=None, yc_um=None, tilt_deg=None, use_wide_pix_center=False) :
        #print 'CSPAD2x2PixCoords.__init__(...)'

        PixCoords2x1.__init__ (sp, use_wide_pix_center)

        if calib == None :

            if xc_um == None : xc = sp.xc_um_def
            else             : xc = sp.xc_um

            if yc_um == None : yc = sp.yc_um_def
            else             : yc = sp.yc_um

            tilt       = tilt_deg

        else :

            [xc,yc,zc] = calib.getCalibPars('center') * PixCoords2x1.pixs # 109.92
            tilt       = calib.getCalibPars('tilt')

            print 'USE xc, yc, zc =', xc, yc, zc

        sp.make_cspad2x2_pix_coordinate_arrays (xc, yc, tilt)

#------------------------------

    def make_cspad2x2_pix_coordinate_arrays (sp, xc_um, yc_um, tilt_deg=None) : # All lists of size[2]
        """Makes [2,185,388] cspad pixel x and y coordinate arrays"""        
        #sp.make_maps_of_2x1_pix_coordinates()

        sp.x_pix_um = np.zeros((sp.sects,sp.rows,sp.cols), dtype=np.float32)
        sp.y_pix_um = np.zeros((sp.sects,sp.rows,sp.cols), dtype=np.float32)

        angle_deg = [180,180]
        if tilt_deg != None : angle_deg += tilt_deg
 
        for sect in range(sp.sects) :

            angle_rad = math.radians(angle_deg[sect])                
            S,C = math.sin(angle_rad), math.cos(angle_rad)
            Xrot, Yrot = rotation(sp.x_map2x1_um, sp.y_map2x1_um, C, S)

            sp.x_pix_um[sect][:] =  Xrot + xc_um[sect]
            sp.y_pix_um[sect][:] =  Yrot + yc_um[sect]

        sp.x_pix_um -= sp.x_pix_um.min()
        sp.y_pix_um -= sp.y_pix_um.min() 

#------------------------------

    def get_cspad2x2_pix_coordinate_arrays_um (sp) : 
        return sp.x_pix_um, sp.y_pix_um


    def get_cspad2x2_pix_coordinate_arrays_pix (sp) : 
        return sp.x_pix_um/sp.pixs, sp.y_pix_um/sp.pixs

#------------------------------

    def print_cspad2x2_coordinate_arrays(sp) :
        print 'sp.x_pix_um:\n',      sp.x_pix_um
        print 'sp.x_pix_um.shape =', sp.x_pix_um.shape
        print 'sp.y_pix_um\n',       sp.y_pix_um
        print 'sp.y_pix_um.shape =', sp.y_pix_um.shape

#------------------------------

def data2x2ToTwo2x1(arr2x2) :
    """Converts array shaped as CSPAD2x2 data (185,388,2)
    to two 2x1 arrays with shape=(2,185,388)
    """
    return np.array([arr2x2[:,:,0], arr2x2[:,:,1]])

#------------------------------

def two2x1ToData2x2(arrTwo2x1) :
    """Converts array shaped as two 2x1 arrays (2,185,388)
    to CSPAD2x2 data shape=(185,388,2)
    """
    arr2x2 = np.array(zip(arrTwo2x1[0].flatten(), arrTwo2x1[1].flatten()))
    arr2x2.shape = (185,388,2)
    return arr2x2

#------------------------------

def getImage(X,Y,W=None) :
    """Makes image from X, Y coordinate arrays and associated weights.
    """
    xsize = X.max()
    ysize = Y.max()
    if W==None : weights = None
    else       : weights = W.flatten()

    H,Xedges,Yedges = np.histogram2d(X.flatten(), Y.flatten(), bins=[xsize,ysize], range=[[-0.5,xsize-0.5],[-0.5,ysize-0.5]], normed=False, weights=weights) 
    return H

#------------------------------
#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------
#------------------------------

def main_test_cspad2x2() :

    xc_arr   = np.array([198., 198.]) * PixCoords2x1.pixs # 109.92
    yc_arr   = np.array([ 95., 308.]) * PixCoords2x1.pixs # 109.92
    tilt_arr = np.array([  0.,   0.])

    print 'xc_um   : ', xc_arr
    print 'yc_um   : ', xc_arr
    print 'tilt_deg: ', tilt_arr

    t0_sec = time()
    #w = CSPAD2x2PixCoords(xc_um, yc_um, tilt_deg)
    w = CSPAD2x2PixCoords(tilt_deg=tilt_arr)
    print 'Consumed time for coordinate arrays (sec) =', time()-t0_sec

    #w.print_cspad2x2_coordinate_arrays()
    X,Y = w.get_cspad2x2_pix_coordinate_arrays_pix ()

    #print 'X(pix) :\n', X
    print 'X.shape =\n', X.shape

    H = getImage(X,Y,W=None)

    gg.plotImageLarge(H, range=(-1, 2), figsize=(12,11))
    gg.show()

#------------------------------
 
if __name__ == "__main__" :
    main_test_cspad2x2()
    sys.exit ( 'End of test.' )

#------------------------------
