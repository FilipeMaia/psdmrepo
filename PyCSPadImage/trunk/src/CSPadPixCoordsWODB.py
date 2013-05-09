#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CSPadPixCoordsWODB...
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
import GlobalGraphics     as gg # For test purpose in main only
#------------------------------

class CSPadPixCoordsWODB (PixCoords2x1) :
    """Self-sufficient class for generation of CSPad pixel coordinate array without data base (WODB)"""

    quads = 4 # Total number of quads in cspad
    sects = 8 # Total number of sections in quad

    # Orientation of sect:   0   1     2     3     4     5     6     7
    orient_opt = np.array ( [0., 0., 270., 270., 180., 180., 270., 270.] )

    # Orientation of quad:         0              1           2              3 
    orient_def = np.array ( [orient_opt+90, orient_opt, orient_opt-90, orient_opt-180] )
 
    tilt_def = np.zeros((quads,sects), dtype=np.float32)

#------------------------------

    def __init__ (sp, xc_um=None, yc_um=None, orient_deg=None, tilt_deg=None, use_wide_pix_center=True) :
        print 'CSPadPixCoordsWODB.__init__(...)'

        PixCoords2x1.__init__ (sp, use_wide_pix_center)

        if xc_um      == None : return
        if yc_um      == None : return

        if orient_deg == None : orient = sp.orient_def
        else                  : orient = orient_deg

        sp.make_cspad_pix_coordinate_arrays (xc_um, yc_um, orient, tilt_deg)

#------------------------------

    def make_cspad_pix_coordinate_arrays (sp, xc_um, yc_um, orient_deg, tilt_deg=None) : # All lists of [4,8]
        """Makes [4,8,185,388] cspad pixel x and y coordinate arrays"""        
        sp.make_maps_of_2x1_pix_coordinates()

        sp.x_pix_um = np.zeros((sp.quads,sp.sects,sp.rows,sp.cols), dtype=np.float32)
        sp.y_pix_um = np.zeros((sp.quads,sp.sects,sp.rows,sp.cols), dtype=np.float32)

        angle_deg = orient_deg
        if tilt_deg != None : angle_deg += tilt_deg
 
        for quad in range(sp.quads) :
            for sect in range(sp.sects) :

                angle_rad = math.radians(angle_deg[quad][sect])                
                S,C = math.sin(angle_rad), math.cos(angle_rad)
                Xrot, Yrot = rotation(sp.x_map2x1_um, sp.y_map2x1_um, C, S)

                sp.x_pix_um[quad][sect][:] =  Xrot + xc_um[quad][sect]
                sp.y_pix_um[quad][sect][:] =  Yrot + yc_um[quad][sect]

        sp.x_pix_um -= sp.x_pix_um.min()
        sp.y_pix_um -= sp.y_pix_um.min()

#------------------------------

    def get_cspad_pix_coordinate_arrays_um (sp) : 
        return sp.x_pix_um, sp.y_pix_um


    def get_cspad_pix_coordinate_arrays_pix (sp) : 
        return sp.x_pix_um/sp.pixs, sp.y_pix_um/sp.pixs

#------------------------------

    def print_cspad_coordinate_arrays(sp) :
        print 'sp.x_pix_um:\n',      sp.x_pix_um
        print 'sp.x_pix_um.shape =', sp.x_pix_um.shape
        print 'sp.y_pix_um\n',       sp.y_pix_um
        print 'sp.y_pix_um.shape =', sp.y_pix_um.shape

#------------------------------
#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------
#------------------------------

import PyCSPadImage.HDF5Methods as hm # For test purpose in main only
#import GlobalMethods            as gm


def main_test_cspad() :

    fname_arr = 'cxi64813-r0058-cspad-arr.bin'

    #fname, runnum = '/reg/d/psdm/CXI/cxi64813/hdf5/cxi64813-r0058.h5',     58
    #dsname = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDs1.0:Cspad.0/data'
    #event   = 0
    #arr = hm.getAverageCSPadEvent( fname, dsname, event, nevents=10 )
    ##np.savetxt(fname_arr, arr, fmt='%f')
    #arr.tofile(fname_arr, sep="", format="%s")

    #arr = np.loadtxt(fname_arr, dtype=np.float32)
    arr = np.fromfile(fname_arr, dtype=np.float32)
    print 'arr.shape:', arr.shape # (32, 185, 388)

    xc_um = np.array(
            [[ 473.38,  685.26,  155.01,  154.08,  266.81,   53.95,  583.04,  582.15],  
             [ 989.30,  987.12, 1096.93,  884.11, 1413.16, 1414.94, 1500.83, 1288.02],  
             [1142.59,  930.23, 1459.44, 1460.67, 1347.57, 1559.93, 1032.27, 1033.44],  
             [ 626.78,  627.42,  516.03,  729.15,  198.28,  198.01,  115.31,  327.66]]) * PixCoords2x1.pixs # 109.92

    yc_um = np.array(
            [[1028.07, 1026.28, 1139.46,  926.91, 1456.78, 1457.35, 1539.71, 1327.89],  
             [1180.51,  967.36, 1497.74, 1498.54, 1385.08, 1598.19, 1069.65, 1069.93],  
             [ 664.89,  666.83,  553.60,  765.91,  237.53,  236.06,  152.17,  365.47],  
             [ 510.38,  722.95,  193.33,  193.41,  308.04,   95.25,  625.28,  624.14]]) * PixCoords2x1.pixs # 109.92

    tilt_deg = np.array(
                    [[0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],  
                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],  
                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],  
                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    print 'xc_um:\n',      xc_um
    print 'yc_um:\n',      yc_um
    #print 'orient_deg:\n', orient_deg
    print 'tilt_deg:\n',   tilt_deg


    t0_sec = time()
    w = CSPadPixCoordsWODB(xc_um, yc_um, tilt_deg=tilt_deg, use_wide_pix_center=False)
    print 'Consumed time for coordinate arrays (sec) =', time()-t0_sec

    #w.print_cspad_coordinate_arrays()
    X,Y = w.get_cspad_pix_coordinate_arrays_pix ()

    #print 'X(pix) :\n', X
    print 'X.shape =\n', X.shape

    xsize = X.max() + 1
    ysize = Y.max() + 1
    H,Xedges,Yedges = np.histogram2d(X.flatten(), Y.flatten(), bins=[xsize,ysize], range=[[0,xsize],[0,ysize]], normed=False, weights=arr.flatten()) 

    range = (-5, 5)
    gg.plotImageLarge(H, range=(0, 10), figsize=(12,11))
    gg.show()

#------------------------------
 
if __name__ == "__main__" :

    main_test_cspad()

    sys.exit ( 'End of test.' )

#------------------------------
