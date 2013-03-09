#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CSPad2x2PixCoordsWODB...
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

class CSPad2x2PixCoordsWODB (PixCoords2x1) :
    """Self-sufficient class for generation of CSPad2x2 pixel coordinate array without data base (WODB)"""

    sects = 2 # Total number of sections in quad

##------------------------------

    def __init__ (sp, xc_um=None, yc_um=None, tilt_deg=None, use_wide_pix_center=True) :
        print 'CSPad2x2PixCoordsWODB.__init__(...)'

        PixCoords2x1.__init__ (sp, use_wide_pix_center)

        if xc_um == None : return
        if yc_um == None : return

        sp.make_cspad2x2_pix_coordinate_arrays (xc_um, yc_um, tilt_deg)

#------------------------------

    def make_cspad2x2_pix_coordinate_arrays (sp, xc_um, yc_um, tilt_deg=None) : # All lists of size[2]
        """Makes [2,185,388] cspad pixel x and y coordinate arrays"""        
        #sp.make_maps_of_2x1_pix_coordinates()

        sp.x_pix_um = np.zeros((sp.sects,sp.rows,sp.cols), dtype=np.float32)
        sp.y_pix_um = np.zeros((sp.sects,sp.rows,sp.cols), dtype=np.float32)

        angle_deg = 0
        if tilt_deg != None : angle_deg += tilt_deg
 
        for sect in range(sp.sects) :

            angle_rad = math.radians(angle_deg[sect])                
            S,C = math.sin(angle_rad), math.cos(angle_rad)
            Xrot, Yrot = rotation(sp.x_map2x1_um, sp.y_map2x1_um, C, S)

            sp.x_pix_um[sect][:] =  Xrot + xc_um[sect]
            sp.y_pix_um[sect][:] =  Yrot + yc_um[sect]

        sp.x_pix_um -= sp.x_pix_um.min() + 5 # add offset in um to get rid of "rounding" strips...
        sp.y_pix_um -= sp.y_pix_um.min() + 5

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
#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------
#------------------------------

def main_test_cspad2x2() :

    xc_um    = np.array([198., 198.]) * 109.92 # CSPad2x2PixCoordsWODB.pixs # 109.92
    yc_um    = np.array([ 95., 308.]) * 109.92
    tilt_deg = np.array([  0.,   0.])

    print 'xc_um   : ', xc_um
    print 'yc_um   : ', yc_um
    print 'tilt_deg: ', tilt_deg

    t0_sec = time()
    w = CSPad2x2PixCoordsWODB(xc_um, yc_um, tilt_deg)
    print 'Consumed time for coordinate arrays (sec) =', time()-t0_sec

    #w.print_cspad2x2_coordinate_arrays()
    X,Y = w.get_cspad2x2_pix_coordinate_arrays_pix ()

    #print 'X(pix) :\n', X
    print 'X.shape =\n', X.shape

    xsize = X.max() + 1
    ysize = Y.max() + 1
    H,Xedges,Yedges = np.histogram2d(X.flatten(), Y.flatten(), bins=[xsize,ysize], range=[[0,xsize],[0,ysize]], normed=False, weights=None) 

    range = (-5, 5)
    gg.plotImageLarge(H, range=(-1, 2), figsize=(12,11))
    gg.show()

#------------------------------
 
if __name__ == "__main__" :
    main_test_cspad2x2()
    sys.exit ( 'End of test.' )

#------------------------------
