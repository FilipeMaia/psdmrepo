#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PixCoords2x1...
#
#------------------------------------------------------------------------

"""
This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 2013-03-08$

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
import GlobalGraphics as gg # For test purpose in main only
#------------------------------

def rotation(X, Y, C, S) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot
    """
    Xrot = X*C + Y*S 
    Yrot = Y*C - X*S 
    return Xrot, Yrot

#------------------------------

class PixCoords2x1() :
    """Self-sufficient class for generation of CSPad 2x1 sensor pixel coordinate array"""

    rows  = 185    # Number of rows in 2x1 at rotation 0
    cols  = 388    # Number of cols in 2x1 at rotation 0
    pixs  = 109.92 # Pixel size in um (micrometer)
    pixw  = 274.80 # Wide pixel size in um (micrometer)

    colsh = cols/2
    pixsh = pixs/2
    pixwh = pixw/2

#------------------------------

    def __init__(sp, use_wide_pix_center=True) :
        #print 'PixCoords2x1.__init__()'

        sp.use_wide_pix_center = use_wide_pix_center

        sp.make_maps_of_2x1_pix_coordinates()

#------------------------------

    def make_maps_of_2x1_pix_coordinates(sp) :
        """Makes [185,388] maps of x and y 2x1 pixel coordinates
        with origin in the center of 2x1
        """        
        x_rhs = np.arange(sp.colsh)*sp.pixs + sp.pixw - sp.pixsh
        if sp.use_wide_pix_center : x_rhs[0] = sp.pixwh # set x-coordinate of the wide pixel in its geometry center
        x_arr = np.hstack([-x_rhs[::-1],x_rhs])

        y_arr = np.arange(sp.rows) * sp.pixs
        y_arr -= y_arr[-1]/2 # move origin to the center of array

        sp.x_map2x1_um, sp.y_map2x1_um = np.meshgrid(x_arr, y_arr)
        
#------------------------------

    def print_maps_2x1(sp) :
        print 'x_map2x1_um = ',       sp.x_map2x1_um
        print 'x_map2x1_um.shape = ', sp.x_map2x1_um.shape
        print 'y_map2x1_um = ',       sp.y_map2x1_um
        print 'y_map2x1_um.shape = ', sp.y_map2x1_um.shape

#------------------------------

    def get_cspad2x1_pix_maps_pix(sp) : 
        return sp.x_map2x1_um/sp.pixs, sp.y_map2x1_um/sp.pixs

    def get_cspad2x1_pix_maps_um(sp) : 
        return sp.x_map2x1_um, sp.y_map2x1_um

    def get_cspad2x1_pix_maps_um_with_offset(sp) : 
        sp.xmin_um = sp.x_map2x1_um.min()
        sp.ymin_um = sp.y_map2x1_um.min()
        return sp.x_map2x1_um - sp.xmin_um, sp.y_map2x1_um - sp.xmin_um

#------------------------------
#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------
#------------------------------

def test_2x1_xy_maps() :

    w = PixCoords2x1()
    w.make_maps_of_2x1_pix_coordinates()
    w.print_maps_2x1()

    #for i,arr2d in enumerate([w.x_map2x1,w.y_map2x1]) :
    for i,arr2d in enumerate( w.get_cspad2x1_pix_maps_pix() ) :
        range = (arr2d.min(), arr2d.max())
        gg.plotImage(arr2d, range, figsize=(10,5))
        gg.move(200*i,100*i)

    gg.show()

#------------------------------

def test_2x1_img() :

    t0_sec = time()
    w = PixCoords2x1(use_wide_pix_center=False)
    print 'Consumed time for coordinate arrays (sec) =', time()-t0_sec

    X,Y = w.get_cspad2x1_pix_maps_pix()

    #print 'X(pix) :\n', X
    print 'X.shape =\n', X.shape

    xmin, xmax, ymin, ymax = X.min(), X.max(), Y.min(), Y.max()
    xsize = xmax - xmin + 1
    ysize = ymax - ymin + 1
    print 'xsize =', xsize # 391.0 
    print 'ysize =', ysize # 185.0

    H,Xedges,Yedges = np.histogram2d(X.flatten(), Y.flatten(), bins=[xsize,ysize], range=[[xmin,xmax],[ymin,ymax]], normed=False, weights=None) 

    gg.plotImageLarge(H, range=(-1, 2), figsize=(8,10))
    gg.show()

#------------------------------
 
if __name__ == "__main__" :

    #test_2x1_xy_maps()
    test_2x1_img()

    sys.exit( 'End of test.' )

#------------------------------
