#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module SegGeometryCspad2x1V1...
#
#------------------------------------------------------------------------

"""
This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see SegGeometry

@version $Id: 2013-03-08$

@author Mikhail S. Dubrovin

Use matrix notations (like in data array)
DIFFERENT from the detector map... rows<->cols:
Assume that 2x1 has 195 rows and 388 columns
The (r,c)=(0,0) is in the top left corner of the matrix, has coordinates (xmin,ymax)

                    ^ Y          (Xmax,Ymax)
   (0,0)            |            (0,387)
      ------------------------------
      |             |              |
      |             |              |
      |             |              |
    --|-------------+--------------|----> X
      |             |              |
      |             |              |
      |             |              |
      ------------------------------
   (184,0)          |           (184,387)
   (Xmin,Ymin)

"""

#--------------------------------
#  Module's version from CVS --
#--------------------------------
__version__ = "$Revision$"
# $Source$
#--------------------------------

import sys
import math
import numpy as np
from time import time

from PSCalib.SegGeometry import *

#------------------------------

def rotation(X, Y, C, S) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot
    """
    Xrot = X*C - Y*S 
    Yrot = Y*C + X*S 
    return Xrot, Yrot

#------------------------------

class SegGeometryCspad2x1V1(SegGeometry) :
    """Self-sufficient class for generation of CSPad 2x1 sensor pixel coordinate array"""

    _rows  = 185    # Number of rows in 2x1 at rotation 0
    _cols  = 388    # Number of cols in 2x1 at rotation 0
    pixs   = 109.92 # Pixel size in um (micrometer)
    pixw   = 274.80 # Wide pixel size in um (micrometer)
    pixd   = 400.00 # Pixel depth in um (micrometer)

    colsh = _cols/2
    pixsh = pixs/2
    pixwh = pixw/2

#------------------------------

    def __init__(sp, use_wide_pix_center=True) :
        #print 'SegGeometryCspad2x1V1.__init__()'

        SegGeometry.__init__(sp)
        #super(SegGeometry, self).__init__()

        sp.use_wide_pix_center = use_wide_pix_center

        sp.x_pix_arr_um_offset  = None
        sp.pix_area_arr = None

        sp.make_pixel_coord_arrs()

#------------------------------

    def make_pixel_coord_arrs(sp) :
        """Makes [185,388] maps of x, y, and z 2x1 pixel coordinates
        with origin in the center of 2x1
        """        
        x_rhs = np.arange(sp.colsh)*sp.pixs + sp.pixw - sp.pixsh
        if sp.use_wide_pix_center : x_rhs[0] = sp.pixwh # set x-coordinate of the wide pixel in its geometry center
        sp.x_arr_um = np.hstack([-x_rhs[::-1],x_rhs])

        sp.y_arr_um = -np.arange(sp._rows) * sp.pixs
        sp.y_arr_um -= sp.y_arr_um[-1]/2 # move origin to the center of array

        #sp.x_arr_pix = sp.x_arr_um/sp.pixs
        #sp.y_arr_pix = sp.y_arr_um/sp.pixs

        #sp.x_pix_arr_pix, sp.y_pix_arr_pix = np.meshgrid(sp.x_arr_pix, sp.y_arr_pix)
        sp.x_pix_arr_um, sp.y_pix_arr_um  = np.meshgrid(sp.x_arr_um, sp.y_arr_um)
        sp.z_pix_arr_um = np.zeros((sp._rows,sp._cols))
        
#------------------------------

    def make_pixel_size_arrs(sp) :
        """Makes [185,388] maps of x, y, and z 2x1 pixel size 
        """        
        if sp.pix_area_arr is not None : return

        x_rhs_size_um = np.ones(sp.colsh)*sp.pixs
        x_rhs_size_um[0] = sp.pixw
        x_arr_size_um = np.hstack([x_rhs_size_um[::-1],x_rhs_size_um])
        y_arr_size_um = np.ones(sp._rows) * sp.pixs

        sp.x_pix_size_um, sp.y_pix_size_um = np.meshgrid(x_arr_size_um, y_arr_size_um)
        sp.z_pix_size_um = np.ones((sp._rows,sp._cols)) * sp.pixd
        
        factor = 1./(sp.pixs*sp.pixs)
        sp.pix_area_arr = sp.x_pix_size_um * sp.y_pix_size_um * factor

#------------------------------

    def print_member_data(sp) :
        print 'SegGeometryCspad2x1V1.print_member_data()'
        print '    _rows : %d'    % sp._rows
        print '    _cols : %d'    % sp._cols
        print '    pixs  : %7.2f' % sp.pixs 
        print '    pixw  : %7.2f' % sp.pixw 
        print '    pixd  : %7.2f' % sp.pixd 
        print '    colsh : %d'    % sp.colsh
        print '    pixsh : %7.2f' % sp.pixsh
        print '    pixwh : %7.2f' % sp.pixwh

#------------------------------

    def print_pixel_size_arrs(sp) :
        print 'SegGeometryCspad2x1V1.print_pixel_size_arrs()'
        sp.make_pixel_size_arrs()
        print 'sp.x_pix_size_um[0:10,190:198]:\n', sp.x_pix_size_um[0:10,190:198]
        print 'sp.x_pix_size_um.shape = ',         sp.x_pix_size_um.shape
        print 'sp.y_pix_size_um:\n',               sp.y_pix_size_um
        print 'sp.y_pix_size_um.shape = ',         sp.y_pix_size_um.shape
        print 'sp.z_pix_size_um:\n',               sp.z_pix_size_um
        print 'sp.z_pix_size_um.shape = ',         sp.z_pix_size_um.shape
        print 'sp.pix_area_arr[0:10,190:198]:\n',  sp.pix_area_arr[0:10,190:198]
        print 'sp.pix_area_arr.shape  = ',         sp.pix_area_arr.shape

#------------------------------

    def print_maps_2x1_um(sp) :
        print 'SegGeometryCspad2x1V1.print_maps_2x1_um()'
        print 'x_pix_arr_um = ',       sp.x_pix_arr_um
        print 'x_pix_arr_um.shape = ', sp.x_pix_arr_um.shape
        print 'y_pix_arr_um = ',       sp.y_pix_arr_um
        print 'y_pix_arr_um.shape = ', sp.y_pix_arr_um.shape
        print 'z_pix_arr_um = ',       sp.z_pix_arr_um
        print 'z_pix_arr_um.shape = ', sp.z_pix_arr_um.shape

#------------------------------

    def print_xy_1darr_um(sp) :
        print 'SegGeometryCspad2x1V1.print_xy_1darr_um()'
        print 'x_arr_um:\n',       sp.x_arr_um
        print 'x_arr_um.shape = ', sp.x_arr_um.shape
        print 'y_arr_um:\n',       sp.y_arr_um
        print 'y_arr_um.shape = ', sp.y_arr_um.shape

#------------------------------

    def print_xyz_min_max_um(sp) :
        print 'SegGeometryCspad2x1V1.print_xyz_min_max_um()'
        xmin, ymin, zmin = sp.get_xyz_min_um()
        xmax, ymax, zmax = sp.get_xyz_max_um()
        print 'In [um] xmin:%9.2f, xmax:%9.2f, ymin:%9.2f, ymax:%9.2f, zmin:%9.2f, zmax:%9.2f' \
              % (xmin, xmax, ymin, ymax, zmin, zmax)

#------------------------------

    def get_xyz_min_um(sp) : 
        return sp.x_arr_um[0], sp.y_arr_um[-1], 0

    def get_xyz_max_um(sp) : 
        return sp.x_arr_um[-1], sp.y_arr_um[0], 0

    def get_cspad2x1_xy_maps_um(sp) : 
        return sp.x_pix_arr_um, sp.y_pix_arr_um

    def get_cspad2x1_xyz_maps_um(sp) : 
        return sp.x_pix_arr_um, sp.y_pix_arr_um, sp.z_pix_arr_um

    def get_cspad2x1_xy_maps_um_with_offset(sp) : 
        if  sp.x_pix_arr_um_offset == None :
            x_min_um, y_min_um, z_min_um = sp.get_xyz_min_um()
            sp.x_pix_arr_um_offset = sp.x_pix_arr_um - x_min_um
            sp.y_pix_arr_um_offset = sp.y_pix_arr_um - y_min_um
        return sp.x_pix_arr_um_offset, sp.y_pix_arr_um_offset

    def get_cspad2x1_xyz_maps_um_with_offset(sp) : 
        if  sp.x_pix_arr_um_offset == None :
            x_min_um, y_min_um, z_min_um = sp.get_xyz_min_um()
            sp.x_pix_arr_um_offset = sp.x_pix_arr_um - x_min_um
            sp.y_pix_arr_um_offset = sp.y_pix_arr_um - y_min_um
            sp.z_pix_arr_um_offset = sp.z_pix_arr_um - z_min_um
        return sp.x_pix_arr_um_offset, sp.y_pix_arr_um_offset, sp.z_pix_arr_um_offset

    def get_pix_size_um(sp) : 
        return sp.pixs

    def get_pixel_size_arrs_um(sp) :
        sp.make_pixel_size_arrs()
        return sp.x_pix_size_um, sp.y_pix_size_um, sp.z_pix_size_um

    def get_pixel_area_arr(sp) :
        sp.make_pixel_size_arrs()
        return sp.pix_area_arr

    def get_cspad2x1_xy_maps_pix(sp) :
        sp.x_pix_arr_pix = sp.x_pix_arr_um/sp.pixs
        sp.y_pix_arr_pix = sp.y_pix_arr_um/sp.pixs
        return sp.x_pix_arr_pix, sp.y_pix_arr_pix

    def get_cspad2x1_xy_maps_pix_with_offset(sp) :
        X, Y = sp.get_cspad2x1_xy_maps_pix()
        xmin, ymin = X.min(), Y.min()
        return X-xmin, Y-ymin

    def return_switch(sp, meth, axis=None) :
        """ Returns three x,y,z arrays if axis=None, or single array for specified axis 
        """
        if axis==None : return meth()
        else          : return dict( zip( sp.AXIS, meth() ))[axis]

#------------------------------
# INTERFACE METHODS
#------------------------------

    def print_seg_info(sp, pbits=0) :
        """ Prints segment info for selected bits
        """
        if pbits & 1 : sp.print_member_data()
        if pbits & 2 : sp.print_maps_2x1_um()
        if pbits & 4 : sp.print_xyz_min_max_um()
        if pbits & 8 : sp.print_xy_1darr_um()


    def size(sp) :
        """ Returns number of pixels in segment
        """
        return sp._rows*sp._cols


    def rows(sp) :
        """ Returns number of rows in segment
        """
        return sp._rows


    def cols(sp) :
        """ Returns number of cols in segment
        """
        return sp._cols


    def shape(sp) :
        """ Returns shape of the segment (rows, cols)
        """
        return (sp._rows, sp._cols)


    def pixel_scale_size(sp) :
        """ Returns pixel size in um for indexing
        """
        return sp.pixs


    def pixel_area_array(sp) :
        """ Returns pixel area array of shape=(rows, cols)
        """
        return sp.get_pixel_area_arr()


    def pixel_size_array(sp, axis=None) :
        """ Returns numpy array of pixel sizes in um for AXIS
        """
        return sp.return_switch(sp.get_pixel_size_arrs_um, axis)


    def pixel_coord_array(sp, axis=None) :
        """ Returns numpy array of segment pixel coordinates in um for AXIS
        """
        return sp.return_switch(sp.get_cspad2x1_xyz_maps_um, axis)


    def pixel_coord_min(sp, axis=None) :
        """ Returns minimal value in the array of segment pixel coordinates in um for AXIS
        """
        return sp.return_switch(sp.get_xyz_min_um, axis)


    def pixel_coord_max(sp, axis=None) :
        """ Returns maximal value in the array of segment pixel coordinates in um for AXIS
        """
        return sp.return_switch(sp.get_xyz_max_um, axis)
        
  
#------------------------------
#------------------------------

cspad2x1_one = SegGeometryCspad2x1V1(use_wide_pix_center=False)

#------------------------------
#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    import pyimgalgos.GlobalGraphics as gg # For test purpose in main only


def test_xyz_min_max() :
    w = SegGeometryCspad2x1V1()
    w.print_xyz_min_max_um() 
    print 'Ymin = ', w.pixel_coord_min('Y')
    print 'Ymax = ', w.pixel_coord_max('Y')

#------------------------------

def test_xyz_maps() :

    w = SegGeometryCspad2x1V1()
    w.print_maps_2x1_um()

    titles = ['X map','Y map']
    #for i,arr2d in enumerate([w.x_pix_arr,w.y_pix_arr]) :
    for i,arr2d in enumerate( w.get_cspad2x1_xy_maps_pix() ) :
        amp_range = (arr2d.min(), arr2d.max())
        gg.plotImageLarge(arr2d, amp_range=amp_range, figsize=(10,5), title=titles[i])
        gg.move(200*i,100*i)

    gg.show()

#------------------------------

def test_2x1_img() :

    t0_sec = time()
    w = SegGeometryCspad2x1V1(use_wide_pix_center=False)
    #w = SegGeometryCspad2x1V1(use_wide_pix_center=True)
    print 'Consumed time for coordinate arrays (sec) =', time()-t0_sec

    X,Y = w.get_cspad2x1_xy_maps_pix()

    w.print_seg_info(0377)

    #print 'X(pix) :\n', X
    print 'X.shape =', X.shape

    xmin, ymin, zmin = w.get_xyz_min_um()
    xmax, ymax, zmax = w.get_xyz_max_um()
    xmin /= w.pixel_scale_size()
    xmax /= w.pixel_scale_size()
    ymin /= w.pixel_scale_size()
    ymax /= w.pixel_scale_size()

    xsize = xmax - xmin + 1
    ysize = ymax - ymin + 1
    print 'xsize =', xsize # 391.0 
    print 'ysize =', ysize # 185.0

    H, Xedges, Yedges = np.histogram2d(X.flatten(), Y.flatten(), bins=[xsize,ysize], range=[[xmin, xmax], [ymin, ymax]], normed=False, weights=X.flatten()+Y.flatten()) 

    print 'Xedges:', Xedges
    print 'Yedges:', Yedges
    print 'H.shape:', H.shape

    gg.plotImageLarge(H, amp_range=(-250, 250), figsize=(8,10)) # range=(-1, 2), 
    gg.show()

#------------------------------

def test_2x1_img_easy() :
    pc2x1 = SegGeometryCspad2x1V1(use_wide_pix_center=False)
    #X,Y = pc2x1.get_cspad2x1_xy_maps_pix()
    X,Y = pc2x1.get_cspad2x1_xy_maps_pix_with_offset()
    iX, iY = (X+0.25).astype(int), (Y+0.25).astype(int)
    img = gg.getImageFromIndexArrays(iX,iY,iX+iY)
    gg.plotImageLarge(img, amp_range=(0, 500), figsize=(8,10))
    gg.show()

#------------------------------

def test_pix_sizes() :
    w = SegGeometryCspad2x1V1()
    w.print_pixel_size_arrs()
    size_arr = w.pixel_size_array('X')
    area_arr = w.pixel_area_array()
    print 'area_arr[0:10,190:198]:\n',  area_arr[0:10,190:198]
    print 'area_arr.shape :',           area_arr.shape
    print 'size_arr[0:10,190:198]:\n',  size_arr[0:10,190:198]
    print 'size_arr.shape :',           size_arr.shape

#------------------------------
 
if __name__ == "__main__" :

    if len(sys.argv)==1   : print 'For other test(s) use command: python', sys.argv[0], '<test-number=1-3>'
    elif sys.argv[1]=='0' : test_xyz_min_max()
    elif sys.argv[1]=='1' : test_xyz_maps()
    elif sys.argv[1]=='2' : test_2x1_img()
    elif sys.argv[1]=='3' : test_2x1_img_easy()
    elif sys.argv[1]=='4' : test_pix_sizes()
    else : print 'Non-expected arguments: sys.argv=', sys.argv

    sys.exit( 'End of test.' )

#------------------------------
