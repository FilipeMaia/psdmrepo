#!/usr/bin/env python
#------------------------------
"""GeometryObject - building block for hierarchical geometry

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

Revision: $Revision$

@version $Id$

@author Mikhail S. Dubrovin
"""
#--------------------------------
__version__ = "$Revision$"
#--------------------------------

import os
import sys
import math
import numpy as np

from PyCSPadImage.PixCoords2x1 import cspad2x1_one

#------------------------------

def rotation(X, Y, C, S) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot
    """
    Xrot = X*C - Y*S 
    Yrot = Y*C + X*S 
    return Xrot, Yrot

#------------------------------

class GeometryObject :

    def __init__(self, pname=None, pindex=None, \
                 oname=None, oindex=None, \
                 x0=0, y0=0, z0=0, \
                 rot_z=0, rot_y=0, rot_x=0, \
                 tilt_z=0, tilt_y=0, tilt_x=0) : 

        self.pname  = pname
        self.pindex = pindex

        self.oname  = oname
        self.oindex = oindex

        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        self.rot_z  = rot_z  
        self.rot_y  = rot_y 
        self.rot_x  = rot_x 
                            
        self.tilt_z = tilt_z
        self.tilt_y = tilt_y
        self.tilt_x = tilt_x

        if   self.oname == 'SENS2X1:V1' : self.algo = cspad2x1_one # PixCoords2x1(use_wide_pix_center=False)
        elif self.oname == 'SENS2X1:V2' : self.algo = None
        elif self.oname == 'SENS2X1:V3' : self.algo = None
        elif self.oname == 'SENS2X1:V4' : self.algo = None
        elif self.oname == 'SENS2X1:V5' : self.algo = None
        else                            : self.algo = None

        # ---- 2-nd stage
        self.parent = None
        self.list_of_children = []
        
#------------------------------

    def print_geo(self) :
        print 'parent:%10s %2d   geo: %10s %2d' % (self.pname, self.pindex, self.oname, self.oindex) + \
              '  x0:%8.0f  y0:%8.0f  z0:%8.0f' % (self.x0, self.y0, self.z0) + \
              '  rot_z:%6.1f  rot_y:%6.1f  rot_x:%6.1f' % (self.rot_z, self.rot_y, self.rot_x) + \
              '  tilt_z:%8.5f  tilt_y:%8.5f  tilt_x:%8.5f' % (self.tilt_z, self.tilt_y, self.tilt_x)

#------------------------------

    def print_geo_children(self) :
        msg = 'parent:%10s %2d   geo: %10s %2d #children: %d:' % \
              (self.pname, self.pindex, self.oname, self.oindex, len(self.list_of_children))
        for geo in self.list_of_children :
            msg += ' %s:%d' % (geo.oname, geo.oindex)
        print msg

#------------------------------

    def set_parent(self, parent) :
        self.parent = parent

#------------------------------

    def add_child(self, child) :
        self.list_of_children.append(child)

#------------------------------

    def get_parent(self) :
        return self.parent

#------------------------------

    def get_list_of_children(self) :
        return self.list_of_children

#------------------------------

    def transform_geo_coord_arrays(self, X, Y, Z, do_tilt = True) :

        angle_z = self.rot_z + self.tilt_z if do_tilt else self.rot_z
        angle_y = self.rot_y + self.tilt_y if do_tilt else self.rot_y
        angle_x = self.rot_x + self.tilt_x if do_tilt else self.rot_x

        angle_rad_z = math.radians(angle_z)
        angle_rad_y = math.radians(angle_y)
        angle_rad_x = math.radians(angle_x)

        Sz, Cz = math.sin(angle_rad_z), math.cos(angle_rad_z)
        Sy, Cy = math.sin(angle_rad_y), math.cos(angle_rad_y)
        Sx, Cx = math.sin(angle_rad_x), math.cos(angle_rad_x)

        X1, Y1 = rotation(X,  Y,  Cz, Sz)
        Z2, X2 = rotation(Z,  X1, Cy, Sy)
        Y3, Z3 = rotation(Y1, Z2, Cx, Sx)

        Zt = Z3 + self.z0
        Yt = Y3 + self.y0
        Xt = X2 + self.x0

        return Xt, Yt, Zt 

#------------------------------

    def get_pixel_coords(self) :

        #if self.oname == 'SENS2X1:V1' : 
        if self.algo is not None :
            xac, yac, zac = self.algo.get_xyz_maps_um()
            return self.transform_geo_coord_arrays(xac, yac, zac)

        xch, ych, zch = None, None, None        
        for i, child in enumerate(self.list_of_children) :
            xch, ych, zch = child.get_pixel_coords()
            if child.oindex != i :
                print 'WARNING! Geometry object %s:%d has non-consequtive index in calibration file, reconst index:%d' % \
                      (child.oname, child.oindex, i)
            if i==0 :
                xac = xch
                yac = ych
                zac = zch
            else :
                xac = np.vstack((xac, xch))
                yac = np.vstack((yac, ych))
                zac = np.vstack((zac, zch))

        # define shape for output x,y,z arrays
        shape_child = xch.shape
        len_child = len(self.list_of_children)
        geo_shape = np.hstack(([len_child], xch.shape))
        #print 'geo_shape = ', geo_shape        
        xac.shape = geo_shape
        yac.shape = geo_shape
        zac.shape = geo_shape
        return self.transform_geo_coord_arrays(xac, yac, zac)

#------------------------------

    def transform_2d_geo_coord_arrays(self, X, Y) :

        do_tilt = True

        angle_z = self.rot_z + self.tilt_z if do_tilt else self.rot_z
        angle_rad_z = math.radians(angle_z)
        Sz, Cz = math.sin(angle_rad_z), math.cos(angle_rad_z)
        X1, Y1 = rotation(X,  Y,  Cz, Sz)
        Xt = X1 + self.x0
        Yt = Y1 + self.y0
        return Xt, Yt

#------------------------------

    def get_2d_pixel_coords(self) :

        #if self.oname == 'SENS2X1:V1' : 
        if self.algo is not None :
            xac, yac, zac = self.algo.get_xyz_maps_um()
            return self.transform_2d_geo_coord_arrays(xac, yac)

        xac, yac = [], []
        for child in self.list_of_children :
            xch, ych = child.get_2d_pixel_coords()
            xac += list(xch.flatten())
            yac += list(ych.flatten())
        return self.transform_2d_geo_coord_arrays(np.array(xac), np.array(yac))

#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    print 78*'='+'\n==  Tests for this module are available in pyimgalgos/src/GeometryAccess.py ==\n'+78*'='
    sys.exit ('End of %s' % sys.argv[0])

#------------------------------
#------------------------------
#------------------------------


