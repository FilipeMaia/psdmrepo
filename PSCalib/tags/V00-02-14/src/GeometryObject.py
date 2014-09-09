#!/usr/bin/env python
#------------------------------
""":py:class:`PSCalib.GeometryObject` - building block for hierarchical geometry

Usage::

    from PSCalib.GeometryObject import GeometryObject
    ...
    d = <dictionary-of-input-parameters>
    geo = GeometryObject(**d)
    geo.print_geo()
    geo.print_geo_children()
    ...

    Methods of this class are used internally in PSCalib.GeometryAccess
    and are not supposed to be used directly...

@see :py:class:`PSCalib.GeometryAccess`


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

#from PyCSPadImage.PixCoords2x1 import cspad2x1_one
from PSCalib.SegGeometryStore import sgs

#------------------------------

def rotation_cs(X, Y, C, S) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot
    """
    Xrot = X*C - Y*S 
    Yrot = Y*C + X*S 
    return Xrot, Yrot

#------------------------------

def rotation(X, Y, angle_deg) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot rotated by angle_deg
    """
    angle_rad = math.radians(angle_deg)
    S, C = math.sin(angle_rad), math.cos(angle_rad)
    return rotation_cs(X, Y, C, S)

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

        #if   self.oname == 'SENS2X1:V1' : self.algo = cspad2x1_one # PixCoords2x1(use_wide_pix_center=False)
        #elif self.oname == 'SENS2X1:V2' : self.algo = None
        #elif self.oname == 'SENS2X1:V3' : self.algo = None
        #elif self.oname == 'SENS2X1:V4' : self.algo = None
        #elif self.oname == 'SENS2X1:V5' : self.algo = None
        #else                            : self.algo = None

        self.algo = sgs.Create(self.oname, pbits=0) # ex.: SegGeometryCspad2x1V1(...)

        # ---- 2-nd stage
        self.parent = None
        self.list_of_children = []
        
#------------------------------

    def print_geo(self) :
        """ Print info about self geometry object
        """
        print 'parent:%10s %2d   geo: %10s %2d' % (self.pname, self.pindex, self.oname, self.oindex) + \
              '  x0:%8.0f  y0:%8.0f  z0:%8.0f' % (self.x0, self.y0, self.z0) + \
              '  rot_z:%6.1f  rot_y:%6.1f  rot_x:%6.1f' % (self.rot_z, self.rot_y, self.rot_x) + \
              '  tilt_z:%8.5f  tilt_y:%8.5f  tilt_x:%8.5f' % (self.tilt_z, self.tilt_y, self.tilt_x)

#------------------------------

    def print_geo_children(self) :
        """ Print info about children of self geometry object
        """
        msg = 'parent:%10s %2d   geo: %10s %2d #children: %d:' % \
              (self.pname, self.pindex, self.oname, self.oindex, len(self.list_of_children))
        for geo in self.list_of_children :
            msg += ' %s:%d' % (geo.oname, geo.oindex)
        print msg

#------------------------------

    def set_parent(self, parent) :
        """ Set parent geometry object for self
        """
        self.parent = parent

#------------------------------

    def add_child(self, child) :
        """ Add children geometry object to the list
        """
        self.list_of_children.append(child)

#------------------------------

    def get_parent(self) :
        """ Returns parent geometry object
        """
        return self.parent

#------------------------------

    def get_list_of_children(self) :
        """ Returns list of children geometry objects
        """
        return self.list_of_children

#------------------------------

    def get_geo_name(self) :
        """ Returns self geometry object name
        """
        return self.oname

#------------------------------

    def get_geo_index(self) :
        """ Returns self geometry object index
        """
        return self.oindex

#------------------------------

    def get_parent_name(self) :
        """ Returns parent geometry object name
        """
        return self.pname

#------------------------------

    def get_parent_index(self) :
        """ Returns parent geometry object index
        """
        return self.pindex

#------------------------------

    def transform_geo_coord_arrays(self, X, Y, Z, do_tilt = True) :
        """ Transform geometry object coordinates to the parent frame
        """
        angle_z = self.rot_z + self.tilt_z if do_tilt else self.rot_z
        angle_y = self.rot_y + self.tilt_y if do_tilt else self.rot_y
        angle_x = self.rot_x + self.tilt_x if do_tilt else self.rot_x

        X1, Y1 = rotation(X,  Y,  angle_z)
        Z2, X2 = rotation(Z,  X1, angle_y)
        Y3, Z3 = rotation(Y1, Z2, angle_x)

        Zt = Z3 + self.z0
        Yt = Y3 + self.y0
        Xt = X2 + self.x0

        return Xt, Yt, Zt 

#------------------------------

    def get_pixel_coords(self) :
        """ Returns three numpy arrays with pixel X, Y, Z coordinates for self geometry object
        """
        if self.algo is not None :
            xac, yac, zac = self.algo.pixel_coord_array()
            return self.transform_geo_coord_arrays(xac, yac, zac)

        xac, yac, zac = None, None, None
        for ind, child in enumerate(self.list_of_children) :
            if child.oindex != ind :
                print 'WARNING! Geometry object %s:%d has non-consequtive index in calibration file, reconst index:%d' % \
                      (child.oname, child.oindex, ind)

            xch, ych, zch = child.get_pixel_coords()

            if ind==0 :
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

    def get_pixel_areas(self) :
        """ Returns numpy array with pixel areas for self geometry object
        """
        if self.algo is not None :
            return self.algo.pixel_area_array()

        aar = None
        for ind, child in enumerate(self.list_of_children) :
            if child.oindex != ind :
                print 'WARNING! Geometry object %s:%d has non-consequtive index in calibration file, reconst index:%d' % \
                      (child.oname, child.oindex, ind)

            ach = child.get_pixel_areas()
            aar = ach if ind==0 else np.vstack((aar, ach))

        # define shape for output x,y,z arrays
        shape_child = ach.shape
        len_child = len(self.list_of_children)
        geo_shape = np.hstack(([len_child], ach.shape))
        #print 'geo_shape = ', geo_shape        
        aar.shape = geo_shape
        return aar

#------------------------------

    def get_size_geo_array(self) :
        """ Returns size of  self geometry object
        """
        if self.algo is not None : return self.algo.size()

        size_arr = 0
        for child in self.list_of_children :
            size_arr += child.get_size_geo_array()

        return size_arr

#------------------------------

    def get_pixel_scale_size(self) :
        """ Returns pixel scale size of the geometry object from the first found segment
        """
        if self.algo is not None : return self.algo.pixel_scale_size()

        for child in self.list_of_children :
            return child.get_pixel_scale_size()

#------------------------------
#------------------------------
# Additional to interface 2-d methods
#------------------------------
#------------------------------

    def transform_2d_geo_coord_arrays(self, X, Y, do_tilt = True) :
        """ Simplified version of transform_geo_coord_arrays(...) for 2-d case
        """
        angle_z = self.rot_z + self.tilt_z if do_tilt else self.rot_z
        X1, Y1 = rotation(X,  Y,  angle_z)
        Xt = X1 + self.x0
        Yt = Y1 + self.y0
        return Xt, Yt

#------------------------------

    def get_2d_pixel_coords(self) :
        """ Simplified version of get_pixel_coords() for 2-d case 
        """
        #if self.oname == 'SENS2X1:V1' : 
        if self.algo is not None :
            #xac, yac, zac = self.algo.get_xyz_maps_um()
            xac, yac, zac = self.algo.pixel_coord_array()
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


