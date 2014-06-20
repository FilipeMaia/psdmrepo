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
#from time import time
#import GlobalGraphics as gu # for test purpose

#from PyCSPadImage.PixCoords2x1 import PixCoords2x1
from PyCSPadImage.PixCoords2x1 import cspad2x1_one

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

    def transform_geo_coord_arrays(self, X, Y, Z) :

        do_tilt = True

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
#------------------------------
#------------------------------

def rotation(X, Y, C, S) :
    """For numpy arrays X and Y returns the numpy arrays of Xrot and Yrot
    """
    Xrot = X*C - Y*S 
    Yrot = Y*C + X*S 
    return Xrot, Yrot

#------------------------------

def load_pars_from_file(path) :

    list_of_geos = []

    f=open(path,'r')
    for linef in f :
        line = linef.strip('\n')
        if not line :       continue # discard empty strings
        if line[0] == '#' : continue # discard comments
        #print line
        #geo=parse_line(line)
        list_of_geos.append(parse_line(line))

    f.close()

    set_relations(list_of_geos)

    return list_of_geos

#------------------------------

def parse_line(line) :
    keys = ['pname','pindex','oname','oindex','x0','y0','z0','rot_z','rot_y','rot_x','tilt_z','tilt_y','tilt_x']
    f = line.split()
    if len(f) != len(keys) :
        print 'The list length for fields from file: %d is not equal to expected: %d' % (len(f), len(keys))
        return

    vals = [str  (f[0]),
            int  (f[1]),
            str  (f[2]),
            int  (f[3]),
            float(f[4]),
            float(f[5]),
            float(f[6]),
            float(f[7]),
            float(f[8]),
            float(f[9]),
            float(f[10]),
            float(f[11]),
            float(f[12])
           ]

    #print 'keys: ', keys
    #print 'vals: ', vals

    d = dict(zip(keys, vals))
    #print 'd=', d
    #return d
    return GeometryObject(**d)

#------------------------------

def find_parent(geobj, list_of_geos) :

    for geo in list_of_geos :
        if geo == geobj : continue
        if  geo.oname  == geobj.pname \
        and geo.oindex == geobj.pindex :
            return geo

    # The name of parent object is not found among geo names in the list_of_geos
    # add top parent object to the list
    if geobj.pname is not None :
        top_parent = GeometryObject(pname=None, pindex=-1, oname=geobj.pname, oindex=geobj.pindex)
        list_of_geos.append(top_parent)
        return top_parent
               
    return None # for top parent itself

#------------------------------

def get_geo(oname, oindex, list_of_geos) :
    for geo in list_of_geos :
        if  geo.oname  == oname \
        and geo.oindex == oindex :
            return geo
    return None

#------------------------------

def get_top_geo(list_of_geos) :
    return list_of_geos[-1]

#------------------------------

def set_relations(list_of_geos) :
    for geo in list_of_geos :
        #geo.print_geo()
        parent = find_parent(geo, list_of_geos)        
        if parent is not None :
            geo.set_parent(parent)
            parent.add_child(geo)
            print 'geo:%s:%d has parent:%s:%d' % (geo.oname, geo.oindex, parent.oname, parent.oindex)




#------------------------------
#------------------------------
#----------- TESTS ------------
#------------------------------
#------------------------------

from time import time # for test purpose only
import pyimgalgos.GlobalGraphics as gg # for test purpose
import pyimgalgos.TestImageGenerator as tig # for test purpose only

#------------------------------

def test_load(fname) :
    if not os.path.exists(fname) :
        sys.exit ('file %s does not exist' % fname)
    list_of_geos = load_pars_from_file(fname)
    return list_of_geos

#------------------------------

def test_access(list_of_geos) :

    for geo in list_of_geos : geo.print_geo()
    for geo in list_of_geos : geo.print_geo_children()

    print '\nTOP GEO:'
    top_geo = get_top_geo(list_of_geos)
    top_geo.print_geo_children()

    print '\nINTERMEDIATE GEO (QUAD):'
    geo = get_geo('QUAD:V1', 0, list_of_geos) 
    geo.print_geo_children()

    t0_sec = time()
    X,Y,Z = geo.get_pixel_coords()
    #X,Y = geo.get_2d_pixel_coords()
    print 'X:\n', X
    print 'Consumed time to get 3d pixel coordinates = %7.3f sec' % (time()-t0_sec)
    print 'Geometry object: %s:%d X.shape:%s' % (geo.oname, geo.oindex, str(X.shape))

#------------------------------

def test_plot_quad(list_of_geos) :

    # get pixel coordinate arrays:
    geo = get_geo('QUAD:V1', 1, list_of_geos) 
    X,Y,Z = geo.get_pixel_coords()
    print 'X,Y,Z shape:', X.shape 

    # get index arrays
    pix_size = cspad2x1_one.get_pix_size_um()
    xmin, ymin = X.min()-pix_size/2, Y.min()-pix_size/2 
    iX, iY = np.array((X-xmin)/pix_size, dtype=np.uint), np.array((Y-ymin)/pix_size, dtype=np.uint)

    # get intensity array
    arr = tig.cspad_nparr(n2x1=X.shape[0])
    arr.shape = (8,185,388)
    amp_range = (0,185+388)
 
    print 'iX, iY, W shape:', iX.shape, iY.shape, arr.shape 
    img = gg.getImageFromIndexArrays(iX,iY,W=arr)

    gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()

#------------------------------

def test_plot_cspad(list_of_geos, fname_data) :

    # get pixel coordinate arrays:
    geo = get_top_geo(list_of_geos) 
    X,Y,Z = geo.get_pixel_coords()
    print 'X,Y,Z shape:', X.shape 

    # get index arrays
    pix_size = cspad2x1_one.get_pix_size_um()
    #xmin, ymin = X.min()-pix_size/2, Y.min()-pix_size/2 
    #iX, iY = np.array((X-xmin)/pix_size, dtype=np.uint), np.array((Y-ymin)/pix_size, dtype=np.uint)
    offset_um = 850*pix_size
    iX, iY = np.array((X+offset_um)/pix_size, dtype=np.uint), np.array((Y+offset_um)/pix_size, dtype=np.uint)

    arr, amp_range = np.loadtxt(fname_data, dtype=np.float), (0,0.5)
    arr.shape= (4,8,185,388)

    print 'iX, iY, W shape:', iX.shape, iY.shape, arr.shape 
    img = gg.getImageFromIndexArrays(iX,iY,W=arr)

    gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()

#------------------------------

if __name__ == "__main__" :

    #fname = '/reg/d/psdm/cxi/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2013-12-20/'
    fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/1-end.data'
    fname_data     = basedir + 'cspad-ndarr-ave-cxi83714-r0136.dat'
    list_of_geos = test_load(fname_geometry)

    msg = 'Use command: sys.argv[0] <num>, wher num=[1,3]'

    if len(sys.argv)==1   : print 'App needs in input parameter.' + msg
    elif sys.argv[1]=='1' : test_access(list_of_geos)
    elif sys.argv[1]=='2' : test_plot_quad(list_of_geos)
    elif sys.argv[1]=='3' : test_plot_cspad(list_of_geos, fname_data)
    else : print 'Wrong input parameter.' + msg

    sys.exit ('End of %s' % sys.argv[0])

#------------------------------


