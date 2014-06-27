#!/usr/bin/env python
#------------------------------
"""GeometryAccess - holds and access hierarchical geometry for generic pixel detector

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
from pyimgalgos.GeometryObject import GeometryObject

#------------------------------

class GeometryAccess :

    def __init__(self, path, pbits=0) : 

        if not os.path.exists(path) :
            sys.exit ('file %s does not exist' % path)

        self.path = path
        self.pbits = pbits
        self.list_of_geos = []
        self.load_pars_from_file()

        if self.pbits & 1 : self.print_list_of_geos()
        if self.pbits & 2 : self.print_list_of_geos_children()

    #------------------------------

    def load_pars_from_file(self) :
        """Reads input "geometry" file, discards empty lines and comments, fills the list of geometry objects for data lines
        """        
        f=open(self.path,'r')
        for linef in f :
            line = linef.strip('\n')
            if not line :       continue # discard empty strings
            if line[0] == '#' : continue # discard comments
            #print line
            #geo=self.parse_line(line)
            self.list_of_geos.append(self.parse_line(line))
    
        f.close()
    
        self.set_relations()
    
    #------------------------------
    
    def parse_line(self, line) :
        """Gets the string line with data from input file,
           creates and returns the geometry object for this string.
        """
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
    
    def find_parent(self, geobj) :
        """Finds and returns parent for geobj geometry object
        """           
        for geo in self.list_of_geos :
            if geo == geobj : continue
            if  geo.oname  == geobj.pname \
            and geo.oindex == geobj.pindex :
                return geo
    
        # The name of parent object is not found among geo names in the self.list_of_geos
        # add top parent object to the list
        if geobj.pname is not None :
            top_parent = GeometryObject(pname=None, pindex=-1, oname=geobj.pname, oindex=geobj.pindex)
            self.list_of_geos.append(top_parent)
            return top_parent
                   
        return None # for top parent itself
       
    #------------------------------

    def set_relations(self) :
        """Set relations between geometry objects in the list_of_geos
        """
        for geo in self.list_of_geos :
            #geo.print_geo()
            parent = self.find_parent(geo)        
            if parent is not None :
                geo.set_parent(parent)
                parent.add_child(geo)
                print 'geo:%s:%d has parent:%s:%d' % (geo.oname, geo.oindex, parent.oname, parent.oindex)

    #------------------------------

    def get_geo(self, oname, oindex) :
        """Returns specified geometry object
        """
        for geo in self.list_of_geos :
            if  geo.oname  == oname \
            and geo.oindex == oindex :
                return geo
        return None
    
    #------------------------------
    
    def get_top_geo(self) :
        """Returns top geometry object
        """
        return self.list_of_geos[-1]
    
    #------------------------------

    def get_pixel_coords(self, oname=None, oindex=0) :
        """Returns pixel coordinate X,Y,Z arrays for top or specified geometry object 
        """
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        return geo.get_pixel_coords()

    #------------------------------
    
    def print_list_of_geos(self) :
        for geo in self.list_of_geos : geo.print_geo()

    #------------------------------
    
    def print_list_of_geos_children(self) :
        for geo in self.list_of_geos : geo.print_geo_children()


  
#------------------------------
#------------------------------
#----------- TESTS ------------
#------------------------------
#------------------------------

from time import time # for test purpose only
import numpy as np

from PyCSPadImage.PixCoords2x1 import cspad2x1_one
import pyimgalgos.GlobalGraphics as gg # for test purpose
import pyimgalgos.TestImageGenerator as tig # for test purpose only

#------------------------------

def test_access(geometry) :

    geometry.print_list_of_geos()
    geometry.print_list_of_geos_children()

    print '\nTOP GEO:'
    top_geo = geometry.get_top_geo()
    top_geo.print_geo_children()

    print '\nINTERMEDIATE GEO (QUAD):'
    geo = geometry.get_geo('QUAD:V1', 0) 
    geo.print_geo_children()

    t0_sec = time()
    X,Y,Z = geo.get_pixel_coords()
    #X,Y = geo.get_2d_pixel_coords()
    print 'X:\n', X
    print 'Consumed time to get 3d pixel coordinates = %7.3f sec' % (time()-t0_sec)
    print 'Geometry object: %s:%d X.shape:%s' % (geo.oname, geo.oindex, str(X.shape))

#------------------------------

def test_plot_quad(geometry) :

    # get pixel coordinate arrays:
    #geo = geometry.get_geo('QUAD:V1', 1) 
    #X,Y,Z = geo.get_pixel_coords()
    X,Y,Z = geometry.get_pixel_coords('QUAD:V1', 1)
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

def test_plot_cspad(geometry, fname_data, amp_range=(0,0.5)) :

    # get pixel coordinate arrays:
    #geo = geometry.get_top_geo() 
    #X,Y,Z = geo.get_pixel_coords()
    X,Y,Z = geometry.get_pixel_coords()
    print 'X,Y,Z shape:', X.shape 

    # get index arrays
    pix_size = cspad2x1_one.get_pix_size_um()
    #xmin, ymin = X.min()-pix_size/2, Y.min()-pix_size/2 
    #iX, iY = np.array((X-xmin)/pix_size, dtype=np.uint), np.array((Y-ymin)/pix_size, dtype=np.uint)

    xyc = xc, yc = 900, 900
    #rad1 =  93
    #rad2 = 146
    rad1 = 655
    rad2 = 670
    x_off_um = xc*pix_size
    y_off_um = yc*pix_size
    iX, iY = np.array((X+x_off_um)/pix_size, dtype=np.uint), np.array((Y+y_off_um)/pix_size, dtype=np.uint)

    arr = np.loadtxt(fname_data, dtype=np.float)
    arr.shape= (4,8,185,388)

    print 'iX, iY, W shape:', iX.shape, iY.shape, arr.shape 
    img = gg.getImageFromIndexArrays(iX,iY,W=arr)

    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.drawCircle(axim, xyc, rad1, linewidth=1, color='w', fill=False) 
    gg.drawCircle(axim, xyc, rad2, linewidth=1, color='w', fill=False) 
    gg.drawCenter(axim, xyc, rad1, linewidth=1, color='w') 
    gg.move(500,10)
    gg.show()

#------------------------------

if __name__ == "__main__" :

    ##fname = '/reg/d/psdm/cxi/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2013-12-20/'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/1-end.data'
    #fname_data     = basedir + 'cspad-ndarr-ave-cxi83714-r0136.dat'
    #amp_range = (0,0.5)

    basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-03-19/'
    fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    fname_data     = basedir + 'cspad-ndarr-ave-cxii0114-r0227.dat'
    amp_range = (0,500)

    geometry = GeometryAccess(fname_geometry)

    msg = 'Use command: sys.argv[0] <num>, wher num=[1,3]'

    if len(sys.argv)==1   : print 'App needs in input parameter.' + msg
    elif sys.argv[1]=='1' : test_access(geometry)
    elif sys.argv[1]=='2' : test_plot_quad(geometry)
    elif sys.argv[1]=='3' : test_plot_cspad(geometry, fname_data, amp_range)
    else : print 'Wrong input parameter.' + msg

    sys.exit ('End of %s' % sys.argv[0])

#------------------------------


