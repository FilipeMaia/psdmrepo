#!/usr/bin/env python
#------------------------------
"""GeometryAccess - holds and access hierarchical geometry for generic pixel detector

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

Revision: $Revision$

@version $Id$

@author Mikhail S. Dubrovin

Interface

"""
#--------------------------------
__version__ = "$Revision$"
#--------------------------------

import os
import sys
from PSCalib.GeometryObject import GeometryObject

import numpy as np

#------------------------------

class GeometryAccess :
    """ :py:class:`GeometryAccess`
    """

    def __init__(self, path, pbits=0) : 
        """  Constructor of the class :py:class:`GeometryAccess`      
        """        
        if not os.path.exists(path) :
            sys.exit ('file %s does not exist' % path)

        self.path = path
        self.pbits = pbits
        self.load_pars_from_file()

        if self.pbits & 1 : self.print_list_of_geos()
        if self.pbits & 2 : self.print_list_of_geos_children()
        if self.pbits & 4 : self.print_comments_from_dict()

    #------------------------------

    def load_pars_from_file(self) :
        """Reads input "geometry" file, discards empty lines and comments, fills the list of geometry objects for data lines
        """        
        self.dict_of_comments = {}
        self.list_of_geos = []

        f=open(self.path,'r')
        for linef in f :
            line = linef.strip('\n')
            if not line : continue   # discard empty strings
            if line[0] == '#' :      # process line of comments
                self.add_comment_to_dict(line)
                continue
            #print line
            #geo=self.parse_line(line)
            self.list_of_geos.append(self.parse_line(line))
    
        f.close()
    
        self.set_relations()
    
    #------------------------------
    
    def add_comment_to_dict(self, line) :
        """Splits the line of comments for keyward and value and store it in the dictionary
        """
        beginline, endline = line.lstrip('# ').split(' ', 1)
        #print 'line: ', line
        #print '  split parts - key:%s  val:%s' % (beginline,  endline)
        self.dict_of_comments[beginline] = endline.strip()

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
            if  geo.oindex == geobj.pindex \
            and geo.oname  == geobj.pname :
                return geo
    
        # The name of parent object is not found among geo names in the self.list_of_geos
        # add top parent object to the list
        if geobj.pname is not None :
            top_parent = GeometryObject(pname=None, pindex=0, oname=geobj.pname, oindex=geobj.pindex)
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

            if parent is None : continue

            geo.set_parent(parent)
            parent.add_child(geo)

            print 'geo:%s:%d has parent:%s:%d' % (geo.oname, geo.oindex, parent.oname, parent.oindex)

    #------------------------------

    def get_geo(self, oname, oindex) :
        """Returns specified geometry object
        """
        for geo in self.list_of_geos :
            if  geo.oindex == oindex \
            and geo.oname  == oname :
                return geo
        return None
    
    #------------------------------
    
    def get_top_geo(self) :
        """Returns top geometry object
        """
        return self.list_of_geos[-1]
    
    #------------------------------

    def get_pixel_coords(self, oname=None, oindex=0) :
        """Returns three pixel X,Y,Z coordinate arrays for top or specified geometry object 
        """
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        if self.pbits & 8 :
            print 'get_pixel_coords(...) for geo:',
            geo.print_geo_children();
        
        return geo.get_pixel_coords()

    #------------------------------

    def get_pixel_areas(self, oname=None, oindex=0) :
        """Returns pixel areas array for top or specified geometry object 
        """
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        return geo.get_pixel_areas()

    #------------------------------

    def get_pixel_scale_size(self, oname=None, oindex=0) :
        """Returns pixel scale size for top or specified geometry object 
        """
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)        
        return geo.get_pixel_scale_size()

      #------------------------------
    
    def get_dict_of_comments(self) :
        """Returns dictionary of comments
        """
        return self.dict_of_comments

    #------------------------------
    
    def print_list_of_geos(self) :
        ss = '\nprint_list_of_geos():'
        if len(self.list_of_geos) == 0 : print '%s List_of_geos is empty...' % ss
        for geo in self.list_of_geos : geo.print_geo()

    #------------------------------
    
    def print_list_of_geos_children(self) :
        ss = '\nprint_list_of_geos_children():'
        if len(self.list_of_geos) == 0 : print '%s List_of_geos is empty...' % ss
        for geo in self.list_of_geos : geo.print_geo_children()

    #------------------------------
    
    def print_comments_from_dict(self) :
        print '\nprint_comments_from_dict():'
        #for k,v in self.dict_of_comments.iteritems():
        for k in sorted(self.dict_of_comments):
            print 'key: %s  val: %s' % (k.ljust(10), self.dict_of_comments[k])

    #------------------------------

    def print_pixel_coords(self, oname=None, oindex=0) :
        """Partial print of pixel coordinate X,Y,Z arrays for selected or top(by default) geo
        """
        X, Y, Z = self.get_pixel_coords(oname, oindex)

        print 'size=', X.size
        print 'X: %s...'% ', '.join(['%10.1f'%v for v in X.flatten()[0:9]])
        print 'Y: %s...'% ', '.join(['%10.1f'%v for v in Y.flatten()[0:9]])
        print 'Z: %s...'% ', '.join(['%10.1f'%v for v in Z.flatten()[0:9]])

    #------------------------------

    def get_pixel_coord_indexes(self, oname=None, oindex=0, pix_scale_size_um=None, xy0_off_pix=None) :
        """Returns three pixel X,Y,Z coordinate index arrays for top or specified geometry object 
        """
        X, Y, Z = self.get_pixel_coords(oname, oindex)

        pix_size = self.get_pixel_scale_size() if pix_scale_size_um is None else pix_scale_size_um
        pix_half = pix_size/2

        if xy0_off_pix is not None :
            # Offset in pix -> um
            x_off_um = xy0_off_pix[0] * pix_size
            y_off_um = xy0_off_pix[1] * pix_size
            # Protection against wrong offset bringing negative indexes
            xmin = (X+x_off_um).min()
            ymin = (Y+y_off_um).min()
            x_off_um = x_off_um + pix_half if xmin>0 else x_off_um - xmin + pix_half
            y_off_um = y_off_um + pix_half if ymin>0 else y_off_um - ymin + pix_half

            iX, iY = np.array((X+x_off_um)/pix_size, dtype=np.uint), np.array((Y+y_off_um)/pix_size, dtype=np.uint)
            return iX, iY
        else :
            # Find coordinate min values
            xmin, ymin = X.min()-pix_size/2, Y.min()-pix_size/2 
            iX, iY = np.array((X-xmin)/pix_size, dtype=np.uint), np.array((Y-ymin)/pix_size, dtype=np.uint)
            return iX, iY


#------------------------------
#------ Global Method(s) ------
#------------------------------

def img_default(shape=(10,10), dtype = np.float32) :
    """Returns default image
    """
    arr = np.arange(shape[0]*shape[1], dtype=dtype)
    arr.shape = shape
    return arr

#------------------------------

def img_from_pixel_arrays(iX, iY, W=None, dtype=np.float32) :
    """Returns image from iX, iY coordinate index arrays and associated weights W.
    """
    if iX.size != iY.size \
    or iX.size !=  W.size :
        msg = 'img_from_pixel_arrays(): WARNING input array sizes are different;' \
            + ' iX.size=%d, iY.size=%d, W.size=%d' % (iX.size, iY.size, W.size)
        print msg
        return img_default()

    xsize = iX.max()+1 
    ysize = iY.max()+1
    if W==None : weight = np.ones_like(iX)
    else       : weight = W
    img = np.zeros((xsize,ysize), dtype=dtype)
    img[iX,iY] = weight # Fill image array with data 
    return img

#------------------------------
#------------------------------
#----------- TESTS ------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    from time import time # for test purpose only

    #from PSCalib.SegGeometryCspad2x1V1 import cspad2x1_one
    import pyimgalgos.GlobalGraphics as gg # for test purpose
    import pyimgalgos.TestImageGenerator as tig # for test purpose only

#------------------------------

def test_access(geometry) :
    """ Tests geometry acess methods of the class GeometryAccess
    """
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

    print '\nTest of print_pixel_coords() for quad:'
    geometry.print_pixel_coords('QUAD:V1', 1)
    print '\nTest of print_pixel_coords() for CSPAD:'
    geometry.print_pixel_coords()

    print '\nTest of get_pixel_areas() for QUAD:'
    A = geo.get_pixel_areas()
    print 'Geometry object: %s:%d A.shape:%s' % (geo.oname, geo.oindex, str(A.shape))
    print 'A[0,0:5,190:198]:\n', A[0,0:5,190:198]
 
    print '\nTest of get_pixel_areas() for CSPAD:'
    A = top_geo.get_pixel_areas()
    print 'Geometry object: %s:%d A.shape:%s' % (geo.oname, geo.oindex, str(A.shape))
    print 'A[0,0,0:5,190:198]:\n', A[0,0,0:5,190:198]

    print '\nTest of get_size_geo_array()'
    print 'for QUAD : %d' % geo.get_size_geo_array()
    print 'for CSPAD: %d' % top_geo.get_size_geo_array()

    print '\nTest of get_pixel_scale_size()'
    print 'for QUAD    : %8.2f' % geo.get_pixel_scale_size()
    print 'for CSPAD   : %8.2f' % top_geo.get_pixel_scale_size()
    print 'for geometry: %8.2f' % geometry.get_pixel_scale_size()

    print '\nTest of get_dict_of_comments():'
    d = geometry.get_dict_of_comments()
    print "d['HDR'] = %s" % d['HDR']

#------------------------------

def test_plot_quad(geometry) :
    """ Tests geometry acess methods of the class GeometryAccess object for CSPAD quad
    """
    ## get index arrays
    iX, iY = geometry.get_pixel_coord_indexes('QUAD:V1', 1, pix_scale_size_um=None, xy0_off_pix=None)

    # get intensity array
    arr = tig.cspad_nparr(n2x1=iX.shape[0])
    arr.shape = (8,185,388)
    amp_range = (0,185+388)
 
    print 'iX, iY, W shape:', iX.shape, iY.shape, arr.shape 
    img = img_from_pixel_arrays(iX,iY,W=arr)

    gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()

#------------------------------

def test_plot_cspad(geometry, fname_data, amp_range=(0,0.5)) :
    """ The same test as previous, but use get_pixel_coord_indexes(...) method
    """
    #rad1 =  93
    #rad2 = 146
    rad1 = 655
    rad2 = 670

    # get pixel coordinate index arrays:
    xyc = xc, yc = 1000, 1000
    #iX, iY = geometry.get_pixel_coord_indexes(xy0_off_pix=None)
    iX, iY = geometry.get_pixel_coord_indexes(xy0_off_pix=xyc)

    root, ext = os.path.splitext(fname_data)
    arr = np.load(fname_data) if ext == '.npy' else np.loadtxt(fname_data, dtype=np.float) 
    arr.shape= (4,8,185,388)

    print 'iX, iY, W shape:', iX.shape, iY.shape, arr.shape 
    img = img_from_pixel_arrays(iX,iY,W=arr)

    xyc_ring = (yc, xc)
    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.drawCircle(axim, xyc_ring, rad1, linewidth=1, color='w', fill=False) 
    gg.drawCircle(axim, xyc_ring, rad2, linewidth=1, color='w', fill=False) 
    gg.drawCenter(axim, xyc_ring, rad1, linewidth=1, color='w') 
    gg.move(500,10)
    gg.show()

#------------------------------

def test_img_default() :
    """ The same test as previous, but use get_pixel_coord_indexes(...) method
    """
    axim = gg.plotImageLarge( img_default() )
    gg.move(500,10)
    gg.show()

#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :

    ##fname = '/reg/d/psdm/cxi/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2013-12-20/'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/1-end.data'
    #fname_data     = basedir + 'cspad-ndarr-ave-cxi83714-r0136.dat'
    #amp_range = (0,0.5)

    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-03-19/'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #fname_data     = basedir + 'cspad-ndarr-ave-cxii0114-r0227.dat'
    #fname_geometry = '/reg/d/psdm/cxi/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #amp_range = (0,500)

    basedir = '/home/pcds/LCLS/calib/geometry/'
    #fname_geometry = basedir + '0-end.data'
    fname_geometry = basedir + '2-end.data'
    fname_data     = basedir + 'cspad-ndarr-ave-cxii0114-r0227.dat'
    amp_range = (0,500)

    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-05-15/'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data'
    #fname_data     = basedir + 'cspad-arr-cxid2714-r0023-lysozyme-rings.npy'
    #amp_range = (0,500)

    geometry = GeometryAccess(fname_geometry, 0377)

    msg = 'Use command: sys.argv[0] <num>, wher num=[1,4]'

    if len(sys.argv)==1   : print 'App needs in input parameter.' + msg
    elif sys.argv[1]=='1' : test_access(geometry)
    elif sys.argv[1]=='2' : test_plot_quad(geometry)
    elif sys.argv[1]=='3' : test_plot_cspad(geometry, fname_data, amp_range)
    elif sys.argv[1]=='4' : test_img_default()
    else : print 'Wrong input parameter.' + msg

    sys.exit ('End of %s' % sys.argv[0])

#------------------------------


