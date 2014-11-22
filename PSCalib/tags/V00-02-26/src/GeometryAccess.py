#!/usr/bin/env python
#------------------------------
""":py:class:`GeometryAccess` - holds and access hierarchical geometry for generic pixel detector

Usage::
 
    from PSCalib.GeometryAccess import GeometryAccess, img_from_pixel_arrays

    fname_geometry = '/reg/d/psdm/CXI/cxitut13/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    geometry = GeometryAccess(fname_geometry, 0377)

    # get pixel coordinate [um] arrays
    X, Y, Z = geometry.get_pixel_coords(oname=None, oindex=0)

    # get pixel area array; A=1 for regular pixels, =2.5 for wide.
    area = geometry.get_pixel_areas(oname=None, oindex=0)

    # get pixel mask array;
    # mbits = +1-mask edges, +2-wide pixels, +4-non-bounded pixels, +8-non-bounded pixel neighbours
    mask = geometry.get_pixel_mask(oname=None, oindex=0, mbits=0377)

    # get index arrays for entire CSPAD
    iX, iY = geometry.get_pixel_coord_indexes()

    # get index arrays for specified quad with offset
    iX, iY = geometry.get_pixel_coord_indexes('QUAD:V1', 1, pix_scale_size_um=None, xy0_off_pix=(1000,1000))

    # get 2-d image from index arrays
    img = img_from_pixel_arrays(iX,iY,W=arr)

    # modify currect geometry objects' parameters
    geometry.set_geo_pars('QUAD:V1', 1, x0, y0, z0, rot_z,...<entire-list-of-9-parameters>);
    geometry.move_geo('QUAD:V1', 1, 10, 20, 0);
    geometry.tilt_geo('QUAD:V1', 1, 0.01, 0, 0);

    # save current geometry parameters in file
    geometry.save_pars_in_file(fname_geometry_new)

    # return geometry parameters in "psf" format for TJ as a tuple psf[32][3][3]
    psf = geometry.get_psf()
    geometry.print_psf()

@see :py:class:`PSCalib.GeometryObject`, :py:class:`PSCalib.SegGeometry`, :py:class:`PSCalib.SegGeometryCspad2x1V1`, :py:class:`PSCalib.SegGeometryStore`

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

    def load_pars_from_file(self, path=None) :
        """Reads input "geometry" file, discards empty lines and comments, fills the list of geometry objects for data lines
        """        
        if path is not None : self.path = path
            
        self.dict_of_comments = {}
        self.list_of_geos = []

        if self.pbits & 32 : print 'Load file: %s' % self.path

        f=open(self.path,'r')
        for linef in f :
            line = linef.strip('\n')
            if self.pbits & 128 : print line
            if not line : continue   # discard empty strings
            if line[0] == '#' :      # process line of comments
                self._add_comment_to_dict(line)
                continue
            #geo=self._parse_line(line)
            self.list_of_geos.append(self._parse_line(line))
    
        f.close()
    
        self._set_relations()
    
    #------------------------------

    def save_pars_in_file(self, path) :
        """Save "geometry" file with current content
        """        

        if self.pbits & 32 : print 'Save file: %s' % path

        txt = ''
        # save comments
        for k in sorted(self.dict_of_comments) :
            txt += '# %10s  %s\n' % (k.ljust(10), self.dict_of_comments[k])

        txt += '\n'        

        # save data
        for geo in self.list_of_geos :
            if geo.get_parent_name() is None : continue
            txt += '%s\n' % (geo.str_data())

        f=open(path,'w')
        f.write(txt)
        f.close()

        if self.pbits & 64 : print txt
    
    #------------------------------
    
    def _add_comment_to_dict(self, line) :
        """Splits the line of comments for keyward and value and store it in the dictionary
        """
        lst = line.lstrip('# ').split(' ', 1)
        if len(lst)<1 : return
        if len(lst)==1 :
            self.dict_of_comments[lst[0]] = ''
            return

        beginline, endline = lst
        #print '  lst      : "%s"' % lst
        #print '  len(lst) : %d' % len(lst)        
        #print '  line     : "%s"' % line

        self.dict_of_comments[beginline] = endline.strip()

    #------------------------------
    
    def _parse_line(self, line) :
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
    
    def _find_parent(self, geobj) :
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

    def _set_relations(self) :
        """Set relations between geometry objects in the list_of_geos
        """
        for geo in self.list_of_geos :
            #geo.print_geo()
            parent = self._find_parent(geo)        

            if parent is None : continue

            geo.set_parent(parent)
            parent.add_child(geo)

            if self.pbits & 16 : print 'geo:%s:%d has parent:%s:%d' % (geo.oname, geo.oindex, parent.oname, parent.oindex)

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

    def get_pixel_mask(self, oname=None, oindex=0, mbits=0377) :
        """Returns pixel mask array for top or specified geometry object.
           mbits =+1 - mask edges
                  +2 - two wide-pixel central columns
                  +4 - non-bounded pixels
                  +8 - neighbours of non-bounded pixels
        """
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        return geo.get_pixel_mask(mbits)

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

    def set_geo_pars(self, oname=None, oindex=0, x0=0, y0=0, z0=0, rot_z=0, rot_y=0, rot_x=0, tilt_z=0, tilt_y=0, tilt_x=0) :
        """Sets geometry parameters for specified or top geometry object
        """
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        return geo.set_geo_pars(x0, y0, z0, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x)

    #------------------------------

    def move_geo(self, oname=None, oindex=0, dx=0, dy=0, dz=0) :
        """Moves specified or top geometry object by dx, dy, dz
        """
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        return geo.move_geo(dx, dy, dz)

    #------------------------------

    def tilt_geo(self, oname=None, oindex=0, dt_x=0, dt_y=0, dt_z=0) :
        """Tilts specified or top geometry object by dt_x, dt_y, dt_z
        """
        geo = self.get_top_geo() if oname is None else self.get_geo(oname, oindex)
        return geo.tilt_geo(dt_x, dt_y, dt_z)

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

    def set_print_bits(self, pbits=0) :
        """ Sets printout control bitword
        """
        self.pbits = pbits

    #------------------------------

    def get_psf(self) :
        """Returns array of vectors in TJ format (psf stands for position-slow-fast vectors)
        """
        X, Y, Z = self.get_pixel_coords() # pixel positions for top level object
        if X.size != 32*185*388 : return None
        # For now it works for CSPAD only
        shape_cspad = (32,185,388)
        X.shape, Y.shape, Z.shape,  = shape_cspad, shape_cspad, shape_cspad

        psf = []

        for s in range(32) :
            vp = (X[s,0,0], Y[s,0,0], Z[s,0,0])

            vs = (X[s,1,0]-X[s,0,0], \
                  Y[s,1,0]-Y[s,0,0], \
                  Z[s,1,0]-Z[s,0,0])

            vf = (X[s,0,1]-X[s,0,0], \
                  Y[s,0,1]-Y[s,0,0], \
                  Z[s,0,1]-Z[s,0,0])

            psf.append((vp,vs,vf))

        return psf


    #------------------------------

    def print_psf(self) :
        """ Gets and prints psf array for test purpose
        """
        psf = np.array( geometry.get_psf() )
        print 'print_psf(): psf.shape: %s \npsf vectors:' % (str(psf.shape)) 
        for (px,py,pz), (sx,xy,xz), (fx,fy,fz) in psf:
            print '    p=(%12.2f, %12.2f, %12.2f),    s=(%8.2f, %8.2f, %8.2f)   f=(%8.2f, %8.2f, %8.2f)' \
                  % (px,py,pz,  sx,xy,xz,  fx,fy,fz)

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

def img_from_pixel_arrays(iX, iY, W=None, dtype=np.float32, vbase=0) :
    """Returns image from iX, iY coordinate index arrays and associated weights W.
    """
    if iX.size != iY.size \
    or (W!=None and iX.size !=  W.size) :
        msg = 'img_from_pixel_arrays(): WARNING input array sizes are different;' \
            + ' iX.size=%d, iY.size=%d, W.size=%d' % (iX.size, iY.size, W.size)
        print msg
        return img_default()

    xsize = iX.max()+1 
    ysize = iY.max()+1

    weight = W if W!=None else np.ones_like(iX)
    img = vbase*np.ones((xsize,ysize), dtype=dtype)
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
    #geo = geometry.get_top_geo() 
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

def test_mask_quad(geometry) :
    """ Tests geometry acess methods of the class GeometryAccess object for CSPAD quad
    """
    ## get index arrays
    iX, iY = geometry.get_pixel_coord_indexes('QUAD:V1', 1, pix_scale_size_um=None, xy0_off_pix=None)

    # get intensity array
    arr = geometry.get_pixel_mask('QUAD:V1', 1, 1+2+4+8)
    arr.shape = (8,185,388)
    amp_range = (-1,2)
 
    print 'iX, iY, W shape:', iX.shape, iY.shape, arr.shape 
    img = img_from_pixel_arrays(iX, iY, W=arr, vbase=0.5)

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

    arr.shape = iX.shape
    img = img_from_pixel_arrays(iX, iY, W=arr)

    xyc_ring = (yc, xc)
    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.drawCircle(axim, xyc_ring, rad1, linewidth=1, color='w', fill=False) 
    gg.drawCircle(axim, xyc_ring, rad2, linewidth=1, color='w', fill=False) 
    gg.drawCenter(axim, xyc_ring, rad1, linewidth=1, color='w') 
    gg.move(500,10)
    gg.show()

#------------------------------

def test_img_default() :
    """ Test default image
    """
    axim = gg.plotImageLarge( img_default() )
    gg.move(500,10)
    gg.show()

#------------------------------

def test_save_pars_in_file(geometry) :
    """ Test default image
    """
    geometry.set_print_bits(32)
    geometry.save_pars_in_file('./test.txt')

#------------------------------

def test_load_pars_from_file(geometry) :
    """ Test default image
    """
    geometry.set_print_bits(32+64)
    geometry.load_pars_from_file('./test.txt')
    geometry.print_list_of_geos()

#------------------------------

def test_cspad2x2() :
    """ Test cspad2x2 geometry table
    """
    ## MecTargetChamber.0:Cspad2x2.1 
    basedir = '/reg/neh/home1/dubrovin/LCLS/CSPad2x2Alignment/calib-cspad2x2-01-2013-02-13/'    
    fname_geometry = basedir + 'calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/geometry/0-end.data'
    fname_data     = basedir + 'cspad2x2.1-ndarr-ave-meca6113-r0028.dat'    

    ## MecTargetChamber.0:Cspad2x2.2 
    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPad2x2Alignment/calib-cspad2x2-02-2013-02-13/'    
    #fname_geometry = basedir + 'calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.2/geometry/0-end.data'
    #fname_data     = basedir + 'cspad2x2.2-ndarr-ave-meca6113-r0028.dat'    

    geometry = GeometryAccess(fname_geometry, 0377)
    amp_range = (0,15000)

    # get pixel coordinate index arrays:
    #xyc = xc, yc = 1000, 1000
    #iX, iY = geometry.get_pixel_coord_indexes(xy0_off_pix=xyc)

    iX, iY = geometry.get_pixel_coord_indexes()

    root, ext = os.path.splitext(fname_data)
    arr = np.load(fname_data) if ext == '.npy' else np.loadtxt(fname_data, dtype=np.float) 
    arr.shape= (185,388,2)

    print 'iX, iY, W shape:', iX.shape, iY.shape, arr.shape 
    img = img_from_pixel_arrays(iX,iY,W=arr)

    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()

#------------------------------

def test_epix100a() :
    """ Test test_epix100a geometry table
    """
    ## MecTargetChamber.0:Cspad2x2.1 
    basedir = '/reg/neh/home1/dubrovin/LCLS/GeometryCalib/calib-xpp-Epix100a-2014-11-05/'    
    fname_geometry = basedir + 'calib/Epix100a::CalibV1/NoDetector.0:Epix100a.0/geometry/0-end.data'
    fname_data     = basedir + 'epix100a-ndarr-ave-clb-xppi0614-r0073.dat'    

    geometry = GeometryAccess(fname_geometry, 0177777)
    amp_range = (-4,10)

    iX, iY = geometry.get_pixel_coord_indexes()

    root, ext = os.path.splitext(fname_data)
    arr = np.load(fname_data) if ext == '.npy' else np.loadtxt(fname_data, dtype=np.float) 

    print 'iX, iY, W shape:', iX.shape, iY.shape, arr.shape 
    img = img_from_pixel_arrays(iX,iY,W=arr)

    axim = gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()

#------------------------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :

    ##fname = '/reg/d/psdm/cxi/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2013-12-20/'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/1-end.data'
    #fname_data     = basedir + 'cspad-ndarr-ave-cxi83714-r0136.dat'
    #amp_range = (0,0.5)

    # CXI
    basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-03-19/'
    fname_data     = basedir + 'cspad-ndarr-ave-cxii0114-r0227.dat'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #fname_geometry = '/reg/d/psdm/cxi/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    fname_geometry = '/reg/d/psdm/CXI/cxitut13/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    amp_range = (0,500)

    ## XPP
    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-xpp-2013-01-29/'
    #fname_data     = basedir + 'cspad-xpptut13-r1437-nda.dat'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/XppGon.0:Cspad.0/geometry/0-end.data'
    #amp_range = (1500,2500)


    #basedir = '/home/pcds/LCLS/calib/geometry/'
    #fname_geometry = basedir + '0-end.data'
    #fname_geometry = basedir + '2-end.data'
    #fname_data     = basedir + 'cspad-ndarr-ave-cxii0114-r0227.dat'
    #fname_geometry = '/reg/d/psdm/cxi/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    #amp_range = (0,500)

    #basedir = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-05-15/'
    #fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data'
    #fname_data     = basedir + 'cspad-arr-cxid2714-r0023-lysozyme-rings.npy'
    #amp_range = (0,500)

    geometry = GeometryAccess(fname_geometry, 0)

    msg = 'Use command: sys.argv[0] <num>, wher num=1,2,3,...,10'

    if len(sys.argv)==1   : print 'App needs in input parameter.' + msg
    elif sys.argv[1]=='1' : test_access(geometry)
    elif sys.argv[1]=='2' : test_plot_quad(geometry)
    elif sys.argv[1]=='3' : test_plot_cspad(geometry, fname_data, amp_range)
    elif sys.argv[1]=='4' : test_img_default()
    elif sys.argv[1]=='5' :
        print 'Init GeometryAccess is silent? (see below)'
        ga0 = GeometryAccess(fname_geometry, 0)
    elif sys.argv[1]=='6' : ga0377 = GeometryAccess(fname_geometry, 0377)
    elif sys.argv[1]=='7' : test_save_pars_in_file(geometry)
    elif sys.argv[1]=='8' : test_load_pars_from_file(geometry)
    elif sys.argv[1]=='9' : test_mask_quad(geometry)
    elif sys.argv[1]=='10': geometry.print_psf()
    elif sys.argv[1]=='11': test_cspad2x2()
    elif sys.argv[1]=='12': test_epix100a()
    else : print 'Wrong input parameter.' + msg

    sys.exit ('End of %s' % sys.argv[0])

#------------------------------


