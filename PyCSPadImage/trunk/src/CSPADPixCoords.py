#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CSPADPixCoords...
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
import GlobalGraphics  as gg # For test purpose in main only
#------------------------------

class CSPADPixCoords (PixCoords2x1) :
    """Self-sufficient class for generation of CSPad pixel coordinate array without data base (WODB)"""

    quads = 4 # Total number of quads in cspad
    sects = 8 # Total number of sections in quad

    # Old default:
    #xc_um_def = np.array(
    #        [[ 473.38,  685.26,  155.01,  154.08,  266.81,   53.95,  583.04,  582.15],  
    #         [ 989.30,  987.12, 1096.93,  884.11, 1413.16, 1414.94, 1500.83, 1288.02],  
    #         [1142.59,  930.23, 1459.44, 1460.67, 1347.57, 1559.93, 1032.27, 1033.44],  
    #         [ 626.78,  627.42,  516.03,  729.15,  198.28,  198.01,  115.31,  327.66]]) * PixCoords2x1.pixs # 109.92
     
    #yc_um_def = np.array(
    #        [[1028.07, 1026.28, 1139.46,  926.91, 1456.78, 1457.35, 1539.71, 1327.89],  
    #         [1180.51,  967.36, 1497.74, 1498.54, 1385.08, 1598.19, 1069.65, 1069.93],  
    #         [ 664.89,  666.83,  553.60,  765.91,  237.53,  236.06,  152.17,  365.47],  
    #         [ 510.38,  722.95,  193.33,  193.41,  308.04,   95.25,  625.28,  624.14]]) * PixCoords2x1.pixs # 109.92


    # Optical measurement from 2013-01-29
    xc_um_def  = np.array(
          [[ 477.78,    690.20,    159.77,    160.06,    277.17,     64.77,    591.30,    591.01],
           [ 990.78,    989.30,   1105.38,    891.19,   1421.65,   1423.66,   1502.28,   1289.93],
           [1143.85,    932.00,   1461.86,   1463.74,   1349.75,   1562.62,   1032.39,   1033.60],
           [ 633.06,    632.80,    518.88,    731.75,    200.62,    198.75,    118.50,    331.23]] ) * PixCoords2x1.pixs # 109.92

    yc_um_def = np.array(
          [[1018.54,   1019.42,   1134.27,    921.94,   1451.06,   1451.01,   1532.55,   1319.23],
           [1173.24,    960.71,   1490.18,   1491.45,   1374.97,   1587.78,   1058.56,   1061.14],
           [ 658.23,    658.54,    542.73,    755.26,    225.91,    224.22,    146.39,    358.27],
           [ 507.44,    720.59,    189.73,    190.28,    306.25,     93.65,    620.68,    619.85]] ) * PixCoords2x1.pixs # 109.92

    zc_um_def = np.zeros((4,8), dtype=np.float32)

    # Orientation of sect:   0   1     2     3     4     5     6     7
    orient_opt = np.array ( [0., 0., 270., 270., 180., 180., 270., 270.] )

     # Orientation of quad:         0              1           2              3 
    orient_def = np.array ( [orient_opt+90, orient_opt, orient_opt-90, orient_opt-180] )

    # Tilt:
    tilt_2013_01_29 = np.array (
          [[  0.27766,   0.37506,   0.11976,   0.17369,  -0.04934,   0.01119,   0.13752,  -0.00921],   
           [ -0.45066,  -0.18880,  -0.20400,  -0.33507,  -0.62242,  -0.40196,  -0.56593,  -0.59475],   
           [ -0.03290,   0.00658,  -0.33954,  -0.27106,  -0.71923,  -0.31647,   0.02829,   0.10723],   
           [ -0.11054,   0.10658,   0.25005,   0.16121,  -0.58560,  -0.43369,  -0.26916,  -0.18225]] )

    #tilt_def = tilt_2013_01_29
    tilt_def = np.zeros((quads, sects), dtype=np.float32)

#------------------------------

    def __init__ (sp, calib=None, xc_um=None, yc_um=None, tilt_deg=None, use_wide_pix_center=False) :
        #print 'CSPAD2x2PixCoords.__init__(...)'

        PixCoords2x1.__init__ (sp, use_wide_pix_center)

        if calib == None :

            if xc_um == None : sp.xc = sp.xc_um_def
            else             : sp.xc = xc_um

            if yc_um == None : sp.yc = sp.yc_um_def
            else             : sp.yc = yc_um

            sp.zc   = sp.zc_um_def 

            if tilt_deg == None : sp.tilt = sp.tilt_def
            else                : sp.tilt = tilt_deg 

        else :
            [sp.xc,sp.yc,sp.zc] = calib.getCalibPars('center_global') * PixCoords2x1.pixs # 109.92
            sp.tilt             = calib.getCalibPars('tilt')
            #print 'USE xc, yc, zc =', sp.xc, sp.yc, sp.zc

        sp.orient        = sp.orient_def

        sp.calib = calib
        sp.make_cspad_pix_coordinate_arrays (sp.xc, sp.yc, sp.orient, sp.tilt)

#------------------------------

    def print_cspad_geometry_pars (sp) :
        msg = 'print_cspad_geometry_pars():' \
            + '\nxc [pix]:\n' + str( sp.xc/PixCoords2x1.pixs ) \
            + '\nyc [pix]:\n' + str( sp.yc/PixCoords2x1.pixs ) \
            + '\norient:\n'   + str( sp.orient ) \
            + '\ntilt:\n'     + str( sp.tilt )
            #+ '\nxc:'       + str( sp.xc ) \
            #+ '\nyc:'       + str( sp.yc ) \
        print msg

#------------------------------

    def make_cspad_pix_coordinate_arrays (sp, xc_um, yc_um, orient_deg, tilt_deg=None) : # All lists of [4,8]
        """Makes [4,8,185,388] cspad pixel x and y coordinate arrays"""        
        sp.make_maps_of_2x1_pix_coordinates()

        sp.x_pix_um = np.zeros((sp.quads,sp.sects,sp.rows,sp.cols), dtype=np.float32)
        sp.y_pix_um = np.zeros((sp.quads,sp.sects,sp.rows,sp.cols), dtype=np.float32)

        angle_deg = np.array(orient_deg)
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

        sp.x_pix_pix = (sp.x_pix_um/sp.pixs+0.25).astype(int) 
        sp.y_pix_pix = (sp.y_pix_um/sp.pixs+0.25).astype(int)

#------------------------------

    def get_cspad_pix_coordinate_arrays_um (sp) : 
        return sp.x_pix_um, sp.y_pix_um


    def get_cspad_pix_coordinate_arrays_pix (sp) : 
        return sp.x_pix_pix, sp.y_pix_pix

#------------------------------

    def print_cspad_coordinate_arrays(sp) :
        print 'sp.x_pix_um:\n',      sp.x_pix_um
        print 'sp.x_pix_um.shape =', sp.x_pix_um.shape
        print 'sp.y_pix_um\n',       sp.y_pix_um
        print 'sp.y_pix_um.shape =', sp.y_pix_um.shape

#------------------------------

    def get_cspad_image(sp, data_arr=None) : # preferable data_arr.shape=(32,185,388) like in data
        """ Test of coordinate arrays, plot image map.
        """
        iX,iY = sp.get_cspad_pix_coordinate_arrays_pix()
        data = data_arr
        if data_arr != None and data.shape != (4,8,185,388) :
            data.shape == (4,8,185,388)
            data = data.flatten()
        return gg.getImageFromIndexArrays(iX.flatten(),iY.flatten(),data) # All arrays should have the same shape

#------------------------------
#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------
#------------------------------

def test_of_coord_arrs_h2(coord) :
    """ DEPRICATED: Test of coordinate arrays, plot image map.
    """
    coord.print_cspad_geometry_pars()
    X,Y = coord.get_cspad_pix_coordinate_arrays_pix ()
    arr = np.ones((32, 185, 388), dtype=np.float32)

    #print 'X(pix) :\n', X
    print 'X.shape =\n', X.shape

    xsize = X.max() + 1
    ysize = Y.max() + 1
    H,Xedges,Yedges = np.histogram2d(X.flatten(), Y.flatten(), bins=[xsize,ysize], range=[[0,xsize],[0,ysize]], normed=False, weights=arr.flatten()) 

    range = (-5, 5)
    gg.plotImageLarge(H, amp_range=(0, 2000), figsize=(12,11))
    gg.show()

#------------------------------

def test_of_coord_arrs(coord) :
    """ Test of coordinate arrays, plot image map.
    """

    iX,iY = coord.get_cspad_pix_coordinate_arrays_pix ()

    print 'iX.shape =', iX.shape
    print 'iY.shape =', iY.shape

    t0_sec = time()
    #img2d = gg.getImageAs2DHist(X,Y,W=None)
    img2d = gg.getImageFromIndexArrays(iX,iY,W=None)
    print 'Consumed time to create image (sec) =', time()-t0_sec

    gg.plotImageLarge(img2d, amp_range=(-1, 2), figsize=(12,11)) #amp_range=(0, 2000)
    gg.show()


#------------------------------

def test_cspadpixcoords_instantiation_1() :
    """ Instantiation with external geometry parameters.
    """
    # Optical measurement for XPP from 2013-01-29
    xc = np.array(
          [[ 477.78,    690.20,    159.77,    160.06,    277.17,     64.77,    591.30,    591.01],
           [ 990.78,    989.30,   1105.38,    891.19,   1421.65,   1423.66,   1502.28,   1289.93],
           [1143.85,    932.00,   1461.86,   1463.74,   1349.75,   1562.62,   1032.39,   1033.60],
           [ 633.06,    632.80,    518.88,    731.75,    200.62,    198.75,    118.50,    331.23]] ) * PixCoords2x1.pixs # 109.92

    yc = np.array(
          [[1018.54,   1019.42,   1134.27,    921.94,   1451.06,   1451.01,   1532.55,   1319.23],
           [1173.24,    960.71,   1490.18,   1491.45,   1374.97,   1587.78,   1058.56,   1061.14],
           [ 658.23,    658.54,    542.73,    755.26,    225.91,    224.22,    146.39,    358.27],
           [ 507.44,    720.59,    189.73,    190.28,    306.25,     93.65,    620.68,    619.85]] ) * PixCoords2x1.pixs # 109.92

    tilt = np.array([[  0.27766,   0.37506,   0.11976,   0.17369,  -0.04934,   0.01119,   0.13752,  -0.00921],   
                     [ -0.45066,  -0.18880,  -0.20400,  -0.33507,  -0.62242,  -0.40196,  -0.56593,  -0.59475],   
                     [ -0.03290,   0.00658,  -0.33954,  -0.27106,  -0.71923,  -0.31647,   0.02829,   0.10723],   
                     [ -0.11054,   0.10658,   0.25005,   0.16121,  -0.58560,  -0.43369,  -0.26916,  -0.18225]] )

    #tilt = np.zeros((4, 8), dtype=np.float32)

    t0_sec = time()
    coord = CSPADPixCoords(xc_um=xc, yc_um=yc, tilt_deg=tilt, use_wide_pix_center=False)
    coord.print_cspad_geometry_pars()
    print 'Consumed time for CSPADPixCoords instatiation (sec) =', time()-t0_sec
    return coord

#------------------------------

from PyCSPadImage.CalibPars import *

def test_cspadpixcoords_instantiation_2() :
    """ Instantiation with regular calibration parameters.
    """
    path = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-xpp-2013-01-29'
    run   = 123
    calib = CalibPars(path, run)
    print 'center_global:\n', calib.getCalibPars ('center_global') 
    coord = CSPADPixCoords(calib)
    coord.print_cspad_geometry_pars()
    return coord

#------------------------------

def test_cspadpixcoords_0() :
    """Test of default constructor.
    """
    coord = CSPADPixCoords() 
    coord.print_cspad_geometry_pars()
    test_of_coord_arrs(coord)

#------------------------------

def test_cspadpixcoords_1() :
    """Test of instantiation with external parameters.
    """
    coord = test_cspadpixcoords_instantiation_1() 
    test_of_coord_arrs(coord)

#------------------------------

def test_cspadpixcoords_2() :
    """Test of instantiation with calib=CSPADCalibPars(path, run).
    """
    coord = test_cspadpixcoords_instantiation_2() 
    coord.print_cspad_geometry_pars()
    test_of_coord_arrs(coord)

#------------------------------

def test_cspadpixcoords_3() :
    """Test of instantiation with external parameters.
    """
    coord = test_cspadpixcoords_instantiation_1() 
    img2d = coord.get_cspad_image(None)
    print 'img2d.shape =', img2d.shape
    
    gg.plotImageLarge(img2d, amp_range=(-1, 2), figsize=(12,11))
    gg.show()

#------------------------------

if __name__ == "__main__" :
    if len(sys.argv)==1   : print 'Use command: python', sys.argv[0], '<test-number=0-3>'
    elif sys.argv[1]=='0' : test_cspadpixcoords_0() # Instatiation default
    elif sys.argv[1]=='1' : test_cspadpixcoords_1() # Instatiation using external geometry parameters
    elif sys.argv[1]=='2' : test_cspadpixcoords_2() # Instatiation using calib = CalibPars(path, run)
    elif sys.argv[1]=='3' : test_cspadpixcoords_3() # Test of coord.get_cspad_image()
    else : print 'Non-expected arguments: sys.argv=', sys.argv

    sys.exit ( 'End of test.' )

#------------------------------
