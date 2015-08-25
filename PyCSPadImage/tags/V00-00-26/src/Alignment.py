#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Alignment...
#
#------------------------------------------------------------------------

"""This module provides examples of how to get and use the CSPad image

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 2014-03-24$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#----------
#  Imports 
#----------
import sys
import os
import numpy as np

import PyCSPadImage.CalibParsDefault   as cald
import PyCSPadImage.CalibPars          as calp
import PyCSPadImage.CSPadConfigPars    as ccp
import PyCSPadImage.CSPadImageProducer as cip
import PyCSPadImage.GlobalMethods      as gm # getCSPadArrayFromFile for pedestal subtraction 
import PyCSPadImage.CSPADPixCoords     as pixcoor
import PyCSPadImage.HDF5Methods        as hm # For test purpose in main only

import pyimgalgos.GlobalGraphics          as gg # For test purpose in main only
import pyimgalgos.FastArrayTransformation as fat 
import pyimgalgos.AngularIntegrator       as ai

#----------------------------------------------

def main_example_CSpad2x2() :

    print 'Start test in main_example_CSpad2x2()'

    fname = '/reg/d/psdm/xpp/xppi0212/hdf5/xppi0212-r0046.h5'
    dsname = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad2x2::ElementV1/XppGon.0:Cspad2x2.0/data'
    event = 0

    h5file = hm.hdf5mets.open_hdf5_file(fname)
    #grp = hm.hdf5mets.get_dataset_from_hdf5_file('/')    
    grp = hm.hdf5mets.get_dataset_from_hdf5_file('/Configure:0000/Run:0000/CalibCycle:0000/CsPad2x2::ElementV1')    
    hm.print_hdf5_item_structure(grp)
    arrevts = hm.hdf5mets.get_dataset_from_hdf5_file(dsname)
    arr1ev = arrevts[event]
    hm.hdf5mets.close_hdf5_file()

    #print 'arr1ev=\n',       arr1ev
    print 'arr1ev.shape=\n', arr1ev.shape
    #arr = arr1ev[:,:,0]

    cspadimg = cip.CSPadImageProducer()
    arr = cspadimg.getImageArrayForCSpad2x2Element( arr1ev )

    AmpRange = (0,1200)
    gg.plotImage(arr,range=AmpRange,figsize=(11.6,10))
    gg.move(300,100)

    gg.plotSpectrum(arr,range=AmpRange)
    gg.move(10,100)

    gg.show()

#----------------------------------------------

def plot_img_in_polar_coords(image, center=(50,40), rad_range=(600,700,100), phi_range=(-180,180,90), amp_range=None, rad_plot=700) :

    origin = center
    RRange = rad_range
    PRange = phi_range
    RPRange= (PRange[0], PRange[1], RRange[0], RRange[1])

    polar_arr = fat.transformCartToPolarArray(image, RRange, PRange, origin)

    axis = gg.plotImageLarge(polar_arr, img_range=RPRange, amp_range=amp_range, figsize=(12,8), origin='down')
    gg.drawLine(axis, xarr=(PRange[0], PRange[1]), yarr=(rad_plot, rad_plot))
    gg.savefig('cspad-img-r-phi.png')

#----------------------------------------------

def alignment_for_cspad_ndarray(fname, amps=(0,500), center=(877.0,875.5), rad_range=(600,720,120), radius=664, amps_rad=(0,20000), path_calib='./', runnum=1) :

    #xc, yc = 859, 859
    #xc, yc = 855.3+22.7, 860.2+14.5
    xc, yc = center
    rmin, rmax, rbins = rad_range
    nda = gm.getCSPadArrayFromFile(fname, dtype=np.float32, shape = (32, 185, 388)) 

    print 'Use default configuration parameters for entire cspad'
    config = ccp.CSPadConfigPars()
    config.printCSPadConfigPars()

    print 'Start alignment_for_cspad_ndarray()'
    print 'Load calibration parameters from', path_calib 
    calib = calp.CalibPars( path=path_calib, run=runnum  )
    print 'center:\n',          calib.getCalibPars('center')
    print 'tilt:\n',            calib.getCalibPars('tilt')
    print 'marg_gap_shift::\n', calib.getCalibPars('marg_gap_shift')
    print 'offset:\n',          calib.getCalibPars('offset')
    print 'offset_corr:\n',     calib.getCalibPars('offset_corr')
    print 'quad_rotation:\n',   calib.getCalibPars('quad_rotation')
    print 'quad_tilt:\n',       calib.getCalibPars('quad_tilt')

    #coord = pixcoor.CSPADPixCoords(calib, do_crop=True)
    coord = pixcoor.CSPADPixCoords(calib, do_crop=False)
    coord.print_cspad_geometry_pars()

    # Make mask of active pixels on image
    #quads = [1,1,1,1]
    quads = [1,1,1,1]

    mask_quad0_nda = np.ones((8, 185, 388)) if quads[0] else np.zeros((8, 185, 388))
    mask_quad1_nda = np.ones((8, 185, 388)) if quads[1] else np.zeros((8, 185, 388))
    mask_quad2_nda = np.ones((8, 185, 388)) if quads[2] else np.zeros((8, 185, 388))
    mask_quad3_nda = np.ones((8, 185, 388)) if quads[3] else np.zeros((8, 185, 388))
    mask_cspad_nda = np.vstack((mask_quad0_nda, mask_quad1_nda, mask_quad2_nda, mask_quad3_nda))
    mask_cspad_nda.shape = (32, 185, 388)
    mask = coord.get_cspad_image(mask_cspad_nda, config)

    #print 'cspad nda:\n', nda
    print 'cspad nda.shape: ', nda.shape

    image = coord.get_cspad_image(nda*mask_cspad_nda, config)    
    print 'image.shape =', image.shape


    #image_cropped = image[700:1050,700:1050]

    #gg.plotImageLarge(mask, amp_range=amps, figsize=(12,11))
    axis = gg.plotImageLarge(image, amp_range=amps, figsize=(12,11))
    #gg.plotImageLarge(img2d, amp_range=None, figsize=(12,11))

    angint = ai.AngularIntegrator()
    ysize, xsize = image.shape
    
    #-------- find corrections to center position
    #if True :
    if False :
        dc_list = xrange(-4,5,1)
        
        dxmax = 0
        dymax = 0
        intmax = 0
        
        print '\ndx: ', 
        for dx in dc_list :       
            print '        %2d' % dx,
        
        for dy in dc_list :
            print '\ndy: %2d ' % dy, 
            for dx in dc_list :       
                angint.setParameters(xsize, ysize, xc+dx, yc+dy, rmin, rmax, rbins, mask=mask)
                bincent, binint = angint.getRadialHistogramArrays(image)
                intval = max(binint)
                print '%10.3f' % intval,
                if intval > intmax :
                    dxmax = dx
                    dymax = dy
                    intmax = intval
        
        print '\nMaximum dx=%2d, dy=%2d, val=%.3f' % (dxmax, dymax, intmax)

    #-------- 

    gg.drawCircle(axis, (xc,yc), radius)
    gg.drawCenter(axis, (xc,yc), 40)
    gg.savefig('cspad-img.png')

    plot_img_in_polar_coords(image, (xc,yc), rad_range, phi_range=(-180,180,180), amp_range=amps_rad, rad_plot=radius)

    angint.setParameters(xsize, ysize, xc, yc, rmin, rmax, rbins, mask=mask)
    bincent, binint = angint.getRadialHistogramArrays(image)
    gg.plotGraph(bincent, binint)
    gg.savefig('cspad-img-angular-integral.png')

    gg.show()
 
#----------------------------------------------
#----------------------------------------------

if __name__ == "__main__" :

    if len(sys.argv)==1 or len(sys.argv)>2:
        print 'Use %s command with one argument - test number [0-2]' % sys.argv[0]

    elif sys.argv[1]=='0' :
        fname  = '/reg/neh/home1/dubrovin/LCLS/HDF5Analysis-v01/cspad-ndarr-ave-cxii0114-r0227.dat'
        amps   = (0,500)    
        center = (880,873)
        rad_range = (600,720,120)
        radius = 664
        amps_rad  = (0,20000)
        path_calib, runnum = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-03-19', 1
        alignment_for_cspad_ndarray(fname, amps, center, rad_range, radius, amps_rad, path_calib, runnum)

    elif sys.argv[1]=='1' :
        fname  = '/reg/neh/home1/dubrovin/LCLS/HDF5Analysis-v01/cspad-ndarr-ave-cxii0114-r0227.dat'
        amps   = (0,500)    
        center = (877.0,875.5)
        rad_range = (600,720,120)
        radius = 664
        amps_rad  = (0,20000)
        path_calib, runnum = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-03-19', 227
        #path_calib, runnum = '/reg/d/psdm/CXI/cxii0114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0', 227
        alignment_for_cspad_ndarray(fname, amps, center, rad_range, radius, amps_rad, path_calib, runnum)

    elif sys.argv[1]=='2' :
        fname     = '/reg/neh/home1/dubrovin/LCLS/HDF5Analysis-v01/cspad-ndarr-ave-cxi83714-r0136.dat'
        amps      = (0,0.5)
        center    = (882,875)
        rad_range = (100,400,300)
        #rad_range = (125,175,50)
        radius    = 146
        amps_rad  = (0,5)
        #path_calib, runnum = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2013-12-20', 1 # for v1 of offset_corr
        path_calib, runnum = '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2013-12-20', 136
        alignment_for_cspad_ndarray(fname, amps, center, rad_range, radius, amps_rad, path_calib, runnum)

    elif sys.argv[1]=='3' :
        basedir =  '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-05-15/'
        fname  = basedir + 'cspad-arr-cxid2714-r0023-lysozyme-rings.txt'
        amps   = (0,2000)    
        center = (904.0,887)
        rad_range = (50,300,250)
        radius = 117
        amps_rad  = (0,12000)
        #path_calib, runnum = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0', 23
        #path_calib, runnum = '/reg/d/psdm/CXI/cxid6214/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0', 23
        path_calib, runnum = '/reg/d/psdm/CXI/cxid9114/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0', 23
        alignment_for_cspad_ndarray(fname, amps, center, rad_range, radius, amps_rad, path_calib, runnum)

    elif sys.argv[1]=='4' :
        basedir =  '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds2-2014-05-15/'
        fname  = basedir + 'cspad-ndarr-ave-cxid9114-r0170-ds2.txt'
        amps   = (0,50)    
        center = (892.0,875)
        rad_range = (380,480,100)
        radius = 418
        amps_rad  = (0,1000)
        path_calib, runnum = basedir + 'calib/CsPad::CalibV1/CxiDs2.0:Cspad.0', 170
        #path_calib, runnum = '/reg/d/psdm/CXI/cxid9114/calib/CsPad::CalibV1/CxiDs2.0:Cspad.0', 170
        alignment_for_cspad_ndarray(fname, amps, center, rad_range, radius, amps_rad, path_calib, runnum)

    elif sys.argv[1]=='5' :
        basedir =  '/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds2-2014-05-15/'
        fname  = basedir + 'cspad-arr-cxid6014-rxxxx-powder-far-ds2.npy'
        amps   = (0,100)    
        center = (892.0,875)
        rad_range = (100,200,100)
        radius = 150
        amps_rad  = (0,1000)
        #path_calib, runnum = basedir + 'calib/CsPad::CalibV1/CxiDs2.0:Cspad.0', 50
        path_calib, runnum = '/reg/d/psdm/CXI/cxid9114/calib/CsPad::CalibV1/CxiDs2.0:Cspad.0', 50
        alignment_for_cspad_ndarray(fname, amps, center, rad_range, radius, amps_rad, path_calib, runnum)

    else :
        print 'Command argument "%s" - is not recognized as a test number...' % sys.argv[1]

    #main_example_CSpad2x2()
    sys.exit ( 'End of test.' )

#----------------------------------------------
