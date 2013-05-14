#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CSPAD2x2PixCoords...
#
#------------------------------------------------------------------------

"""
This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 2013-05-10$

@author Mikhail S. Dubrovin
"""

#--------------------------------
#  Module's version from CVS --
#--------------------------------
__version__ = "$Revision: 4 $"
# $Source$
#--------------------------------
import sys
import numpy as np

import PyCSPadImage.CSPAD2x2PixCoords as pixcoor
import PyCSPadImage.CSPAD2x2CalibPars as calpars

import PyCSPadImage.HDF5Methods       as hm 
import PyCSPadImage.GlobalGraphics    as gg
#------------------------------

def test_cspad2x2_calib_geometry() :
    """Test method, demonstrates how to work with CSPAD2x2CalibPars and CSPAD2x2PixCoords modules
    """    
    #======= Define input parameters
    Ndet = 5
    run  = 180
    path = '/reg/d/psdm/mec/mec73313/calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.%1d/' % Ndet
    #path = '/reg/neh/home1/dubrovin/LCLS/CSPad2x2Alignment/calib-cspad2x2-0%1d-2013-02-13/' % Ndet
    #fname  = '/reg/d/psdm/mec/mec73313/hdf5/mec73313-r%04d.h5' % run
    fname  = '/reg/neh/home1/dubrovin/LCLS/HDF5Analysis-v01/PyCSPadImage/src/mec73313-r%04d.h5' % run
    dsname = '/Configure:0000/Run:0000/CalibCycle:0000/CsPad2x2::ElementV1/MecTargetChamber.0:Cspad2x2.%1d/data' % Ndet
    list_of_clib_types = ['center', 'tilt', 'pedestals']

    #======= Get calibration object
    calib = calpars.CSPAD2x2CalibPars(path, run, list_of_clib_types)

    #======= Get CSPAD2x2 pixel coordinate arrays, shaped as (2, 185, 388)
    coord = pixcoor.CSPAD2x2PixCoords(calib)
    X,Y = coord.get_cspad2x2_pix_coordinate_arrays_pix()

    #======= Get CSPAD2x2 pedestals array, shaped as (185, 388, 2)
    peds_arr = calib.getCalibPars('pedestals')

    #======= Get data array from hdf5 dataset, shaped as (185, 388, 2)
    data_arr = hm.getDataSetForOneEvent(fname, dsname, event=0) - peds_arr
    
    #======= Convert shape from (185, 388, 2) to (2, 185, 388)    
    ord_arr  = calpars.data2x2ToTwo2x1(data_arr)

    #======= Compose and plot CSPAD2x2 image from coordinate and intensity arrays
    img2d = pixcoor.getImage(X,Y,ord_arr)

    #======= Print for test purpose 
    calib.printCalibParsStatus()
    #print 'pedestals:\n', calib.getCalibPars('pedestals')
    print 'center:\n',    calib.getCalibPars('center')
    print 'tilt:\n',      calib.getCalibPars('tilt')
    print 'peds_arr.shape:', peds_arr.shape  # = (185, 388, 2)  
    print 'Get data array from file: ' + fname
    print 'data_arr.shape:', data_arr.shape
    print 'ord_arr.shape:', ord_arr.shape
    print 'img2d.shape:', img2d.shape

    #======= Plot image and spectrum
    my_range = (-10,40) # None
    gg.plotImageLarge(img2d, range=my_range)        
    gg.plotSpectrum(img2d, range=my_range)
    gg.show()

#------------------------------

if __name__ == "__main__" :
    test_cspad2x2_calib_geometry()
    sys.exit ( 'End of test.' )

#------------------------------
