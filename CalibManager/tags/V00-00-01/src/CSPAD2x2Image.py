
#------------------------------

import sys
import numpy as np

from   PyCSPadImage.CSPAD2x2CalibPars import *
from   PyCSPadImage.CSPAD2x2PixCoords import *
import PyCSPadImage.GlobalGraphics    as gg

#------------------------------

def test_plot_for_cspad2x2_coords(calib_path='.', run_num=0) :
    #calib_path  = '/reg/d/psdm/xpp/xpptut13/calib/CsPad2x2::CalibV1/XppGon.0:Cspad2x2.1/'
    #run_num   = 123

    #calib = CSPAD2x2CalibPars(calib_path, run_num)
    #coord = CSPAD2x2PixCoords(calib)
    coord = CSPAD2x2PixCoords()
    #coord.print_cspad2x2_geometry_pars()
    iX,iY = coord.get_cspad2x2_pix_coordinate_arrays_shapeed_as_data_pix ()

    raw_arr = np.arange(185*388*2) # np.zeros((sp.sects,sp.rows,sp.cols)
    raw_arr.shape = (185,388,2)

    img2d = gg.getImageFromIndexArrays(iX,iY,raw_arr)

    gg.plotImageLarge(img2d, amp_range=(-1, 185*388*2), figsize=(12,11))
    gg.show()

#------------------------------

def get_cspad2x2_non_corrected_image_for_raw_data_array(arr) :
    if arr.shape != (185,388,2) :
        msg = 'Non-expected array shape for cspad2x2: %s' % arr.shape
        print msg
        return None
    
    coord = CSPAD2x2PixCoords()
    iX,iY = coord.get_cspad2x2_pix_coordinate_arrays_shapeed_as_data_pix ()
    
    img2d = gg.getImageFromIndexArrays(iX,iY,arr)
    return img2d

#------------------------------
 
if __name__ == "__main__" :
    #if len(sys.argv)==1   : print 'Use command: python', sys.argv[0], '<test-number=0-5>'

    test_plot_for_cspad2x2_coords()
    sys.exit ( 'End of test.' )

#------------------------------
