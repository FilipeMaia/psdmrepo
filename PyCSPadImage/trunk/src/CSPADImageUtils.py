
#------------------------------

import sys
#import numpy as np

#from   PyCSPadImage.CSPAD2x2CalibPars import *
#from   PyCSPadImage.CSPAD2x2PixCoords import *
from   PyCSPadImage.CSPadArrayImage   import *
#import pyimgalgos.GlobalGraphics    as gg
#from Logger                 import logger

#------------------------------

#def test_plot_for_cspad_coords(calib_path='.', run_num=0) :
    #calib_path  = '/reg/d/psdm/xpp/xpptut13/calib/CsPad::CalibV1/XppGon.0:Cspad.1/'
    #run_num   = 123

    #calib = CSPADCalibPars(calib_path, run_num)
    #coord = CSPADPixCoords(calib)
#    coord = CSPADPixCoords()
    #coord.print_cspad_geometry_pars()
#    iX,iY = coord.get_cspad_pix_coordinate_arrays_shapeed_as_data_pix ()

#    raw_arr = np.arange(185*388*2) # np.zeros((sp.sects,sp.rows,sp.cols)
#    raw_arr.shape = (185,388,2)

#    img2d = gg.getImageFromIndexArrays(iX,iY,raw_arr)

#    gg.plotImageLarge(img2d, amp_range=(-1, 185*388*2), figsize=(12,11))
#    gg.show()

#------------------------------

def get_cspad_raw_data_array_image(arr) :

    if arr.shape != (4*8*185,388) :
        msg = 'Non-expected array shape for cspad:', arr.shape
        print msg
        #logger.info(msg, 'CSPADImage')
        return None
    
    return getCSPadArrayAs2DImage(arr,3,6,6)

#------------------------------
 
if __name__ == "__main__" :
    #if len(sys.argv)==1   : print 'Use command: python', sys.argv[0], '<test-number=0-5>'

    #test_plot_for_cspad2x2_coords()


    sys.exit ( 'End of test.' )

#------------------------------
