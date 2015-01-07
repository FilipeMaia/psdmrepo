
#------------------------------

import sys
import numpy as np

from   PyCSPadImage.CSPAD2x2CalibPars import *
from   PyCSPadImage.CSPAD2x2PixCoords import *
import pyimgalgos.GlobalGraphics as gg

#------------------------------

def getImageArrayForCSpad2x1Segment(arr2x1):
    """Returns the image array for pair of ASICs"""

    #arr2x1 = arr1ev[0:185,0:388,pair]
    #arr2x1 = arr1ev[:,:,pair]
    #print '2x1 array shape:', arr2x1.shape
      
    asics  = np.hsplit(arr2x1,2)
    arrgap = np.zeros ((185,3), dtype=np.float32)
    arr2d  = np.hstack((asics[0],arrgap,asics[1]))
    return arr2d

#---------------------

def getImageArrayForCSpad2x2FromSegments(arrseg0, arrseg1):
    """Returns the image array for the CSpad2x2Element or CSpad2x2"""       

    arr2x1Pair0 = getImageArrayForCSpad2x1Segment(arrseg0) # arr1ev[:,:,0])
    arr2x1Pair1 = getImageArrayForCSpad2x1Segment(arrseg1) # arr1ev[:,:,1])
    wid2x1      = arr2x1Pair0.shape[0]
    len2x1      = arr2x1Pair0.shape[1]

    arrgapV = np.zeros( (20,len2x1), dtype=np.float ) # dtype=np.int16 
    arr2d   = np.vstack((arr2x1Pair0, arrgapV, arr2x1Pair1))

    #print 'arr2d.shape=', arr2d.shape
    #print 'arr2d=',       arr2d
    return arr2d

#------------------------------

def get_cspad2x2_non_corrected_image_for_raw_data_array(arr) :
    if arr.shape[-1] == 388 :
        arr.shape = (2,185,388)
        #print 'Shaped in natural order:', arr.shape
        return getImageArrayForCSpad2x2FromSegments(arr[0,:,:], arr[1,:,:])

    if arr.shape[-1] == 2 or arr.size == 185*388*2 :
        arr.shape = (185,388,2)
        #print 'Shaped as data:', arr.shape
        return getImageArrayForCSpad2x2FromSegments(arr[:,:,0], arr[:,:,1])

        #coord = CSPAD2x2PixCoords()
        #iX,iY = coord.get_cspad2x2_pix_coordinate_arrays_shapeed_as_data_pix ()    
        #return gg.getImageFromIndexArrays(iX,iY,arr)

    msg = 'Un-expected cspad2x2 array shape %s or size %s' % (str(arr.shape), arr.size)
    print msg
    return None
    
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
 
if __name__ == "__main__" :
    #if len(sys.argv)==1   : print 'Use command: python', sys.argv[0], '<test-number=0-5>'

    test_plot_for_cspad2x2_coords()
    sys.exit ( 'End of test.' )

#------------------------------
