#!/usr/bin/env python
#--------------------

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# FOR ALIGNED CSPAD IMAGE:
from PyCSPadImage import CalibPars          as calp
from PyCSPadImage import CalibParsEvaluated as cpe
from PyCSPadImage import CSPadImageProducer as cip

import GlobalGraphics as gg

#--------------------

def getCSPadArrayWithGap(arr, gap=3) :
    #print 'getCSPadArrayWithGap(...): Input array shape =', arr.shape
    nrows,ncols = arr.shape # (32*185,388) # <== expected input array shape
    if ncols != 388 or nrows<185 :
        print 'getCSPadArrayWithGap(...): WARNING! UNEXPECTED INPUT ARRAY SHAPE =', arr.shape
        return arr
    arr_gap = np.zeros( (nrows,gap), dtype=np.int16 )
    arr_halfs = np.hsplit(arr,2)
    arr_with_gap = np.hstack((arr_halfs[0], arr_gap, arr_halfs[1]))
    arr_with_gap.shape = (nrows,ncols+gap)
    return arr_with_gap


def getCSPadSegments2D(arr) :
    gap   = 3
    space = 10
    arr_all = getCSPadArrayWithGap(arr, gap)
    arr_all.shape = (4,8*185,388+gap) # Reshape for quad index
    arr_sp = np.zeros( (8*185,space), dtype=np.int16 )
    return np.hstack((arr_all[0,:],arr_sp,arr_all[1,:],arr_sp,arr_all[2,:],arr_sp,arr_all[3,:]))


def getQuad2D(arr_all,quad=0) :
    arr_quad_img = np.zeros( (825,825), dtype=np.float32 )
    quadInDetOriInd, pairInQaudOriInd, pairXInQaud, pairYInQaud = getCSPadGeometry()

    for segm in range(8) :
        arr_segm = arr_all[quad,segm,:]    
        arr_segm_rot = np.rot90(arr_segm,pairInQaudOriInd[quad][segm])

        nrows, ncols = arr_segm_rot.shape
        print 'nrows, ncols = ', nrows, ncols

        xOff = pairXInQaud[quad][segm] - nrows/2
        yOff = pairYInQaud[quad][segm] - ncols/2

        arr_quad_img[xOff:nrows+xOff, yOff:ncols+yOff] += arr_segm_rot[0:nrows, 0:ncols]
    return  arr_quad_img


def getQuadImage(arr,quad=0) :
    gap=3
    arr_all       = getCSPadArrayWithGap(arr, gap)
    arr_all.shape = (4,8,185,388+gap) # Reshape for quad and segment indexes
    print 'arr_all.shape =', arr_all.shape
    return getQuad2D(arr_all,quad)

#--------------------

def getCSPadImage(arr) :
    gap=3
    arr_all       = getCSPadArrayWithGap(arr, gap)
    arr_all.shape = (4,8,185,388+gap) # Reshape for quad and segment indexes
    #arr_cspad_img = np.zeros( (1765,1765), dtype=np.float32 )
    arr_cspad_img = np.zeros( (1697,1696), dtype=np.float32 )

    segmX, segmY, segmZ, segmRotInd = getCSPadGeometryShort()
    for quad in range(4) :
        for segm in range(8) :
            arr_segm = arr_all[quad,segm,:]    
            arr_segm_rot = np.rot90(arr_segm,segmRotInd[quad][segm])
        
            nrows, ncols = arr_segm_rot.shape
            #print 'nrows, ncols = ', nrows, ncols
        
            xOff = segmX[quad][segm] - nrows/2 + 41
            yOff = segmY[quad][segm] - ncols/2 + 2 
        
            arr_cspad_img[xOff:nrows+xOff, yOff:ncols+yOff] += arr_segm_rot[0:nrows, 0:ncols]
    return  arr_cspad_img

#--------------------

def getCSPadGeometry() :
    quadInDetOriInd  = [   2,    1,    0,    3]

    pairInQaudOriInd = [[   3,   3,   2,   2,   1,   1,   2,   2],
                        [   3,   3,   2,   2,   1,   1,   2,   2],
                        [   3,   3,   2,   2,   1,   1,   2,   2],
                        [   3,   3,   2,   2,   1,   1,   2,   2]]

    pairXInQaud = [[200,  200,  310,   95,  625,  625,  710,  500],
                   [200,  200,  310,   95,  625,  625,  710,  500],
                   [200,  200,  310,   95,  625,  625,  710,  500],
                   [200,  200,  310,   95,  625,  625,  710,  500]]
    
    pairYInQaud = [[310,   95,  625,  625,  515,  730,  200,  200],
                   [310,   95,  625,  625,  515,  730,  200,  200],
                   [310,   95,  625,  625,  515,  730,  200,  200],
                   [310,   95,  625,  625,  515,  730,  200,  200]]

    return quadInDetOriInd, pairInQaudOriInd, pairXInQaud, pairYInQaud

#--------------------

def getCSPadGeometryShort() :

    segmX = [[ 473.38,  685.26,  155.01,  154.08,  266.81,   53.95,  583.04,  582.15],  
             [ 989.30,  987.12, 1096.93,  884.11, 1413.16, 1414.94, 1500.83, 1288.02],  
             [1142.59,  930.23, 1459.44, 1460.67, 1347.57, 1559.93, 1032.27, 1033.44],  
             [ 626.78,  627.42,  516.03,  729.15,  198.28,  198.01,  115.31,  327.66]]  

    segmY = [[1028.07, 1026.28, 1139.46,  926.91, 1456.78, 1457.35, 1539.71, 1327.89],  
             [1180.51,  967.36, 1497.74, 1498.54, 1385.08, 1598.19, 1069.65, 1069.93],  
             [ 664.89,  666.83,  553.60,  765.91,  237.53,  236.06,  152.17,  365.47],  
             [ 510.38,  722.95,  193.33,  193.41,  308.04,   95.25,  625.28,  624.14]]  

    segmZ = [[  -0.27,    0.03,   -0.64,   -0.42,   -0.93,   -1.16,   -0.66,   -0.41],  
             [   0.22,    0.45,   -0.16,   -0.39,    0.44,    0.03,    0.83,   -0.27],  
             [   0.90,    0.65,    1.26,    1.22,    1.27,    1.52,    0.73,    0.88],  
             [   0.32,    0.23,    0.31,    0.42,   -0.08,    0.04,   -0.36,   -0.09]]

    segmRotInd = [[   0,   0,   3,   3,   2,   2,   3,   3],
                  [   3,   3,   2,   2,   1,   1,   2,   2],
                  [   2,   2,   1,   1,   0,   0,   1,   1],
                  [   1,   1,   0,   0,   3,   3,   0,   0]]

    return segmX, segmY, segmZ, segmRotInd 

#--------------------

def get_array_from_file(fname) :
    print 'get_array_from_file:', fname
    base, ext = os.path.splitext(fname)
    if ext == '.npy' : return np.load(fname)
    else             : return np.loadtxt(fname, dtype=np.float32)

#--------------------

def getCSPadImageAligned(arr_raw, path_calib, runnum) :

    #print 'Load calibration parameters from', path_calib 
    #calp.calibpars.setCalibParsForPath ( run=runnum, path=path_calib )
    #cpe.cpeval.printCalibParsEvaluatedAll() 

    #calib = calp.CalibPars(path=path_calib, run=runnum) #, list_of_clib_types)

    print 'Make the CSPad image from raw array'
    #cspadimg = cip.CSPadImageProducer(calib, rotation=1, tiltIsOn=True)#, mirror=True)
    #return cspadimg.getCSPadImage( arr_raw )

#--------------------

def get_input_parameters() :

    fname_def = 'cspad-pedestals.dat'
    #fname_def = '/reg/neh/home/dubrovin/LCLS/HDF5Explorer-v01/camera-ave-CxiDg1.0:Tm6740.0.txt'
    #fname_def = '/reg/neh/home/dubrovin/LCLS/PSANA-V00/cspad-cxi49012-r0027-pedestals-def-rms.dat'
    #fname_def = '/reg/neh/home/dubrovin/LCLS/PSANA-V00/cspad-cxi49012-r0027-pedestals-rms.dat'
    #fname_def = 'cspad-noise.dat'

    Amin_def = None
    Amax_def = None

    nargs = len(sys.argv)
    print 'sys.argv[0]: ', sys.argv[0]
    print 'nargs: ', nargs

    if nargs == 1 :
        print 'Will use all default parameters\n',\
              'Expected command: ' + sys.argv[0] + ' <infname> <Amin> <Amax>' 
        sys.exit('CHECK INPUT PARAMETERS!')

    if nargs  > 1 : fname = sys.argv[1]
    else          : fname = fname_def

    if nargs  > 2 : Amin = float(sys.argv[2])
    else          : Amin = Amin_def

    if nargs  > 3 : Amax = float(sys.argv[3])
    else          : Amax = Amax_def

    if nargs  > 4 :         
        print 'WARNING: Too many input arguments! Exit program.\n'
        sys.exit('CHECK INPUT PARAMETERS!')

    ampRange = (Amin, Amax)
    if ampRange[0]==None or ampRange[1]==None : ampRange = None

    print 'Input file name  :', fname
    print 'ampRange         :', ampRange
 
    return fname,ampRange 

#--------------------

def do_main() :

    fname, ampRange = get_input_parameters()

    arr_raw = get_array_from_file(fname)
    print 'arr_raw.shape=', arr_raw.shape
    #print 'arr_raw=\n', arr_raw

    arr_segs = getCSPadSegments2D(arr_raw)
    print 'arr_segs.shape=', arr_segs.shape
    
    arr = getCSPadImage(arr_raw) # GET IMAGE WITHOUT ALIGNMENT !!!
    #arr = getQuadImage(arr_raw,quad=1)

    #------------------------------------- New stuff
    #runnum = 150
    #path_calib = '/reg/d/psdm/CXI/cxi49012/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/'
    #arr = getCSPadImageAligned(arr_raw, path_calib, runnum)

    #------------------------------------- New stuff

    # Plot 1
    gg.plot_image(arr_segs, zrange=ampRange)
    title = ''
    for q in range(4) : title += ('Quad %d'%(q) + 20*' ')  
    plt.title(title,color='b',fontsize=20)
    plt.get_current_fig_manager().window.geometry("+10+10")
    plt.savefig('cspad-arr.png')


    # Plot 2
    gg.plot_image(arr, zrange=ampRange)
    plt.get_current_fig_manager().window.geometry("+450+10")
    plt.savefig('cspad-img.png')


    # Plot 3
    gg.plot_histogram(arr_raw, amp_range=ampRange)
    plt.get_current_fig_manager().window.geometry("+950+10")
    plt.savefig('cspad-spe.png')

    plt.show()

    #gg.saveHRImageInFile(arr_segs,ampRange,fname='cspad-arr-hr.png') 
    #gg.saveHRImageInFile(arr, ampRange, fname='cspad-img-hr.png') 


#--------------------

if __name__ == '__main__' :

    do_main()

    sys.exit('The End')

#--------------------
