#!/usr/bin/env python
#--------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # for patches.Circle
import sys

# FOR ALIGNED CSPAD IMAGE:
from PyCSPadImage import CalibPars          as calp
from PyCSPadImage import CalibParsEvaluated as cpe
from PyCSPadImage import CSPadImageProducer as cip

#--------------------

class Storage :
    def __init__(self) :
        print 'Storage object is created'

    def printStorage(self) :
        print 'Object of class Storage'

#--------------------
# Define graphical methods

def plot_image (arr, img_range=None, zrange=None, title='',figsize=(12,12), dpi=80, store=None) :
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='w',edgecolor='w', frameon=True)
    fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)
    store.figAxes = figAxes = fig.add_subplot(111)
    store.imAxes  = imAxes  = figAxes.imshow(arr, origin='upper', interpolation='nearest', aspect='auto', extent=img_range)
    if zrange != None : imAxes.set_clim(zrange[0],zrange[1])
    colbar = fig.colorbar(imAxes, pad=0.03, fraction=0.04, shrink=1.0, aspect=40, orientation='horizontal')
    fig.canvas.set_window_title(title)

def plot_histogram(arr, amp_range=None, figsize=(5,5)) :
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    plt.hist(arr.flatten(), bins=100, range=amp_range)
    #fig.canvas.manager.window.move(500,10)

def saveHRImageInFile(arr, ampRange=None, fname='cspad-arr-hr.png', figsize=(12,12), dpi=300, store=None) :
    print 'SAVE HIGH RESOLUTION IMAGE IN FILE', fname
    plot_image(arr, zrange=ampRange, figsize=figsize, dpi=dpi, store=store)
    title = ''
    for q in range(4) : title += ('Quad %d'%(q) + 20*' ')  
    plt.title(title, color='b', fontsize=20)
    plt.savefig(fname,dpi=dpi)
    #plt.imsave('test.png', format='png',dpi=300)

def plot_peaks_for_arr (arr_peaks, store=None, color='w') :  
    axes = store.figAxes
    #ampmax = np.average(arr_peaks,axis=0)[8]
    ampmax = np.max(arr_peaks,axis=0)[8]
    print 'ampmax=', ampmax

    # arr_peaks : 0  0  26.7143  381.881  5  0.672354  0.570576  152  371
    for peak in arr_peaks :
        q, s, r, c, npix, sig_r, sig_c, amp_max, amp_tot = peak
        #print 'q, s, r, c, npix, amp_tot =',q, s, r, c, npix, amp_tot
        x=q*388 + c
        y=s*185 + r
        xy0 = (x,y)
        #r0  = 10*amp_tot/ampave
        r0  = 4+4*amp_tot/ampmax
        circ = patches.Circle(xy0, radius=r0, linewidth=2, color=color, fill=False)
        axes.add_artist(circ)

def plot_peaks_for_img (arr_peaks, store=None, color='w') :  
    axes = store.figAxes
    #ampave = np.average(arr_peaks,axis=0)[8]
    ampmax = np.max(arr_peaks,axis=0)[8]
    #print 'ampmax=', ampmax

    xpix, ypix = cpe.cpeval.getCSPadPixCoordinates_pix()

    #print " xpix:\n", xpix
    print " xpix.shape:\n", xpix.shape

    # arr_peaks : 0  0  26.7143  381.881  5  0.672354  0.570576  152  371
    for peak in arr_peaks :
        q, s, r, c, npix, sig_r, sig_c, amp_max, amp_tot = peak
        #print 'q, s, r, c, npix, amp_tot=',q, s, r, c, npix, amp_tot
        #x=q*388 + c
        #y=s*185 + r
        x=xpix[q,s,int(r),int(c)]
        y=ypix[q,s,int(r),int(c)]
        #print 'q, s, r, c, npix, amp_tot=',q, s, r, c, npix, amp_tot, ' x,y=',x,y        
        xy0 = (x,y)
        r0  = 4+4*amp_tot/ampmax
        circ = patches.Circle(xy0, radius=r0, linewidth=2, color=color, fill=False)
        axes.add_artist(circ)

def print_peaks (arr_peaks) :  
    # arr_peaks : 0  0  26.7143  381.881  5  0.672354  0.570576  152  371
    for peak in arr_peaks :
        q, s, r, c, npix, sig_r, sig_c, amp_max, amp_tot = peak
        print 'q, s, r, c, npix, amp_max, amp_tot, sig_r, sig_c =',\
               q, s, r, c, npix, amp_max, amp_tot, sig_r, sig_c

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


def getCSPadSegments2D(arr, gap=3, space=10) :
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


def getQuadImage(arr, quad=0, gap=3) :
    arr_all       = getCSPadArrayWithGap(arr, gap)
    arr_all.shape = (4,8,185,388+gap) # Reshape for quad and segment indexes
    print 'arr_all.shape =', arr_all.shape
    return getQuad2D(arr_all,quad)

#--------------------

def getCSPadImage(arr, gap=3) :
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
    return np.loadtxt(fname, dtype=np.float32)

#--------------------

def getCSPadImageAligned(arr_raw, path_calib, runnum) :

    #print 'Load calibration parameters from', path_calib 
    calp.calibpars.setCalibParsForPath ( run=runnum, path=path_calib )
    cpe.cpeval.evaluateCSPadPixCoordinates (rotation=1) #, mirror=True) # for pixel coordinates
    cpe.cpeval.printCalibParsEvaluatedAll() 

    #print 'Make the CSPad image from raw array'
    cspadimg = cip.CSPadImageProducer(rotation=1, tiltIsOn=True) #, mirror=True)
    return cspadimg.getCSPadImage( arr_raw )

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
    print 'N arguments: ', nargs

    if nargs == 1 :
        print 'Will use all default parameters\n',\
              'Expected command: ' + sys.argv[0] + ' <infname> <Amin> <Amax>' 
        sys.exit('CHECK INPUT PARAMETERS!')

    if nargs  > 1 : fname = sys.argv[1]
    else          : fname = fname_def

    if nargs  > 2 : Amin = int(sys.argv[2])
    else          : Amin = Amin_def

    if nargs  > 3 : Amax = int(sys.argv[3])
    else          : Amax = Amax_def

    if nargs  > 4 :         
        print 'WARNING: Too many input arguments! Exit program.\n'
        sys.exit('CHECK INPUT PARAMETERS!')

    ampRange = (Amin, Amax)

    print 'Input file name:', fname
    print 'ampRange       :', ampRange
    if ampRange[0]==None or ampRange[1]==None : ampRange = None
 
    return fname,ampRange 

#--------------------

def do_main() :

    fname, ampRange = get_input_parameters()

    # Get peaks form file
    splitfname, splitfext = fname.rsplit('.',1)
    fname_peaks = splitfname + '-peaks.' + splitfext
    print 'Peaks file name:', fname_peaks
    arr_peaks = np.loadtxt(fname_peaks, dtype=np.double)
    #print 'Array of peaks:\n', arr_peaks

    s = Storage()

    arr_raw = get_array_from_file(fname)
    print 'arr_raw.shape=\n', arr_raw.shape
    #print 'arr_raw=\n', arr_raw

    arr_segs = getCSPadSegments2D(arr_raw, gap=0, space=0)
    
    #------------------------------------- New stuff
    runnum = 150
    path_calib = '/reg/d/psdm/CXI/cxi49012/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/'
    arr = getCSPadImageAligned(arr_raw, path_calib, runnum)
    #------------------------------------- 
    #arr = getCSPadImage(arr_raw)
    #arr = getQuadImage(arr_raw,quad=1)

    # Plot 1
    plot_image(arr_segs, zrange=ampRange, store=s)
    title = ''
    for q in range(4) : title += ('Quad %d'%(q) + 20*' ')  
    plt.title(title,color='b',fontsize=20)
    plt.get_current_fig_manager().window.geometry("+10+10")
    plot_peaks_for_arr(arr_peaks, store=s)
    #print_peaks(arr_peaks)
    plt.savefig('cspad-arr.png')


    # Plot 3
    plot_histogram(arr, amp_range=ampRange)
    plt.get_current_fig_manager().window.geometry("+950+10")
    plt.savefig('cspad-spe.png')

    #plt.show()
    #sys.exit('Test exit A')

    # Plot 2
    plot_image(arr, zrange=ampRange, store=s)
    plt.get_current_fig_manager().window.geometry("+450+10")
    plot_peaks_for_img(arr_peaks, store=s)
    plt.savefig('cspad-img.png')

    plt.show()

    #saveHRImageInFile(arr_segs,ampRange,fname='cspad-arr-hr.png') 
    #saveHRImageInFile(arr,ampRange,fname='cspad-img-hr.png', store=s) 


#--------------------

if __name__ == '__main__' :

    do_main()

    sys.exit('The End')

#--------------------
