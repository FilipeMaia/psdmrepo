#!/usr/bin/env python
#--------------------

import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

import GlobalGraphics as gg

#--------------------
# Define graphical methods

def get_array_from_file(fname) :
    print 'get_array_from_file:', fname
    return np.loadtxt(fname, dtype=np.float32)

#---------------------

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

#---------------------

def getImageArrayForCSpad2x2(arr1ev):
    """Returns the image array for the CSPAD2x2 array as (185,388,2) or (2,185,388)"""       

    if (arr1ev.shape[-1] == 2) :
        arr1ev.shape = (185,388,2)
        print 'Shaped as data:', arr1ev.shape
        return getImageArrayForCSpad2x2FromSegments(arr1ev[:,:,0], arr1ev[:,:,1])

    else :
        arr1ev.shape = (2,185,388)
        print 'Shaped as in natural order:', arr1ev.shape
        return getImageArrayForCSpad2x2FromSegments(arr1ev[0,:,:], arr1ev[1,:,:])

#--------------------

def get_input_parameters() :

    fname_def = './image0_ev000115.txt'
    Amin_def  = None
    Amax_def  = None

    nargs = len(sys.argv)
    print 'sys.argv[0]: ', sys.argv[0]
    print 'nargs: ', nargs

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
    if ampRange[0]==None or ampRange[1]==None : ampRange = None

    print 'Input file name  :', fname
    print 'ampRange         :', ampRange
 
    return fname, ampRange 

#--------------------

#--------------------

def do_main() :

    fname, ampRange = get_input_parameters()
    arr1ev = get_array_from_file(fname)    
    if (arr1ev.size != 185*388*2) :
        msg = 'Input array size %s is not consistent with CSPAD2x2 (185*388*2)' % arr1ev.size
        sys.exit(msg)
    print 'arr:\n', arr1ev
    print 'arr1ev.shape=', arr1ev.shape

    arr = getImageArrayForCSpad2x2(arr1ev)
    
    print 'Image arr.shape=', arr.shape

    gg.plot_image(arr, zrange=ampRange)
    plt.get_current_fig_manager().window.geometry('+10+10') # move(10,10)
    plt.savefig('cspad2x2-img.png')

    print 'Histogram contains pixels only!'
    gg.plot_histogram(arr1ev, ampRange)
    plt.get_current_fig_manager().window.geometry('+950+10') # .move(950,10)
    plt.savefig('cspad2x2-spe.png')

    plt.show()

#--------------------

if __name__ == '__main__' :
    do_main()
    sys.exit('The End')

#--------------------
