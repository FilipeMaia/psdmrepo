#!/usr/bin/env python
#--------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py

#--------------------
# Define graphical methods

def plot_image (arr, img_range=None, zrange=None) :    # range = (left, right, low, high), zrange=(zmin,zmax)
    fig = plt.figure(num=1, figsize=(12,12), dpi=80, facecolor='w', edgecolor='w', frameon=True)
    fig.subplots_adjust(left=0.05, bottom=0.03, right=0.98, top=0.98, wspace=0.2, hspace=0.1)
    figAxes = fig.add_subplot(111)
    #figAxes = fig.add_axes([0.15, 0.06, 0.78, 0.21])
    imAxes = figAxes.imshow(arr, origin='upper', interpolation='nearest', aspect='auto', extent=img_range)
    if zrange != None : imAxes.set_clim(zrange[0],zrange[1])
    colbar = fig.colorbar(imAxes, pad=0.03, fraction=0.04, shrink=1.0, aspect=40, orientation=1)

def plot_histogram(arr, amp_range=None, figsize=(6,6), bins=40) :
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    plt.hist(arr.flatten(), bins=bins, range=amp_range)
    #fig.canvas.manager.window.move(500,10)
    
#--------------------

def get_array_from_file(fname, dtype=np.float32) :
    print 'get_array_from_text_file:', fname
    return np.loadtxt(fname, dtype=dtype)

def get_array_from_bin_file(fname, dtype=np.float32) :
    print 'get_array_from_bin_file:', fname
    return np.fromfile(fname, dtype)

def get_numpy_array_from_file(fname) :
    print 'get_numpy_array_from_file:', fname
    return np.load(fname)

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

def get_array(fname) :

    fname, ampRange = get_input_parameters()

    if os.path.splitext(fname)[1] == '.txt' :
        return get_array_from_file(fname) 

    if os.path.splitext(fname)[1] == '.bin' :

        arr = get_array_from_bin_file(fname, dtype=np.uint16)
        if arr.shape[0] == 1300*1340 :
            arr.shape=(1300,1340)
            return arr

        arr = get_array_from_bin_file(fname, dtype=np.float)
        if arr.shape[0] == 400*400 :
            arr.shape=(400,400)
            return arr

        else :
            print "WARNING !!! Binary file array shape: %s, unknown case of reshaping to 2d..." % arr.shape

    return np.zeros((100,100))

#--------------------

def do_main() :

    fname, ampRange = get_input_parameters()

    arr = get_array(fname)

    #arr = get_array_from_file('/reg/neh/home1/sikorski/xcs_pyana_current/e167-r0020-s00-c00/2013-04-03-09-43-06-939033/e167-r0020-s00-c00_dark_img.txt'
    #arr = get_array_from_file('z-comparison/e167-r0020-s00-c00-dark-img-from-marcin.txt') 
    #arr -= get_array_from_file('z-comparison/t1-xcsi0112-r0020-peds-ave.txt') 

    print 'arr:\n', arr
    print 'arr.shape=', arr.shape

    plot_image(arr, zrange=ampRange)
    #plt.get_current_fig_manager().window.move(10,10)       # works for GTk
    plt.get_current_fig_manager().window.geometry("+10+10") # works for Tk 

    plot_histogram(arr, amp_range=ampRange)
    plt.get_current_fig_manager().window.geometry("+950+10")

    plt.show()

#--------------------

if __name__ == '__main__' :
    do_main()
    sys.exit('The End')

#--------------------
