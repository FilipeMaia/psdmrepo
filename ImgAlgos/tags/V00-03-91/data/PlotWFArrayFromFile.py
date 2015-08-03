#!/usr/bin/env python
#--------------------

import numpy as np
import matplotlib.pyplot as plt
#import scipy.misc as scim
import Image

import os
import sys
import h5py

#--------------------
# Define graphical methods

def plot_graphs(arr, trange=None, figsize=(10,5)) :
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    plt.xlabel('T-scale')
    plt.ylabel('V-scale')


    n_chan, n_samp = arr.shape

    list_of_lines = ['r-', 'g-', 'b-', 'y-', 'k-', \
                     'r-', 'g-', 'b-', 'y-', 'k-', \
                     'r-', 'g-', 'b-', 'y-', 'k-', \
                     'r-', 'g-', 'b-', 'y-', 'k-']

    list_of_pars = []

    #plt.plot(t, arr[0,:], 'r-', \
    #         t, arr[1,:], 'k-')

    if trange is None :
        t = np.arange(0,n_samp)
        for chan in range( n_chan ) :
            list_of_pars.append( t )
            list_of_pars.append( arr[chan,:] )
            list_of_pars.append( list_of_lines[chan] )
    else :
        t = np.arange(trange[0],trange[1])
        for chan in range( n_chan ) :
            list_of_pars.append( t )
            list_of_pars.append( arr[chan,trange[0]:trange[1]] )
            list_of_pars.append( list_of_lines[chan] )

    plt.plot(*list_of_pars)
    #plt.axis([0, 6, 0, 20])
    plt.get_current_fig_manager().window.geometry("+950+10")
    
#--------------------

def get_array_from_file(fname, dtype=np.float32) :
    print 'get_array_from_text_file:', fname
    arr = np.loadtxt(fname, dtype=dtype)
    print 'arr.shape=', arr.shape
    return arr

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
    print 'sys.argv: ', sys.argv
    print 'nargs: ', nargs

    if nargs == 1 :
        print 'Will use all default parameters\n',\
              'Expected command: ' + sys.argv[0] + ' <infname> <Tmin> <Tmax>' 
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

    #fname, ampRange = get_input_parameters()

    if os.path.splitext(fname)[1] == '.txt' \
    or os.path.splitext(fname)[1] == '.dat' :
        return get_array_from_file(fname) 

    elif os.path.splitext(fname)[1] == '.npy' :
        return get_numpy_array_from_file(fname) 

    elif os.path.splitext(fname)[1] == '.tiff' \
    or   os.path.splitext(fname)[1] == '.TIFF' :
        arr = scim.imread(fname, flatten=True) #, dtype=np.uint16)
        #img = Image.open(fname)
        #arr = np.array(img.getdata()).reshape(img.size[::-1])
        #arr.shape=(400,400)
        return arr


    return np.zeros((100,100))

#--------------------

def do_main() :

    fname, trange = get_input_parameters()

    arr = get_array(fname)

    print 'arr:\n', arr
    print 'arr.shape=', arr.shape

    #plot_image(arr, zrange=ampRange)
    #plt.get_current_fig_manager().window.move(10,10)       # works for GTk
    #plt.get_current_fig_manager().window.geometry("+10+10") # works for Tk 
    #plt.savefig('camera-img.png')

    plot_graphs(arr, trange)
    plt.get_current_fig_manager().window.geometry("+950+10")
    plt.savefig('wforms.png')

    plt.show()

#--------------------

if __name__ == '__main__' :
    do_main()
    sys.exit('The End')

#--------------------
