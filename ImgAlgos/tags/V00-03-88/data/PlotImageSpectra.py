#!/usr/bin/env python
#--------------------

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
#import h5py

from optparse import OptionParser

#--------------------

def get_input_parameters_adv() :

    def_fname = 'spec-xppi0412-r0060-20120507-125420.198726277.txt'

    parser = OptionParser(description='Process optional input parameters.', usage = "usage: %prog [options]")
    parser.add_option('-f', '--fname', dest='fname', default=def_fname, action='store', type='string', help='input file name')
#    parser.add_option('-c', '--cols', dest='cols', default=def_cols, action='store', type='int', help='number of columns in the image array')
    parser.add_option('-v', dest='verbose', action='store_true',  help='set flag to print more details',  default=True)
    parser.add_option('-q', dest='verbose', action='store_false', help='set flag to print less details')
    (opts, args) = parser.parse_args()

    print 'opts:',opts
    print 'args:',args

    return (opts, args)

#--------------------

def get_input_parameters() :

    fname_def = './spec-xppi0412-r0060-20120507-125420.198726277.txt'
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
 
    return fname,ampRange 

#--------------------

def plot_histogram(arr, amp_range=None, figsize=(5,5)) :
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    plt.hist(arr.flatten(), bins=40, range=amp_range)
    #fig.canvas.manager.window.move(500,10)
     
#--------------------

def plot_spectra(arr, amp_range=None, figsize=(5,5)) :
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    fig.canvas.set_window_title('Special case spectra from image')

    fsig = arr[0,]
    fref = arr[1,]
    fdif = arr[2,]
    npix = fsig.shape[0] # =1024
    x = np.arange(npix)

    ax1 = plt.subplot2grid((10,10), (0,0), rowspan=6, colspan=10)
    ax1.plot(x, fsig, '-r', x, fref, '-g')
    ax1.set_xlim(0,npix+1)
    ax1.set_title('Signal and reference spectra',color='r',fontsize=20)
    ax1.set_xlabel('Pixel')
    ax1.set_ylabel('sig, ref')

    ax2 = plt.subplot2grid((10,10), (7,0), rowspan=3, colspan=10)
    ax2.plot(x, fdif, '-b')
    ax2.set_xlim(0,npix+1)
    plt.title('Differential spectrum',color='b',fontsize=20)
    plt.xlabel('Pixel')
    plt.ylabel('2 (sig - ref) / (sig + ref)')

    #fig.canvas.manager.window.move(500,10)
     
#--------------------

def get_array_from_file(fname) :
    print 'get_array_from_file:', fname
    return np.loadtxt(fname, dtype=np.float32)

#--------------------

def do_main() :

    #opts, args = get_input_parameters_adv()
    fname, ampRange  = get_input_parameters()

    arr = get_array_from_file(fname)

    #plot_histogram(arr,range=ampRange)
    #plt.get_current_fig_manager().window.geometry("+950+10")

    print 'arr[0,]=', arr[0,]
    print 'arr[1,]=', arr[1,]
    print 'arr[2,]=', arr[2,]

    plot_spectra(arr, figsize=(8,8))
    plt.get_current_fig_manager().window.geometry("+450+10")

    plt.show()

  
#--------------------

if __name__ == '__main__' :

    do_main()

    sys.exit('The End')

#--------------------
