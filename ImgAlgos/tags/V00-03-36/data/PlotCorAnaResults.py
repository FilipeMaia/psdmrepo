#!/usr/bin/env python
#--------------------

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

#--------------------

def get_input_parameters() :

    def_fname = 'work_corana/img-xcs-r0015-hist.txt'

    parser = OptionParser(description='Process optional input parameters.', usage = "usage: %prog [options] <file-name.txt>")
    parser.add_option('-f', '--fname', dest='fname', default=def_fname, action='store', type='string', help='input file name')
#    parser.add_option('-c', '--cols', dest='cols', default=def_cols, action='store', type='int', help='number of columns in the image array')
#    parser.add_option('-v', dest='verbose', action='store_true',  help='set flag to print more details',  default=True)
#    parser.add_option('-q', dest='verbose', action='store_false', help='set flag to print less details')
    (opts, args) = parser.parse_args()
    print 'opts:',opts
    print 'args:',args

    return (opts, args)

#--------------------

def plot_histogram(arr, lims=(0,500), figsize=(5,5)) :
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    plt.hist(arr.flatten(), bins=40, range=lims)
    #fig.canvas.manager.window.move(500,10)
     
#--------------------

def plot_correlators(arr,figsize=(6,8)) :
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w',edgecolor='w',frameon=True)
    fig.canvas.set_window_title('Correlators')
    #fig.subplots_adjust(left=0.20)

    for i in range(1,7,1) :
        plot_corr_subplot(arr,i)
     
#--------------------

def plot_corr_subplot(arr,ind,nr=3,nc=2) :
    tau = arr[...,0]
    cor = arr[...,ind]
#    ax1 = plt.subplot2grid((10,10), (0,0), rowspan=4, colspan=10)
#    ax1.plot(tau, cor1, '-r', tau, cor2, '-g')
    ax1 = plt.subplot(nr,nc,ind)
    ax1.semilogx(tau, cor, '-bo')
#    ax1.set_ylim(99.,101.)
#    ax1.set_xlim(0,npix+1)
    ax1.set_title('Region %d'%(ind),color='r',fontsize=15, position=(0.5,0.8), style='italic')
    ax1.set_xlabel('tau')
    ax1.set_ylabel('G2')
    ax1.grid(True)
     
#--------------------

def get_array_from_file(fname) :
    print 'get_array_from_file:', fname
    return np.loadtxt(fname, dtype=np.float32)

#--------------------

def do_main() :

    opts, args = get_input_parameters()

    fname = opts.fname

    if len(args) :
        fname = args[0]

    arr = get_array_from_file(fname)

    #print 'arr:\n', arr
    print 'arr.shape=', arr.shape
    print 'arr_tau=', arr[...,0]

    plot_correlators(arr, figsize=(8,8))
    plt.get_current_fig_manager().window.geometry("+450+10")
    plt.show()
  
#--------------------

if __name__ == '__main__' :

    do_main()

    sys.exit('The End')

#--------------------
