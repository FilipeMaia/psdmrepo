#!/usr/bin/env python
#--------------------
# Check input parameters

import numpy as np
import matplotlib.pyplot as plt
import sys

print ' sys.argv[0]: ', sys.argv[0]
print ' len(sys.argv): ', len(sys.argv)

if len(sys.argv) == 2 : fname = sys.argv[1]
else                  : fname = '/reg/neh/home/dubrovin/LCLS/PSANA-test/image.txt'

print 'fname=', fname

#--------------------
# Define graphical methods

def plot_histogram (arr) :
    fig1 = plt.figure(num=1, figsize=(6,6), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    ax1 = plt.subplot2grid((4,4), (0,0), rowspan=2, colspan=4)
    plt.hist(arr.flatten(), bins=100 )
    ax2 = plt.subplot2grid((4,4), (2,0), rowspan=2, colspan=4)
    plt.hist(arr.flatten(), bins=100, range=(0,2000) ) 

def plot_image (arr, range=None) :    # range = (left, right, low, high)
    fig = plt.figure(num=2, figsize=(12,12), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)
    figAxes = fig.add_subplot(111)
    imAxes = figAxes.imshow(arr, origin='upper', interpolation='nearest', aspect='auto',extent=range)
    #imAxes.set_clim(0,100)
    colbar = fig.colorbar(imAxes, pad=0.03, fraction=0.04, shrink=1.0, aspect=40, orientation=1)

#--------------------
# Read and plot array

arr_inp = np.loadtxt(fname, dtype=np.float32)

print arr_inp
print arr_inp.shape

#arr = arr_inp

# MiniCSPad test
arr = arr_inp[10000:11000][:]
plot_image (arr, range=(500, 1000, 11000, 10000))

# CSPad test
#arr = arr_inp[5000:10000][:]
#plot_image (arr, range=(10, 2010, 10000, 5000))

#plot_histogram (arr)

plt.show()

sys.exit()
#--------------------
