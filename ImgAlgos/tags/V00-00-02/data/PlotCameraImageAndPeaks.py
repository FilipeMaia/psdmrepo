#!/usr/bin/env python
#--------------------

import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import matplotlib.patches as patches # for patches.Circle

#--------------------

class Storage :
    def __init__(self) :
        print 'Storage object is created'

    def printStorage(self) :
        print 'You print the class Storage object'

#--------------------
# Define graphical methods

def plot_image (arr, range=None, zrange=None, store=None) :    # range = (left, right, low, high), zrange=(zmin,zmax)
    fig = plt.figure(num=1, figsize=(12,12), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)
    store.figAxes = figAxes = fig.add_subplot(111)
    store.imAxes  = imAxes = figAxes.imshow(arr, origin='upper', interpolation='nearest', aspect='auto',extent=range)
    #imAxes.set_clim(1300,2000)
    if zrange != None : imAxes.set_clim(zrange[0],zrange[1])
    colbar = fig.colorbar(imAxes, pad=0.03, fraction=0.04, shrink=1.0, aspect=40, orientation=1)


def plot_histogram(arr,range=(0,500),figsize=(5,5), store=None) :
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w',edgecolor='w',frameon=True)
    plt.hist(arr.flatten(), bins=60, range=range)
    #fig.canvas.manager.window.move(500,10)


def plot_peaks (arr_peaks, store=None) :  
    axes = store.figAxes
    ampave = np.average(arr_peaks,axis=0)[2]
    print 'ampave=', ampave
    for peak in arr_peaks :
        print peak[0], peak[1], peak[2]
        xy0, r0 = (peak[0], peak[1]), 10*peak[2]/ampave
        circ = patches.Circle(xy0, radius=r0, linewidth=2, color='r', fill=False)
        axes.add_artist(circ)

#--------------------
# Read and plot array

print ' sys.argv[0]: ', sys.argv[0]
print ' len(sys.argv): ', len(sys.argv)

if len(sys.argv) == 2 : fname = sys.argv[1]
else                  : fname = './image0_ev000115.txt'

print 'fname=', fname
arr = np.loadtxt(fname, dtype=np.float32)
print arr
print 'arr.shape=', arr.shape

# Read peaks form file
fname_peaks = 'image_peaks_' + fname.rsplit('_')[1]
arr_peaks = np.loadtxt(fname_peaks, dtype=np.double)
print 'Try to get peaks from file: ', fname_peaks

s = Storage()

#ampRange = (1000,1500)
#ampRange = (-20,100) # For subtracted pedestals
ampRange = (-100,500) # For subtracted pedestals

plot_image(arr, zrange=ampRange, store=s)
plt.get_current_fig_manager().window.move(10,10)

plot_peaks(arr_peaks, store=s)

plot_histogram(arr,range=ampRange, store=s)
plt.get_current_fig_manager().window.move(950,10)

plt.show()

#--------------------
