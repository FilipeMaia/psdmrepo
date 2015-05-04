#------------------------------
"""This module provides access to the calibration parameters

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------

import sys
import numpy as np

import matplotlib
#if matplotlib.get_backend() != 'Qt4Agg' : matplotlib.use('Qt4Agg')

import matplotlib.pyplot  as plt
import matplotlib.lines   as lines
import matplotlib.patches as patches

#---------------------
#class CSPadImageProducer (object) :
#    """This is an empty class"""
#---------------------
#    def __init__ (self) :
#        print 'CSPadImageProducer __init__'
#--------------------------------
#------------------------------

class Storage :
    def __init__(self) :
        pass

#------------------------------
store = Storage() # singleton
#------------------------------

#------------------------------

def fig_axes(figsize=(13,12), title='Image') :
    """ Creates and returns figure, and axes for image and color bar
    """
    fig  = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    axim = fig.add_axes([0.05,  0.03, 0.87, 0.93])
    axcb = fig.add_axes([0.923, 0.03, 0.02, 0.93])
    fig.canvas.set_window_title(title)
    store.fig, store.axim, store.axcb = fig, axim, axcb
    return fig, axim, axcb

#------------------------------

def plot_img(img, mode=None, amin=None, amax=None) :
    
    fig, axim, axcb = store.fig, store.axim, store.axcb

    axim.cla()
    imsh = axim.imshow(img, interpolation='nearest', aspect='auto', origin='upper') # extent=img_range)
    colb = fig.colorbar(imsh, cax=axcb) # , orientation='horizontal')

    ave = np.mean(img) if amin is not None or amax is not None else None
    rms = np.std(img)  if amin is not None or amax is not None else None
    #print 'img ave = %f, rms = %f' % (ave, rms)
    store.amin = amin if amin is not None else ave-1*rms
    store.amax = amax if amax is not None else ave+5*rms

    imsh.set_clim(store.amin, store.amax)

    #print_help(1)

    if mode is None : plt.ioff() # hold contraol at show() (connect to keyboard for controllable re-drawing)
    else            : plt.ion()  # do not hold control

    #fig.canvas.draw()
    plt.show()

#------------------------------

def getArrangedImage() :
    arr = np.arange(2400)
    arr.shape = (40,60)
    return arr

#--------------------------------
def getRandomImage() :
    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=2400)
    arr.shape = (40,60)
    return arr

#------------------------------

def getImageAs2DHist(iX,iY,W=None) :
    """Makes image from iX, iY coordinate index arrays and associated weights, using np.histogram2d(...).
    """
    xsize = np.ceil(iX).max()+1 
    ysize = np.ceil(iY).max()+1
    if W==None : weights = None
    else       : weights = W.flatten()
    H,Xedges,Yedges = np.histogram2d(iX.flatten(), iY.flatten(), bins=[xsize,ysize], range=[[-0.5,xsize-0.5],[-0.5,ysize-0.5]], normed=False, weights=weights) 
    return H

#------------------------------

def getImageFromIndexArrays(iX,iY,W=None) :
    """Makes image from iX, iY coordinate index arrays and associated weights, using indexed array.
    """
    xsize = iX.max()+1 
    ysize = iY.max()+1
    if W==None : weight = np.ones_like(iX)
    else       : weight = W
    img = np.zeros((xsize,ysize), dtype=np.float32)
    img[iX,iY] = weight # Fill image array with data 
    return img

#--------------------------------

def plotHistogram(arr, amp_range=None, figsize=(6,6)) : # range=(0,500)
    fig  = plt.figure(figsize=figsize, dpi=80, facecolor='w',edgecolor='w', frameon=True)
    plt.hist(arr.flatten(), bins=100, range=amp_range)

#--------------------------------

def plotSpectrum(arr, amp_range=None, figsize=(6,6)) : # range=(0,500)
    plotHistogram(arr, amp_range, figsize)

#--------------------------------

def plotImage(arr, img_range=None, amp_range=None, figsize=(12,5), title='Image', origin='upper') : 
    fig  = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    axim = fig.add_axes([0.05,  0.05, 0.95, 0.92])
    imsh = plt.imshow(arr, interpolation='nearest', aspect='auto', origin=origin, extent=img_range) #,extent=self.XYRange, origin='lower'
    colb = fig.colorbar(imsh, pad=0.005, fraction=0.1, shrink=1, aspect=20)
    if amp_range is not None : imsh.set_clim(amp_range[0],amp_range[1])
    #axim.set_title(title, color='b', fontsize=20)
    fig.canvas.set_window_title(title)

#--------------------------------

def plotImageLarge(arr, img_range=None, amp_range=None, figsize=(12,10), title='Image', origin='upper') : 
    fig  = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    axim = fig.add_axes([0.05,  0.03, 0.94, 0.94])
    imsh = axim.imshow(arr, interpolation='nearest', aspect='auto', origin=origin, extent=img_range)
    colb = fig.colorbar(imsh, pad=0.005, fraction=0.09, shrink=1, aspect=40) # orientation=1
    if amp_range is not None : imsh.set_clim(amp_range[0],amp_range[1])
    fig.canvas.set_window_title(title)
    return axim

#--------------------------------

def plotImageAndSpectrum(arr, amp_range=None) : #range=(0,500)
    fig  = plt.figure(figsize=(15,5), dpi=80, facecolor='w', edgecolor='w', frameon=True)
    fig.canvas.set_window_title('Image And Spectrum ' + u'\u03C6')

    ax1   = plt.subplot2grid((10,10), (0,4), rowspan=10, colspan=6)
    axim1 = ax1.imshow(arr, interpolation='nearest', aspect='auto') # , origin='lower' 
    colb1 = fig.colorbar(axim1, pad=0.01, fraction=0.1, shrink=1.00, aspect=20)
    if amp_range is not None : axim1.set_clim(amp_range[0], amp_range[1])
    plt.title('Image', color='b', fontsize=20)

    ax2   = plt.subplot2grid((10,10), (0,0), rowspan=10, colspan=4)
    ax2.hist(arr.flatten(), bins=100, range=amp_range)
    plt.title('Spectrum', color='r',fontsize=20)
    plt.xlabel('Bins')
    plt.ylabel('Stat') #u'\u03C6'
    #plt.ion()

#------------------------------

def plotGraph(x,y, figsize=(10,5)) : 
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    #fig.subplots_adjust(left=0.05, bottom=0.03, right=0.98, top=0.98, wspace=0.2, hspace=0.1)
    #figAxes = fig.add_subplot(111)
    ax = fig.add_axes([0.15, 0.10, 0.78, 0.86])
    ax.plot(x,y,'b-')

#------------------------------

def drawCircle(axes, xy0, radius, linewidth=1, color='w', fill=False) : 
    circ = patches.Circle(xy0, radius=radius, linewidth=linewidth, color=color, fill=fill)
    axes.add_artist(circ)


def drawCenter(axes, xy0, s=10, linewidth=1, color='w') : 
    xc, yc = xy0
    d = 0.15*s
    arrx = (xc+s, xc-s, xc-d, xc,   xc)
    arry = (yc,   yc,   yc-d, yc-s, yc+s)
    line = lines.Line2D(arrx, arry, linewidth=linewidth, color=color)   
    axes.add_artist(line)


def drawLine(axes, xarr, yarr, s=10, linewidth=1, color='w') : 
    line = lines.Line2D(xarr, yarr, linewidth=linewidth, color=color)   
    axes.add_artist(line)


def drawRectangle(axes, xy, width, height, linewidth=1, color='w') :
    rect = patches.Rectangle(xy, width, height, linewidth=linewidth, color=color)
    axes.add_artist(rect)

#--------------------------------

def savefig(fname='img.png', do_print=True) :
    if do_print :
        print 'Save plot in file: %s' % fname 
    plt.savefig(fname)

#--------------------------------

def move(x0=200,y0=100) :
    #plt.get_current_fig_manager().window.move(x0, y0)
    move_str = '+' + str(x0) + '+' + str(y0)
    plt.get_current_fig_manager().window.geometry(move_str)
    #plt.get_current_fig_manager().window.geometry("+50+50")
    pass

#--------------------------------

def show() :
    plt.show()
    #file.close()

#----------------------------------------------

def main() :

    arr = getRandomImage()
    if len(sys.argv)==1   :
        print 'Use command > python %s <test-number [1-5]>' % sys.argv[0]
        sys.exit ('Add <test-number> in command line...')

    elif sys.argv[1]=='1' : plotImage(arr, amp_range=(100,300))
    elif sys.argv[1]=='2' : plotImageAndSpectrum(arr, amp_range=(100,300))
    elif sys.argv[1]=='3' : plotHistogram(arr, amp_range=(100,300), figsize=(10,5))
    elif sys.argv[1]=='4' : plotImage(arr, amp_range=(100,300), figsize=(10,10))
    elif sys.argv[1]=='5' : plotImageLarge(arr, amp_range=(100,300), figsize=(10,10))
    else :
        print 'Non-expected arguments: sys.argv=', sys.argv
        sys.exit ('Check input parameters')

    move(500,10)
    show()

#--------------------------------

if __name__ == "__main__" :

    main()
    sys.exit ( 'End of test.' )

#----------------------------------------------
