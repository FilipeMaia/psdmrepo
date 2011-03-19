#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module FastArrayTransformation...
#
#------------------------------------------------------------------------

"""A set of methods for the fast array transformation based on NumPy and SciPy

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

@author Mikhail S. Dubrovin
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------

import time
import math
import numpy as np
import scipy as sp
import scipy.ndimage


#-----------------------------
# Imports for other modules --
#-----------------------------


import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

#import ConfigParameters as cp

#------------------------
# Exported definitions --
#------------------------

#----------------------------------    

def cart2polar(x, y) :
    r = np.sqrt(x*x + y*y)
    theta = np.rad2deg(np.arctan2(y, x))
    return r, theta

def polar2cart(r, theta) :
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

#----------------------------------    

def coordinateToIndexOpenEnd(V,VRange) :
    Vmin = VRange[0]
    Vmax = VRange[1]
    Nbins= VRange[2]
    factor = float(Nbins) / float(Vmax-Vmin) * (1-1e-9) # In order to get open endpoint [Vmin,Vmax)
    return np.uint32( factor * (V-Vmin) )

#----------------------------------    

def coordinateToIndex(V,VRange) :
    Vmin = VRange[0]
    Vmax = VRange[1]
    Nbins= VRange[2]
    factor = float(Nbins) / float(Vmax-Vmin)
    return np.uint32( factor * (V-Vmin) )

#----------------------------------    

def transformCartToPolarArray(arr, RRange, ThetaRange, origin) :
    """Input Cartesian array elements are summed together in 2D polar array bins"""

    dimY,dimX = arr.shape 
    x = np.arange(dimX) - origin[0]
    y = np.arange(dimY) - origin[1]
    X, Y     = np.meshgrid(x, y)
    R, Theta = cart2polar (X, Y)    
    iR     = coordinateToIndexOpenEnd(R,RRange)        
    iTheta = coordinateToIndexOpenEnd(Theta,ThetaRange)
    i1D = iR + iTheta * RRange[2]
    i1DInPolar = np.arange(ThetaRange[2] * RRange[2], dtype=np.int32)
    
    arrPolar = np.array( sp.ndimage.sum(arr, i1D, index=i1DInPolar) )
    arrPolar.shape = (ThetaRange[2], RRange[2]) 
    return arrPolar
    
#----------------------------------    

def rebinArray(arr2d, XRange, YRange) :
    """Input 2D array elements are summed together in new 2D array bins"""

    arr = arr2d[YRange[0]:YRange[1],XRange[0]:XRange[1]]
    XRangeOff = (0,XRange[1]-XRange[0],XRange[2])
    YRangeOff = (0,YRange[1]-YRange[0],YRange[2])

    dimY,dimX = arr.shape 
    x = np.arange(dimX)
    y = np.arange(dimY)
    X, Y   = np.meshgrid(x, y)
    iXnew = coordinateToIndex(X,XRangeOff)
    iYnew = coordinateToIndex(Y,YRangeOff)
    i1D = iXnew + iYnew * XRange[2]
    i1DInNew = np.arange(XRange[2] * YRange[2], dtype=np.int32)
    
    arrNew = np.array( sp.ndimage.sum(arr, i1D, index=i1DInNew) )
    arrNew.shape = (YRange[2], XRange[2]) 
    return arrNew

#---------------------------------- 
#----------------------------------

def ringIntensity(r,r0,sigma) :
    factor = 1/ (math.sqrt(2) * sigma)
    rr = factor*(r-r0)
    return np.exp(-rr*rr)

def getCartesianArray() :
    """Generates the cortesian 2D array with ring-distributed intensity"""
    npx = 201
    npy = 101

    a1  = 50
    r1  = 80
    s1  = 10

    a2  = 100
    r2  = 110
    s2  = 5

    a3  = 80
    r3  = 160
    s3  = 10
    
    x = np.arange(0,npx,1,dtype = np.float32) # np.linspace(0,200,201)
    y = np.arange(0,npy,1,dtype = np.float32) # np.linspace(0,100,101)
    X, Y = np.meshgrid(x, y)

    #A = 100 * abs( np.sin( np.sqrt(X*X+Y*Y)/20 ) )

    R = np.sqrt(X*X+Y*Y)
    #print 'R=\n',R

    A = a1 * ringIntensity(R, r1, s1)
    A+= a2 * ringIntensity(R, r2, s2)
    A+= a3 * ringIntensity(R, r3, s3)

    #print 'A=\n',A

    return A

#----------------------------------
#----------------------------------

def draw2DImage(arr, showTimeSec=None, winTitle='Three images') :
    """Graphical presentation for three 2D arrays.""" 

    plt.ion() 
    fig = plt.figure(figsize=(6,4), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    #fig.subplots_adjust(left=0.10, bottom=0.05, right=0.98, top=0.94, wspace=0.3, hspace=0)
    fig.canvas.set_window_title(winTitle) 
    plt.clf() 
    #plt.get_current_fig_manager().window.move(100,0)
    drawImage(arr,'Image and Spectrum')

    drawOrShow(showTimeSec)


def drawImageAndSpectrum(arr, showTimeSec=None, winTitle='Three images') :
    """Graphical presentation for three 2D arrays.""" 

    plt.ion() 
    fig = plt.figure(figsize=(8,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    #fig.subplots_adjust(left=0.10, bottom=0.05, right=0.98, top=0.94, wspace=0.3, hspace=0)
    fig.canvas.set_window_title(winTitle) 
    plt.clf() 
    #plt.get_current_fig_manager().window.move(100,0)

    plt.subplot2grid((10,10), (0,0), rowspan=7, colspan=10)  
    drawImage(arr,'Image and Spectrum')
    plt.subplot2grid((10,10), (7,0), rowspan=4, colspan=10)
    drawHistogram(arr)

    drawOrShow(showTimeSec)


def drawOrShow(showTimeSec=None) :

    if str(showTimeSec).lower() == 'show' :
        plt.ioff()
        plt.show()
    
    elif showTimeSec != None :
        plt.draw()
        plt.draw()
        print 'Sleep', showTimeSec, 'sec'
        time.sleep(showTimeSec)
        #plt.close(1)

    else :
        plt.draw()
        plt.draw()
        plt.draw()


def drawImage(arr2d, title) :
    """Draws the image and color bar for a single 2D array."""
    axes = plt.imshow(arr2d, origin='lower', interpolation='nearest') # origin='lower', origin='upper'
    colb = plt.colorbar(axes, pad=0.05, fraction=0.10, shrink = 1, aspect = 20, orientation=1) #, ticks=coltickslocs
    plt.title(title, color='r', fontsize=20)
    #plt.clim(imageAmin,imageAmax)
    

def drawHistogram(arr) :
    """Draws histogram with reduced number of ticks for vertical axes for input array (dimension does not matter)."""
    plt.hist(arr.flatten(), bins=100) #, range=(0,100))
    Ymin, Ymax = plt.ylim()
    plt.yticks( np.arange(int(Ymin), int(Ymax), int((Ymax-Ymin)/3)) )

#----------------------------------

def mainTest() :
    arr = getCartesianArray()
    draw2DImage(arr, showTimeSec=0)
    plt.get_current_fig_manager().window.move(10,10)

    origin     = (0,0)
    RRange     = (0,200,200)
    ThetaRange = (0,90,90)

    polar_arr = transformCartToPolarArray(arr, RRange, ThetaRange, origin)
    draw2DImage(polar_arr, showTimeSec=0)
    plt.get_current_fig_manager().window.move(500,10)

    XRange     = (0,200,100)
    YRange     = (0,100,50)

    rebinned_arr = rebinArray(arr, XRange, YRange)
    draw2DImage(rebinned_arr, showTimeSec=0)
    plt.get_current_fig_manager().window.move(10,394)

    drawOrShow(showTimeSec='show')

#----------------------------------

if __name__ == "__main__" :

    mainTest()

#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------
