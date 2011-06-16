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
    theta  = np.rad2deg(np.arctan2(y, x)) #[-180,180]
    #theta0 = np.select([theta<0, theta>=0],[theta+360,theta]) #[0,360]
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
    factor = float(Nbins-1) / float(Vmax-Vmin) #* (1-1e-9) # In order to get open endpoint [Vmin,Vmax)
    return np.int32( factor * (V-Vmin) )

#----------------------------------    

def coordinateToIndex(V,VRange) :
    Vmin = VRange[0]
    Vmax = VRange[1]
    Nbins= VRange[2]
    factor = float(Nbins) / float(Vmax-Vmin)
    return np.int32( factor * (V-Vmin) )

#----------------------------------    

def coordinateToIndexProtected(V,VRange) :
    Vmin = VRange[0]
    Vmax = VRange[1]
    Nbins= VRange[2]
    factor = float(Nbins) / float(Vmax-Vmin)
    indarr = np.int32( factor * (V-Vmin) )
    return np.select([V<Vmin], [-1000], default=indarr)
 
#----------------------------------    

def transformCartToPolarArray(arr, RRange, ThetaRange, origin) :
    """Input Cartesian array elements are summed together in 2D polar array bins"""

    dimY,dimX = arr.shape 
    x = np.arange(dimX) - origin[0]
    y = np.arange(dimY) - origin[1]

    X, Y       = np.meshgrid(x, y)
    R, Theta   = cart2polar (X, Y)

    iR         = coordinateToIndexProtected(R,RRange)        
    iTheta     = coordinateToIndexProtected(Theta,ThetaRange)
    arrMasked = np.select([iTheta<0,iR<0], [0,0], default=arr)
    #arrMasked  = arr * maskT * maskR

    #i1D        = iR + iTheta * RRange[2] #THIS IS WRONG
    i1D        = iTheta + iR * ThetaRange[2]
    i1DInPolar = np.arange(ThetaRange[2] * RRange[2], dtype=np.int32)

    arrPolar = np.array( sp.ndimage.sum(arrMasked, i1D, index=i1DInPolar) )
    #arrPolar.shape = (ThetaRange[2], RRange[2]) 
    arrPolar.shape = (RRange[2], ThetaRange[2])
    return arrPolar

#----------------------------------    

def applyRadialNormalizationToPolarArray(arrPolar, RRange) :
    """ Apply radial normalization (per bin) to polar array"""

    #print 'arrPolar.shape=', arrPolar.shape
    #print 'RRange=', RRange[0],RRange[1],RRange[2]

    r = np.linspace(RRange[0],RRange[1],RRange[2],endpoint=False)
    #print 'r=\n', r

    twopi = 2 * 3.14159265359
    f =  np.select([r<1],[1], default=1/(twopi * r))
    #print 'f=\n', f

    farr = np.repeat(f,arrPolar.shape[1])
    farr.shape = arrPolar.shape
    #print 'farr.shape=', farr.shape
    #print 'farr=\n', farr

    return arrPolar * farr


    
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

def getCartesianArray3Rings() :
    """Generates the cortesian 2D array with ring-distributed intensity"""
    npx = 200
    npy = 100

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

    R = np.sqrt(X*X+Y*Y)
    #print 'R=\n',R

    A = a1 * ringIntensity(R, r1, s1)
    A+= a2 * ringIntensity(R, r2, s2)
    A+= a3 * ringIntensity(R, r3, s3)
    #print 'A=\n',A

    return A

#----------------------------------


def getCartesianArray1Ring() :
    """Generates the cortesian 2D array with ring-distributed intensity"""
    npx = 100
    npy = 80

    xc  = 50
    yc  = 40
    
    a1  = 100
    r1  = 30
    s1  = 5

    x = np.arange(0,npx,1,dtype = np.float32) # np.linspace(0,200,201)
    y = np.arange(0,npy,1,dtype = np.float32) # np.linspace(0,100,101)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt((X-xc)*(X-xc)+(Y-yc)*(Y-yc))
    #print 'R=\n',R

    A = a1 * X * Y * ringIntensity(R, r1, s1)
    #print 'A=\n',A

    return A


#----------------------------------
#----------------------------------

def draw2DImage(arr, showTimeSec=None, winTitle='Three images',Range=None) :
    """Graphical presentation for three 2D arrays.""" 

    plt.ion() 
    fig = plt.figure(figsize=(6,4), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    #fig.subplots_adjust(left=0.10, bottom=0.05, right=0.98, top=0.94, wspace=0.3, hspace=0)
    fig.canvas.set_window_title(winTitle) 
    plt.clf() 
    #plt.get_current_fig_manager().window.move(100,0)
    drawImage(arr,'Image',Range)

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


def drawImage(arr2d, title, Range=None) :
    """Draws the image and color bar for a single 2D array."""
    axes = plt.imshow(arr2d, origin='lower', interpolation='nearest',extent=Range) # origin='lower', origin='upper'
    colb = plt.colorbar(axes, pad=0.06, fraction=0.10, shrink = 0.7, aspect = 20, orientation=1) #, ticks=coltickslocs
    plt.title(title, color='r', fontsize=20)
    #plt.clim(imageAmin,imageAmax)
    

def drawHistogram(arr) :
    """Draws histogram with reduced number of ticks for vertical axes for input array (dimension does not matter)."""
    plt.hist(arr.flatten(), bins=100) #, range=(0,100))
    Ymin, Ymax = plt.ylim()
    plt.yticks( np.arange(int(Ymin), int(Ymax), int((Ymax-Ymin)/3)) )

#----------------------------------

def mainTest3Rings() :
    arr = getCartesianArray3Rings()
    draw2DImage(arr, showTimeSec=0)
    plt.get_current_fig_manager().window.move(10,10)
    plt.title('Original 2D array', color='r', fontsize=20)
    
    origin     = (0,0)
    RRange     = (0,200,200)
    ThetaRange = (0,90,90)

    polar_arr = transformCartToPolarArray(arr, RRange, ThetaRange, origin)
    draw2DImage(polar_arr, showTimeSec=0)
    plt.get_current_fig_manager().window.move(500,10)
    plt.title('R-Phi transformation', color='r', fontsize=20)

    XRange     = (0,200,100)
    YRange     = (0,100,50)

    rebinned_arr = rebinArray(arr, XRange, YRange)
    draw2DImage(rebinned_arr, showTimeSec=0)
    plt.get_current_fig_manager().window.move(10,394)
    plt.title('Re-binned 2D array', color='r', fontsize=20)

    drawOrShow(showTimeSec='show')

#----------------------------------

def mainTest1Ring() :
    arr = getCartesianArray1Ring()
    draw2DImage(arr, showTimeSec=0)
    plt.get_current_fig_manager().window.move(10,10)
    plt.title('Original 2D array', color='r', fontsize=20)
    
    origin = (50,40)
    RRange = (0,50,10)
    PRange = (-90,90,90)
    #PRange = (-180,180,18)
    RPRange= (RRange[0], RRange[1], PRange[0], PRange[1])

    polar_arr = transformCartToPolarArray(arr, RRange, PRange, origin)
    draw2DImage(polar_arr, showTimeSec=0, Range=RPRange)
    plt.get_current_fig_manager().window.move(500,10)
    plt.title('R-Phi transformation', color='r', fontsize=20)

    XRange     = (0,100,50)
    YRange     = (0,80,40)
    XYRange= (XRange[0], XRange[1], YRange[0], YRange[1])

    rebinned_arr = rebinArray(arr, XRange, YRange)
    draw2DImage(rebinned_arr, showTimeSec=0, Range=XYRange)
    plt.get_current_fig_manager().window.move(10,394)
    plt.title('Re-binned 2D array', color='r', fontsize=20)

    drawOrShow(showTimeSec='show')

#----------------------------------

if __name__ == "__main__" :

    #mainTest1Ring()
    mainTest3Rings()

#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------
