#------------------------------
"""A set of methods for the fast array transformation based on NumPy and SciPy

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
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
    """For numpy arryys x and y returns the numpy arrays of r and theta 
    """
    r = np.sqrt(x*x + y*y)
    theta  = np.rad2deg(np.arctan2(y, x)) #[-180,180]
    #theta0 = np.select([theta<0, theta>=0],[theta+360,theta]) #[0,360]
    return r, theta

def polar2cart(r, theta) :
    """For numpy arryys r and theta returns the numpy arrays of x and y 
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

#----------------------------------    

def coordinateToIndexOpenEnd(V,VRange) :
    Vmin, Vmax, Nbins = VRange
    factor = float(Nbins-1) / float(Vmax-Vmin) #* (1-1e-9) # In order to get open endpoint [Vmin,Vmax)
    return np.int32( factor * (V-Vmin) )

#----------------------------------    

def coordinateToIndex(V,VRange) :
    Vmin, Vmax, Nbins = VRange
    factor = float(Nbins) / float(Vmax-Vmin)
    return np.int32( factor * (V-Vmin) )

#----------------------------------    

def coordinateToIndexProtected(V,VRange) :
    Vmin, Vmax, Nbins = VRange
    factor = float(Nbins) / float(Vmax-Vmin)
    indarr = np.int32( factor * (V-Vmin) )
    return np.select([V<Vmin], [-1000], default=indarr)
 
#----------------------------------    

def transformCartToPolarArray(arr, RRange, ThetaRange, Origin, rCorrIsOn=False) :
    """Input Cartesian array elements are summed together in 2D polar array bins

    This transformation works fine when the ThetaRange is inside a single sheet [-180,180] degree.
    """

    arrShape = arr.shape 
    #print 'arrShape=\n', arrShape

    dimY,dimX = arr.shape
    xc,yc = Origin
    x = np.arange(dimX) - xc
    y = np.arange(dimY) - yc

    X, Y       = np.meshgrid(x,y)
    R, Theta   = cart2polar (X,Y) 

    iR         = coordinateToIndexProtected(R,RRange)        
    iTheta     = coordinateToIndexProtected(Theta,ThetaRange) #Theta is in the range [-180,180] 0-sheet

    #arrZeroes = np.zeros(shape,dtype=float)

    if rCorrIsOn :
        factor = RRange[1]/R
        arrMasked = np.select([iTheta<0, iTheta>=ThetaRange[2], iR<0, iR>=RRange[2]], [0,0,0,0], default=arr*factor)
    else :
        arrMasked = np.select([iTheta<0, iTheta>=ThetaRange[2], iR<0, iR>=RRange[2]], [0,0,0,0], default=arr)

    i1DInPolar = np.arange(ThetaRange[2] * RRange[2], dtype=np.int32)

    i1D      = iTheta + iR * ThetaRange[2]
    arrPolar = np.array( sp.ndimage.sum(arrMasked, i1D, index=i1DInPolar) )
    arrPolar.shape = (RRange[2], ThetaRange[2])
    return arrPolar

    #i1D      = iR + iTheta * RRange[2]
    #arrPolar = np.array( sp.ndimage.sum(arrMasked, i1D, index=i1DInPolar) )
    #arrPolar.shape = (ThetaRange[2], RRange[2]) 
    #return arrPolar.T

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

def gaussian(r,r0,sigma) :
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

    A = a1 * gaussian(R, r1, s1)
    A+= a2 * gaussian(R, r2, s2)
    A+= a3 * gaussian(R, r3, s3)
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

    A = a1 * X * Y * gaussian(R, r1, s1)
    #print 'A=\n',A

    return A

#----------------------------------
#----------------------------------

def getGainCorrectionArrayFromAverage(arr_ave) :

    print 'gainCorrectionArrayFromAverage'
    arr_weights = np.select([arr_ave==0], [0], default=1) # set 0/1 weights for <1/positive array elements
    average_over_nonzero = np.average(arr_ave, weights=arr_weights) # get average for non-zero elements
    print 'average_over_nonzero =', average_over_nonzero 
    arr_gain_corr = np.select([arr_ave>0], [average_over_nonzero/arr_ave], default=0) # get gain factors
    arr_gain_corr = np.select([arr_gain_corr < 10], [arr_gain_corr], default=0) # select gain factors < 10

    #printArrForTest(arr_gain_corr)
    printMeanAndStandardDeviation(arr_gain_corr)
    return arr_gain_corr

def printMeanAndStandardDeviation(arr) :
    arr_weights = np.select([arr==0], [0], default=1)
    print 'Mean of elements for the gain correction array (ixcluding 0s) =',np.average(arr, weights=arr_weights)
    print 'Mean of elements for the gain correction array (including 0s) =',np.mean(arr)
    print 'Standard deviation of elements for the gain correction array  =',np.std(arr)


def printArrForTest(arr) :
    print 'arr_gain_corr.shape =',arr_gain_corr.shape

    number_non_zero = 0
    for row in range(arr.shape[0]) :
        for col in range(arr.shape[1]) :

            val = arr[row][col]
            if val != 0 :
                number_non_zero += 1
                if number_non_zero > 1000 : break
                print 'row,col,val:',row,col,val

#----------------------------------
#----------------------------------

def draw2DImage(arr, showTimeSec=None, winTitle='Three images',Range=None) :
    """Graphical presentation for three 2D arrays.""" 

    plt.ion() 
    fig = plt.figure(figsize=(6,4), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    #fig.subplots_adjust(left=0.10, bottom=0.05, right=0.98, top=0.94, wspace=0.3, hspace=0)
    fig.canvas.set_window_title(winTitle) 
    plt.clf() 
    #plt.get_current_fig_manager().window.geometry("+0+100")
    drawImage(arr,'Image',Range)

    drawOrShow(showTimeSec)


def drawImageAndSpectrum(arr, showTimeSec=None, winTitle='Three images') :
    """Graphical presentation for three 2D arrays.""" 

    plt.ion() 
    fig = plt.figure(figsize=(8,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    #fig.subplots_adjust(left=0.10, bottom=0.05, right=0.98, top=0.94, wspace=0.3, hspace=0)
    fig.canvas.set_window_title(winTitle) 
    plt.clf() 
    #plt.get_current_fig_manager().window.geometry("+100+0")

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
    axes = plt.imshow(arr2d, origin='upper', interpolation='nearest',extent=Range, aspect='auto') # origin='lower', origin='upper',origin='lower', 
    colb = plt.colorbar(axes, pad=0.01, fraction=0.10, aspect = 20, orientation='vertical') #, ticks=coltickslocs, shrink = 0.7
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
    plt.get_current_fig_manager().window.geometry("+10+10")
    plt.title('Original 2D array', color='r', fontsize=20)
    
    origin     = (0,0)
    RRange     = (0,200,200)
    ThetaRange = (0,90,90)

    polar_arr = transformCartToPolarArray(arr, RRange, ThetaRange, origin)
    draw2DImage(polar_arr, showTimeSec=0)
    plt.get_current_fig_manager().window.geometry("+500+10")
    plt.title('R-Phi transformation', color='r', fontsize=20)

    XRange     = (0,200,100)
    YRange     = (0,100,50)

    rebinned_arr = rebinArray(arr, XRange, YRange)
    draw2DImage(rebinned_arr, showTimeSec=0)
    plt.get_current_fig_manager().window.geometry("+10+394")
    plt.title('Re-binned 2D array', color='r', fontsize=20)

    drawOrShow(showTimeSec='show')

#----------------------------------

def mainTest1Ring() :
    arr = getCartesianArray1Ring()
    draw2DImage(arr, showTimeSec=0)
    plt.get_current_fig_manager().window.geometry("+10+10")
    plt.title('Original 2D array', color='r', fontsize=20)
    
    origin = (50,40)
    RRange = (0,50,10)
    PRange = (-90,180,90)
    #PRange = (0,180,90)
    #PRange = (-180,180,18)
    RPRange= (PRange[0], PRange[1], RRange[0], RRange[1])

    polar_arr = transformCartToPolarArray(arr, RRange, PRange, origin)
    draw2DImage(polar_arr, showTimeSec=0, Range=RPRange)
    plt.get_current_fig_manager().window.geometry("+500+10")
    plt.title('R-Phi transformation', color='r', fontsize=20)

    XRange     = (0,100,50)
    YRange     = (0,80,40)
    XYRange= (XRange[0], XRange[1], YRange[0], YRange[1])

    rebinned_arr = rebinArray(arr, XRange, YRange)
    draw2DImage(rebinned_arr, showTimeSec=0, Range=XYRange)
    plt.get_current_fig_manager().window.geometry("+10+394")
    plt.title('Re-binned 2D array', color='r', fontsize=20)

    drawOrShow(showTimeSec='show')

#----------------------------------

if __name__ == "__main__" :

    mainTest1Ring()
    #mainTest3Rings()

#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------
