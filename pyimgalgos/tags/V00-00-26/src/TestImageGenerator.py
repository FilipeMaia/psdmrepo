#!/usr/bin/env python
#------------------------------
"""TestImageGenerator - a set of methods to generate test images as numpy arrays

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""
#--------------------------------
__version__ = "$Revision$"
#--------------------------------

from time import time
import os
import sys
import math
import numpy as np

import GlobalGraphics as gg # for test purpose

#------------------------------

def random_normal(shape=(1300,1340), mu=200, sigma=25, pbits=0377) :
    arr = mu + sigma*np.random.standard_normal(size=shape)
    if pbits & 1 : print 'Created arr.shape=', arr.shape
    return arr

#------------------------------

def random_exponential(shape=(1300,1340), a0=100, pbits=0377) :
    arr = a0*np.random.standard_exponential(size=shape)
    if pbits & 1 : print 'Created arr.shape=', arr.shape
    return arr

#--------------------------------

def gaussian(r,r0,sigma) :
    factor = 1/ (math.sqrt(2) * sigma)
    rr = factor*(r-r0)
    return np.exp(-rr*rr)

#------------------------------

def ring_xy_random(shape=(1300, 1340)) :
    """Generates the cortesian 2D array with ring-distributed intensity"""

    npy,npx = shape
    xc  = npx/2
    yc  = npy/2
    
    #random = np.random.standard_normal(2)
    random = np.random.random(2)
    #print 'random =', random

    a1  = 100
    r1  = 0.5*npx*random[0]
    s1  = 0.1*npx*random[1]

    x = np.arange(0,npx,1,dtype = np.float32) # np.linspace(0,200,201)
    y = np.arange(0,npy,1,dtype = np.float32) # np.linspace(0,100,101)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt((X-xc)*(X-xc)+(Y-yc)*(Y-yc))
    #print 'R=\n',R

    A = a1 * X * Y * gaussian(R, r1, s1)
    #A = a1 * gaussian(R, r1, s1)
    #print 'A=\n',A

    return A

#------------------------------

def peaks_on_ring(shape=(1300, 1340), npeaks=3) :
    """Generates the cortesian 2D array with ring-distributed intensity"""

    npy,npx = shape
    xc  = npx/2
    yc  = npy/2
    
    #random = np.random.standard_normal(2)
    random = np.random.random(2)
    #print 'random =', random

    a1  = 100
    r1  = 0.4*npx
    s1  = 0.01*npx

    t0 = math.pi*random[0] # random phase

    x = np.arange(0,npx,1,dtype = np.float32) # np.linspace(0,200,201)
    y = np.arange(0,npy,1,dtype = np.float32) # np.linspace(0,100,101)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt((X-xc)*(X-xc)+(Y-yc)*(Y-yc))
    T = np.arctan2(Y-yc, X-xc)
    SINT = np.sin(npeaks*T/2 + t0 )

    A = a1 * gaussian(R, r1, s1) * SINT * SINT

    return A

#------------------------------

def rings_sinc(shape=(1024,1024), pbits=0377) :
    # Create test image - a sinc function, centered in the middle of
    if pbits & 1 : print "Creating test image",

    xsize, ysize = shape
    ratio = float(ysize)/float(xsize)
    if pbits & 1 : print 'ratio = ', ratio
    xmin, xmax = -4, 6
    ymin, ymax = -7*ratio, 3*ratio

    if pbits & 1 : print '\nxmin, xmax, xsize = ', xmin, xmax, xsize
    if pbits & 1 : print '\nymin, ymax, ysize = ', ymin, ymax, ysize

    xarr = np.linspace(xmin, xmax, xsize)
    yarr = np.linspace(ymin, ymax, ysize)
    xgrid, ygrid = np.meshgrid(xarr, yarr)
    rgrid = np.sqrt(xgrid**2 + ygrid**2)
    image = np.abs(np.sinc(rgrid))
    return image

#------------------------------

def cspad2x1_arr() :
    """returns test np.array for cspad 2x1"""
    rows, cols = 185, 388
    #arr2x1  = np.ones((rows,cols),dtype=np.int)
    row2x1 = np.arange(cols)
    col2x1 = np.arange(rows)
    iY, iX = np.meshgrid(row2x1, col2x1)

    #arr2x1 = np.arange(rows*cols)*0.001
    #arr2x1.shape = (rows,cols)

    arr2x1 = gg.getImageFromIndexArrays(iX,iY,iX+iY)
    
    return np.array(arr2x1,dtype=np.float)

#------------------------------

def cspad_nparr(n2x1=32, pbits=0377) :
    """returns test np.array for cspad"""
    segs, rows, cols = 32, 185, 388
    arr2x1 = cspad2x1_arr()
    arr = np.vstack([arr2x1 for seg in range(n2x1)])
    arr.shape = [n2x1, rows, cols]
    return arr

#------------------------------

def monotonicly_rising(shape=(10,10), dtype = np.float32) :
    """Returns monotonicly rising image intensity along the diagonal
    """
    rows, cols = shape
    arr = np.arange(rows*cols, dtype=dtype)
    arr.shape = shape
    return arr

#------------------------------
#------------------------------
#------------------------------
#------------------------------
#------------------------------
#------------------------------

def main() :

    if len(sys.argv)!=2 :
        print 'Use command > python %s <test-number>' % sys.argv[0]
        return

    print 'Test # %s' % sys.argv[1]

    if   sys.argv[1]=='1' : gg.plotImageLarge(random_normal()) #, amp_range=(100,300))
    elif sys.argv[1]=='2' : gg.plotImageLarge(random_exponential())
    elif sys.argv[1]=='3' : gg.plotImageLarge(ring_xy_random())
    elif sys.argv[1]=='4' : gg.plotImageLarge(rings_sinc())
    elif sys.argv[1]=='5' : gg.plotImageLarge(peaks_on_ring())
    else :
        print 'Non-expected arguments: sys.argv=', sys.argv
        sys.exit ('Check input parameters')

    gg.move(500,10)
    gg.show()

#------------------------------

if __name__ == "__main__" :
    main()
    sys.exit ( 'End of %s' % sys.argv[0] )

#------------------------------
