#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module DeviationFromLine...
#
#------------------------------------------------------------------------

"""Module DeviationFromLine for CSPadAlignment package

CSPadAlignment package is intended to check quality of the CSPad alignment
using image of wires illuminated by flat field.
Shadow of wires are compared with a set of straight lines, which can be
interactively adjusted.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$
#------------------------------
#!/usr/bin/env python
#----------------------------------

import numpy as np
#import scipy.ndimage as spi             
import matplotlib.pyplot as plt
import time                    
import sys
import math # cos(x), sin(x), radians(x), degrees() 

import ImageParameters   as imp # for the plot file name

#import GlobalMethods as gm

#----------------------------------

class DeviationFromLine :
    """Evaluate and draw deviation from line"""
    def __init__ (self) :
        print 'DeviationFromLine - actually does not use any member data...'
                   
##-----------------------------------------------------

    def evaluateDeviationFromLine(self, arr, line_coord, line_index, APeak, AWireMin, AWireMax, band=12):

        arr2d = np.array(arr)
        xmin, xmax, ymin, ymax, orient = line_coord
        print 'xmin, xmax, ymin, ymax = ', xmin, xmax, ymin, ymax, orient 

        xhis = []
        dhis = []
        #--------------------
        if orient  == 'h' :
            print 'HORIZONTAL line'

            k = float(ymax-ymin)/float(xmax-xmin)

            for x in range(xmin,xmax) :
                y = k*(x-xmin) + ymin

                ybandmin = max( int(y-band),    0 ) 
                ybandmax = min( int(y+band), 1710 ) 

                arr1     = arr2d[ybandmin:ybandmax,x]
                yarr     = np.arange(ybandmin,ybandmax,1) # dtype=float32
                cond     = np.logical_and([arr1>AWireMin],[arr1<AWireMax])
                weights  = np.select(cond, [APeak - arr1], default=0.)
                #print 'weights.shape=', weights.shape
                #print 'yarr.shape=', yarr.shape
                #print 'yarr=', yarr
                #print 'weights=', weights

                xhis.append(x)

                if np.sum(weights) != 0 :
                    yave = np.average(yarr,0,weights)
                    dhis.append(yave-y)
                    #print 'x, y, yav, deviation=', x, y, yave, yave-y 
                else:
                    dhis.append(0)

        #--------------------

        elif orient == 'v' :
            print 'VERTICAL line'

            k = float(xmax-xmin)/float(ymax-ymin)

            for y in range(ymax,ymin) :
                x = k*(y-ymin) + xmin

                xbandmin = max( int(x-band),    0 ) 
                xbandmax = min( int(x+band), 1710 ) 

                arr1     = arr2d[y,xbandmin:xbandmax]
                xarr     = np.arange(xbandmin,xbandmax,1) # dtype=float32
                cond     = np.logical_and([arr1>AWireMin],[arr1<AWireMax])
                weights  = np.select(cond, [APeak - arr1], default=0.)

                #print 'arr1.shape=', arr1.shape
                #print 'arr1=', arr1
    

                xhis.append(y)
    
                if np.sum(weights) != 0 :
                    xave = np.average(xarr,0,weights)
                    dhis.append(xave-x)
                    #print 'y, x, xav, deviation=', y, x, xave, xave-x 
                else:
                    dhis.append(0)

        #--------------------

        else :
            return

        #--------------------

        arrX = np.array(xhis)
        arrY = np.array(dhis)
        #print 'arrX.shape=', arrX.shape  
        #print 'arrY.shape=', arrY.shape  

        self.drawDeviationFromLine(line_index,arrX,arrY,band)

##-----------------------------------------------------
    
    def drawDeviationFromLine(self, line_index, arrX, arrY, band) :

        title = 'Deviation histogram for line '+str(line_index)
        #gm.openFigure(line_index+10,12,5,title)
        axes = plt.hist(arrX, bins=arrX.shape[0], weights=arrY, histtype='step')
        plt.ylim(-band,band)
        plt.grid()
        plt.title(title, color='b', fontsize=20)
        #plt.savefig('plot-dev-from-line-'+str(line_index)+'.png')
        plt.savefig('plot-dev-from-line-' + str(line_index) + '-' + imp.impars.plot_fname_suffix + '.png')

##-----------------------------------------------------
    
    def drawProfileAlongLine(self, arr, line_coord, lind) :

        arr2d = np.array(arr)
        x1, x2, y1, y2, orient = line_coord
        xmin = min(x1,x2)
        xmax = max(x1,x2)
        ymin = min(y1,y2)
        ymax = max(y1,y2)
        print 'xmin, xmax, ymin, ymax = ', xmin, xmax, ymin, ymax, orient 

        profile = []
        if xmax-xmin > ymax-ymin :
            print 'Plot profile for X bins'
            k = float(ymax-ymin)/float(xmax-xmin)
            arrX = np.arange(xmin,xmax,dtype=np.int16)

            for x in arrX :
                y = int( k*(x-xmin) + ymin )
                profile.append(arr2d[y,x])
                print 'x,y=',x,y,'   profile=',arr2d[y,x]
 
        else :
            print 'Plot profile for Y bins'
            k = float(xmax-xmin)/float(ymax-ymin)
            arrX = np.arange(ymin,ymax,dtype=np.int16)

            for y in arrX :
                x = int( k*(y-ymin) + xmin )
                profile.append(arr2d[y,x])
                print 'x,y=',x,y,'   profile=',arr2d[y,x]

        arrY = np.array(profile)
        axes = plt.hist(arrX, bins=arrX.shape[0], weights=arrY, histtype='step')
        #plt.ylim(0,4000)
        plt.grid()
        plt.title('Profile along the line '+str(lind), color='b', fontsize=20)
        plt.savefig('plot-profile-along-line-' + str(lind) + '-' + imp.impars.plot_fname_suffix + '.png')

        return

##-----------------------------------------------------

def main():

    #for line in range(13) :
    #    evaluateDeviationFromLine(imarr.arr,imarr.line_coord[line],line, APeak, AWireMin, AWireMax)

    print 'Not a standalone example...'

##-----------------------------------------------------
if __name__ == '__main__':
    main()
##-----------------------------------------------------
