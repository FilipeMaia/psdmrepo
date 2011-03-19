
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotsForCSpadProjections...
#
#------------------------------------------------------------------------

"""Plots for CSpad detector in the EventeDisplay project.

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
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
#from matplotlib.patches import Rectangle
#from matplotlib.artist  import Artist
#from matplotlib.lines   import Line2D
import time
from numpy import *  # for use like       array(...)
import numpy as np

#-----------------------------
# Imports for other modules --
#-----------------------------

import ConfigParameters        as cp
import ConfigCSpad             as cs
import PrintHDF5               as printh5
import FastArrayTransformation as fat

#---------------------
#  Class definition --
#---------------------
class PlotsForCSpadProjections ( object ) :
    """Plots for CSpad detector projections in the EventeDisplay project."""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, plotsCSpad ) :
        """Constructor - initialization."""

        print '\n Initialization of the PlotsForCSpadProjections'

        self.plotsCSpad = plotsCSpad



    #-------------------
    #  Public methods --
    #-------------------


    def plotProjR(self, arr1ev, fig) :
        print 'plotProjR is not implemented yet'

    def plotProjPhi(self, arr1ev, fig) :
        print 'plotProjPhi is not implemented yet'



    def plotProjX(self, arr1ev, fig) :
        print 'plotProjX'

        arrDet = self.plotsCSpad.getImageArrayForDet( arr1ev )

        NBins    = cp.confpars.projX_NBins    
        BinWidth = cp.confpars.projX_BinWidth
        NSlices  = cp.confpars.projX_NSlices 
        SliWidth = cp.confpars.projX_SliWidth
        Xmin     = cp.confpars.projX_Xmin    
        Xmax     = cp.confpars.projX_Xmax    
        Ymin     = cp.confpars.projX_Ymin    
        Ymax     = cp.confpars.projX_Ymax    

        XRange        = (Xmin,Xmax,NBins)
        YRange        = (Ymin,Ymax,NSlices)
        self.XYRange  = (Xmin,Xmax,Ymax,Ymin)
        self.HRange   = (Xmin,Xmax)

        fig.canvas.set_window_title('CSpad X projection')

        fig.subplots_adjust(left=0.15, bottom=0.06, right=0.98, top=0.95, wspace=0.35, hspace=0.3)
        plt.clf()

        #plt.subplot(222)
        #axes = plt.imshow(arrDet[Ymin:Ymax,Xmin:Xmax], origin='upper',interpolation='nearest',extent=self.XYRange)

        plt.subplot(211)
        arr2d = fat.rebinArray(arrDet, XRange, YRange) 
        axes = plt.imshow(arr2d, origin='upper',interpolation='nearest',extent=self.XYRange)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Event '+str(cp.confpars.eventCurrent),color='r',fontsize=20) 

        #print 'arr2d.shape=',arr2d.shape

        plt.subplot(212)
        for slice in range(NSlices):

            arr1slice = arr2d[slice,...]
            xarr = linspace(Xmin, Xmax, num=NBins, endpoint=True)
            #print 'xarr.shape=',xarr.shape
            #print 'arr1slice.shape=',arr1slice.shape

            axes = plt.hist(xarr, bins=NBins, weights=arr1slice, histtype='step')
            plt.xlim(Xmin,Xmax)

        plt.xlabel('X')



    def plotProjY(self, arr1ev, fig) :
        print 'plotProjY'

        arrDet = self.plotsCSpad.getImageArrayForDet( arr1ev )

        NBins    = cp.confpars.projY_NBins    
        BinWidth = cp.confpars.projY_BinWidth
        NSlices  = cp.confpars.projY_NSlices 
        SliWidth = cp.confpars.projY_SliWidth
        Xmin     = cp.confpars.projY_Xmin    
        Xmax     = cp.confpars.projY_Xmax    
        Ymin     = cp.confpars.projY_Ymin    
        Ymax     = cp.confpars.projY_Ymax    

        XRange          = (Xmin,Xmax,NSlices)
        YRange          = (Ymin,Ymax,NBins)
        self.HRange     = (Ymin,Ymax)
        self.XYRange    = (Xmin,Xmax,Ymax,Ymin)
        self.XYRangeR90 = (Ymin,Ymax,Xmin,Xmax,) 

        arr2d = fat.rebinArray(arrDet, XRange, YRange) 

        fig.canvas.set_window_title('CSpad Y projection')

        fig.subplots_adjust(left=0.15, bottom=0.06, right=0.98, top=0.95, wspace=0.35, hspace=0.3)
        plt.clf()

        #plt.subplot(222)
        #axes = plt.imshow(arrDet[Ymin:Ymax,Xmin:Xmax], origin='upper',interpolation='nearest',extent=self.XYRange)

        plt.subplot(211)
        #axes = plt.imshow(arr2d, origin='upper',interpolation='nearest',extent=self.XYRange)
        #plt.xlabel('X')
        #plt.ylabel('Y')

        axes = plt.imshow(np.rot90(arr2d), origin='upper',interpolation='nearest',extent=self.XYRangeR90)
        plt.xlabel('Y')
        plt.ylabel('X')

        plt.title('Event '+str(cp.confpars.eventCurrent),color='r',fontsize=20) 
        #print 'arr2d.shape=',arr2d.shape

        plt.subplot(212)
        for slice in range(NSlices):

            arr1slice = arr2d[...,slice]
            arrbins   = linspace(Ymin, Ymax, num=NBins, endpoint=True)

            axes = plt.hist(arrbins, bins=NBins, weights=arr1slice, histtype='step')
            plt.xlim(Ymin,Ymax)

        plt.xlabel('Y')



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
