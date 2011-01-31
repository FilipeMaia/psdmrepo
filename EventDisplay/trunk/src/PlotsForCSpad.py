#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotsForCSpad...
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
import time
from numpy import *  # for use like       array(...)
import numpy as np

#-----------------------------
# Imports for other modules --
#-----------------------------

import ConfigParameters as cp

#---------------------
#  Class definition --
#---------------------
class PlotsForCSpad ( object ) :
    """Plots for CSpad detector in the EventeDisplay project.

    @see BaseClass
    @see OtherClass
    """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor initialization.
        """

        print '\n Initialization of the PlotsForCSpad'
        #print 'using MPL version: ', matplotlib.__version__
        #self.fig1_window_is_open = False

    #-------------------
    #  Public methods --
    #-------------------

    def plotCSpadV1Image( self, arr1ev, fig ):
        """Plot 2d image from input array. V1 for run ~546, array for entire detector"""
        quad=2
        arr1quad = arr1ev[quad,...] 
        self.plotCSpadQuadImage( arr1quad, fig )

    def plotCSpadV2Image( self, arr1quad, fig ):
        """Plot 2d image from input array. V2 for run ~900 contain array for quad 2"""
        self.plotCSpadQuadImage( arr1quad, fig )


    def plotCSpadV1Spectrum( self, arr1ev, fig, plot=16 ):
        """Plot 2d image from input array. V1 for run ~546, array for entire detector"""
        quad=2
        arr1quad = arr1ev[quad,...] 
        if plot ==  8 : self.plotCSpadQuad8SpectraOf2x1( arr1quad, fig )
        if plot == 16 : self.plotCSpadQuad16Spectra( arr1quad, fig )

    def plotCSpadV2Spectrum( self, arr1quad, fig, plot=16 ):
        """Plot 2d image from input array. V2 for run ~900 contain array for quad 2"""
        if plot ==  8 : self.plotCSpadQuad8SpectraOf2x1( arr1quad, fig )
        if plot == 16 : self.plotCSpadQuadSpectrum( arr1quad, fig )

    def plotCSpadQuadImage( self, arr1quad, fig ):
        """Plot 2d image from input array."""

        #print 'plot_CSpadQuad()'       

        fig.canvas.set_window_title("CSpad image")
        plt.clf() # clear plot
        
        arrgap=zeros( (185,4) ) # make additional 2D-array of 0-s for the gap between two 1x1 pads
        
        #pair=7
        
        for pair in xrange(8): # loop for pair = 0,1,2,...,7
            #print 'pair=', pair
        
            asic1x2  = arr1quad[pair,...]
            #asic1x2  = arr1ev[quad,pair,...]
            #print 'asic1x2=',asic1x2    
        
            asics    = hsplit(asic1x2,2)
            arr      = hstack((asics[0],arrgap,asics[1]))
        
            panel = 421+pair
            pantit='ASIC ' + str(2*pair) + ', ' + str(2*pair+1)
            plt.subplot(panel)
            plt.imshow(arr, origin='upper', interpolation='nearest') # Just a histogram
            plt.title(pantit,color='r',fontsize=20) # pars like in class Text

            #subp1 = self.fig1.add_subplot(panel)
            #subp1.imshow(arr, origin='upper', interpolation='nearest') # Just a histogram
            #subp1.title(pantit,color='r',fontsize=20) # pars like in class Text

            #plt.xlabel('X pixels')
            #plt.ylabel('Y pixels')
        
            #plt.margins(x=0.05,y=0.05,tight=True)
            #plt.rc('lines', linewidth=2, color='r') # Set the current default parameters
        
            #plt.savefig("my-image-hdf5.png")

            if pair==0 :
                str_event = 'Event ' + str(cp.confpars.eventCurrent)
                #subp1.text(370, -10, str_event, fontsize=24)
                plt.text(370, -10, str_event, fontsize=24)


    def plotCSpadQuad8SpectraOf2x1( self, arr1quad, fig ):
        """Amplitude specra from 2d array."""

        fig.canvas.set_window_title('CSpad Quad Specra of 2x1')
        plt.clf() # clear plot
        #plt.title('Spectra',color='r',fontsize=20)
        fig.subplots_adjust(left=0.10, bottom=0.05, right=0.98, top=0.95, wspace=0.2, hspace=0.1)

        t_start = time.clock()
        
        for pair in xrange(8): # loop for pair = 0,1,2,...,7

            #print 20*'=',' Pair =', pair

            asic1x2  = arr1quad[pair,...]
            #print 'asic1x2.shape =', asic1x2.shape
            arrdimX,arrdimY = asic1x2.shape
            asic1d = asic1x2

            #asic1d.shape=(arrdimX*arrdimY,1)  
            asic1d.resize(arrdimX*arrdimY)            

            plt.subplot(421+pair)
            plt.hist(asic1d, bins=cp.confpars.cspadSpectrumNbins, range=(cp.confpars.cspadSpectrumAmin,cp.confpars.cspadSpectrumAmax))

            xmin, xmax = plt.xlim()
            plt.xticks( arange(int(xmin), int(xmax), int((xmax-xmin)/3)) )
            ymin, ymax = plt.ylim()
            plt.yticks( arange(int(ymin), int(ymax), int((ymax-ymin)/3)) )

            pantit='ASIC ' + str(2*pair) + ', ' + str(2*pair+1)
            ax = plt.gca()
            plt.text(0.04,0.84,pantit,color='r',fontsize=20,transform = ax.transAxes)

            if pair==0 :
                str_event = 'Event ' + str(cp.confpars.eventCurrent)
                plt.text(0.8,1.05,str_event,color='b',fontsize=24,transform = ax.transAxes)

        print 'Time to generate all histograms (sec) = %f' % (time.clock() - t_start)


    def plotCSpadQuad16Spectra( self, arr1quad, fig ):
        """Amplitude specra from 2d array."""

        fig.canvas.set_window_title('CSpad Quad Specra of 2x1')
        plt.clf() # clear plot
        #plt.title('Spectra',color='r',fontsize=20)
        fig.subplots_adjust(left=0.10, bottom=0.05, right=0.98, top=0.95, wspace=0.35, hspace=0.3)

        t_start = time.clock()
        
        for pair in xrange(8): # loop for pair = 0,1,2,...,7
            #print 20*'=',' Pair =', pair
            asic1x2  = arr1quad[pair,...]
            #print 'asic1x2.shape =', asic1x2.shape
            arrdimX,arrdimY = asic1x2.shape
            asic1d = asic1x2
            #print 'asic1d.shape =', asic1d.shape
            #asic1d.shape=(arrdimX*arrdimY,1)  
            asic1d.resize(arrdimX*arrdimY)            
            
            asics=hsplit(asic1d,2)

            for inpair in xrange(2) :
                asic = asics[inpair]
                #print 'asic.shape =', asic.shape
                plt.subplot(4,4,2*pair+inpair+1)
                #plt.xticks( arange(4), rotation=17 )
                #plt.yticks( arange(4) )
                #plt.hist(asic, bins=50, range=(0,1000))
                Amin  = cp.confpars.cspadSpectrumAmin
                Amax  = cp.confpars.cspadSpectrumAmax
                plt.hist(asic, bins=cp.confpars.cspadSpectrumNbins,range=(Amin,Amax))

                xmin, xmax = plt.xlim()
                plt.xticks( arange(int(xmin), int(xmax), int((xmax-xmin)/3)) )
                ymin, ymax = plt.ylim()
                plt.yticks( arange(int(ymin), int(ymax), int((ymax-ymin)/3)) )

                pantit='ASIC ' + str(2*pair+inpair)
                plt.title(pantit,color='r',fontsize=20)

                if pair==0 and inpair==1:
                    str_event = 'Event ' + str(cp.confpars.eventCurrent)
                    #ax = plt.gca()
                    plt.text(0.8,1.08,str_event,color='b',fontsize=24,transform = plt.gca().transAxes)

        print 'Time to generate all histograms (sec) = %f' % (time.clock() - t_start)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
