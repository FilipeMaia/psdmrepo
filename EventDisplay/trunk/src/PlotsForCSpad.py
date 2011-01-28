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
#import time   # for sleep(sec)
from numpy import *  # for use like       array(...)

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

    def plotCSpadV1Spectrum( self, arr1ev, fig ):
        """Plot 2d image from input array. V1 for run ~546, array for entire detector"""
        quad=2
        arr1quad = arr1ev[quad,...] 
        #self.plotCSpadQuadSpectrum( arr1quad, fig )
        self.plotCSpadQuad8SpectraOf2x1( arr1quad, fig )

    def plotCSpadV2Image( self, arr1quad, fig ):
        """Plot 2d image from input array. V2 for run ~900 contain array for quad 2"""
        self.plotCSpadQuadImage( arr1quad, fig )


    def plotCSpadV2Spectrum( self, arr1quad, fig ):
        """Plot 2d image from input array. V2 for run ~900 contain array for quad 2"""
        #self.plotCSpadQuadSpectrum( arr1quad, fig )
        self.plotCSpadQuad8SpectraOf2x1( arr1quad, fig )

  

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
        
        for pair in xrange(8): # loop for pair = 0,1,2,...,7

            asic1x2  = arr1quad[pair,...]
            #print 'asic1x2.shape =', asic1x2.shape
            arrdimX,arrdimY = asic1x2.shape
            asic1d = asic1x2
            #asic1d.shape = (1,arrdimX*arrdimY)
            asic1d.resize(arrdimX*arrdimY)            
            panel = 421+pair
            plt.subplot(panel)
            plt.hist(asic1d, bins=cp.confpars.cspadSpectrumNbins, range=(cp.confpars.cspadSpectrumAmin,cp.confpars.cspadSpectrumAmax))
            pantit='ASIC ' + str(2*pair) + ', ' + str(2*pair+1)
            ax = plt.gca()
            plt.text(0.04,0.84,pantit,color='r',fontsize=20,transform = ax.transAxes)

            if pair==0 :
                str_event = 'Event ' + str(cp.confpars.eventCurrent)
                plt.text(0.8,1.05,str_event,color='b',fontsize=24,transform = ax.transAxes)


    def plotCSpadQuadSpectrum( self, arr1quad, fig ):
        """Amplitude specra from 2d array."""

        fig.canvas.set_window_title("CSpad quad spectra")
        plt.clf() # clear plot

        pantit='Specra, event ' + str(cp.confpars.eventCurrent)
        plt.title(pantit,color='r',fontsize=20) # pars like in class Text
        
        for pair in xrange(8): # loop for pair = 0,1,2,...,7
            #print 'pair=', pair
        
            asic1x2  = arr1quad[pair,...]
            print 'asic1x2.shape =', asic1x2.shape

            arrdimX,arrdimY = asic1x2.shape
            #asic1d = asic1x2.resize(arrdimX*arrdimY)
            asic1d = asic1x2
            asic1d.shape = (1,arrdimX*arrdimY)
            asic1d.resize(arrdimX*arrdimY) 
            
            print 'asic1d =\n',asic1d
            panel = 421+pair
            plt.subplot(panel)
            plt.hist(asic1d, bins=50, range=(0,1000))
            pantit='ASIC ' + str(2*pair) + ', ' + str(2*pair+1)
            plt.title(pantit,color='r',fontsize=20) # pars like in class Text
                
            #asics = hsplit(asic1x2,2)

            #for inpair in xrange(2) :
                #asic = asics[inpair]
                #print 'asic.shape =', asic.shape
                #arrdimX,arrdimY = asic.shape
                #asic1d = asic.resize(arrdimX*arrdimY)
                #asic1d = asic.resize(arrdimX*arrdimY)
                #print asic1d


                #panel = 441+pair+inpair
                #plt.subplot(panel)

                #plt.hist(asic, bins=50, range=(0,1000))

                #pantit='ASIC ' + str(2*pair) + ', ' + str(2*pair+1)
                #plt.title(pantit,color='r',fontsize=20) # pars like in class Text

                #if pair==0 :
                #    str_event = 'Event ' + str(cp.confpars.eventCurrent)
                #    plt.text(370, -10, str_event, fontsize=24)


        
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
