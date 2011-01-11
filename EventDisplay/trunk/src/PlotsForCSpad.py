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

        self.fig1_window_is_open = False

    #-------------------
    #  Public methods --
    #-------------------


    def open_fig1( self ):
        """Open window for fig1."""

        print 'open_fig1()'

        plt.ion() # enables interactive mode
        fig1 = plt.figure(figsize=(10,10), dpi=80, facecolor='w',edgecolor='w',frameon=True) # parameters like in class Figure
        plt.subplots_adjust(left=0.08, bottom=0.02, right=0.98, top=0.98, wspace=0.2, hspace=0.1)

        fig1.canvas.set_window_title("CSpad image") 
        ##f = fig.Figure(figsize=(2,5), dpi=100, facecolor='w',edgecolor='w') #,frameon=True,linewidth=0.05) # set figure parame ters
        ##plt.figure(figsize=(10,6), dpi=100, facecolor='g',edgecolor='b',frameon=True,linewidth=5) # parameters like in class Figure
        ##plt.subplots_adjust(hspace=0.4)
        ##plt.subplot(221)
        ##plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        self.fig1_window_is_open = True


    def close_fig1( self ):
        """Close fig1 and its window."""

        if self.fig1_window_is_open :
            plt.ioff()
            plt.close()
            self.fig1_window_is_open = False 
            print 'close_fig1()'

    def plotCSpadV1( self, arr1ev, mode=1 ):
        """Plot 2d image from input array. V1 for run ~546

        V1 contain array for entire detector. Currently we plat quad 2 only.
        """
        quad=2
        arr1quad = arr1ev[quad,...]      # V1 for run ~546
        self.plotCSpadQuad( arr1quad, mode )


    def plotCSpadV2( self, arr1quad, mode=1 ):
        """Plot 2d image from input array.

        V2 for run ~900 contain array for quad 2, which we plot directly.
        """
        self.plotCSpadQuad( arr1quad, mode )

  
    def plotCSpadQuad( self, arr1quad, mode=1 ):
        """Plot 2d image from input array."""

        #print 'plot_CSpadQuad()'       

        if not self.fig1_window_is_open :
            self.open_fig1()

        plt.clf() # clear plot
        
        arrgap=zeros( (185,4) ) # make additional 2D-array of 0-s for the gap between two 1x1 pads
        
        pair=7
        
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
            #plt.xlabel('X pixels')
            #plt.ylabel('Y pixels')
        
           ##plt.ion() # turn interactive mode on
           ##plt.margins(x=0.05,y=0.05,tight=True)
           ##plt.rc('lines', linewidth=2, color='r') # Set the current default parameters
        
           ##plt.savefig("my-image-hdf5.png")
           ##plt.show()

            if pair==0 :
                str_event = 'Event ' + str(cp.confpars.eventCurrent)
                plt.text(370, -10, str_event, fontsize=24)
        
        if mode == 1 :   # Single event mode
            plt.show()  
        else :           # Slide show 
            plt.draw()   # Draws, but does not block
            #plt.draw()   # Draws, but does not block

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
