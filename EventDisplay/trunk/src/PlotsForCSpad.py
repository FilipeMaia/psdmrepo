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


    def plotCSpadV1( self, arr1ev, fig ):
        """Plot 2d image from input array. V1 for run ~546

        V1 contain array for entire detector. Currently we plat quad 2 only.
        """
        quad=2
        arr1quad = arr1ev[quad,...]      # V1 for run ~546
        self.plotCSpadQuad( arr1quad, fig )


    def plotCSpadV2( self, arr1quad, fig ):
        """Plot 2d image from input array.

        V2 for run ~900 contain array for quad 2, which we plot directly.
        """
        self.plotCSpadQuad( arr1quad, fig )

  
    def plotCSpadQuad( self, arr1quad, fig ):
        """Plot 2d image from input array."""

        #print 'plot_CSpadQuad()'       

        fig.canvas.set_window_title("CSpad image")
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

            #subp1 = self.fig1.add_subplot(panel)
            #subp1.imshow(arr, origin='upper', interpolation='nearest') # Just a histogram
            #subp1.title(pantit,color='r',fontsize=20) # pars like in class Text

            #plt.xlabel('X pixels')
            #plt.ylabel('Y pixels')
        
           ##plt.ion() # turn interactive mode on
           ##plt.margins(x=0.05,y=0.05,tight=True)
           ##plt.rc('lines', linewidth=2, color='r') # Set the current default parameters
        
           ##plt.savefig("my-image-hdf5.png")
           ##plt.show()

            if pair==0 :
                str_event = 'Event ' + str(cp.confpars.eventCurrent)
                #subp1.text(370, -10, str_event, fontsize=24)
                plt.text(370, -10, str_event, fontsize=24)
        
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
