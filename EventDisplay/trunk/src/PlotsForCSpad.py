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
import time   # for sleep(sec)
from numpy import *  # for use like       array(...)

#---------------------------------
#  Imports of base class module --
#---------------------------------

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
    def __init__ ( self, x=0, y=0 ) :
        """Constructor initialization.

        @param x   first parameter
        @param y   second parameter
        """

        print '\n Initialization of the PlotsForCSpad'         
        print 'using MPL version: ', matplotlib.__version__


        self.window_is_open = False
        # define instance variables
        #self.__x = x                  # private 
        #self._p = None                # "protected"
        #self.y = y                    # public

    #-------------------
    #  Public methods --
    #-------------------

    def open_fig1():
        """Open window for fig1."""

        plt.ion() # enables interactive mode
        plt.figure(figsize=(10,10), dpi=80, facecolor='w',edgecolor='w',frameon=True) # parameters like in class Figure
        plt.subplots_adjust(left=0.08, bottom=0.02, right=0.98, top=0.98, wspace=0.1, hspace=0.1)
        #f = fig.Figure(figsize=(2,5), dpi=100, facecolor='w',edgecolor='w') #,frameon=True,linewidth=0.05) # set figure parame ters
        #plt.figure(figsize=(10,6), dpi=100, facecolor='g',edgecolor='b',frameon=True,linewidth=5) # parameters like in class Figure
        #plt.subplots_adjust(hspace=0.4)
        #plt.subplot(221)
        #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)


    def close_fig1():
        """Close fig1 and its window"""

        plt.ioff()
        plt.close()
 

    def plot_CSpad(eventN,arr1ev):
        """Plot 2d image from input array."""
        
        if( not self.window_is_open ) :
            open_fig1()

        plt.clf() # clear plot
        
        arrgap=zeros( (185,4) ) # make additional 2D-array of 0-s for the gap between two 1x1 pads
        
        quad=2
        pair=7
        
        arr1quad = arr1ev[quad,...]
        
        for pair in xrange(8): # loop for pair = 0,1,2,...,7
            #print 'pair=', pair
        
            asic1x2  = arr1quad[pair,...]
            #asic1x2  = arr1ev[quad,pair,...]
            #print 'asic1x2=',asic1x2    
        
            asics    = hsplit(asic1x2,2)
            arr      = hstack((asics[0],arrgap,asics[1]))
        
            panel = 421+pair
            pantit='ASIC'+str(pair)
            plt.subplot(panel)
            plt.imshow(arr, origin='upper', interpolation='nearest') # Just a histogram
            plt.title(pantit,color='r',fontsize=24) # pars like in class Text
            plt.xlabel('X pixels')
            plt.ylabel('Y pixels')
        
           #plt.ion() # turn interactive mode on
           #plt.margins(x=0.05,y=0.05,tight=True)
           #plt.rc('lines', linewidth=2, color='r') # Set the current default parameters
        
           #plt.savefig("my-image-hdf5.png")
           #plt.show()

        event = cp.confpars.eventCurrent
        str_evN = 'Event No.' + str(event)
        plt.text(10, -10, str_evN, fontsize=24)
        
        plt.draw()   # Draws, but does not block
        plt.draw()   # Draws, but does not block



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
