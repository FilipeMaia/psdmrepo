#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotsForImage...
#
#------------------------------------------------------------------------

"""Plots for any 'image' record in the EventeDisplay project.

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
from numpy import *

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
class PlotsForImage ( object ) :
    """Plots for any 'image' record in the EventeDisplay project.

    @see BaseClass
    @see OtherClass
    """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor, initialization
        """
        self.fig1_window_is_open = False 

    #-------------------
    #  Public methods --
    #-------------------

    def open_fig1( self ):
        """Open window for fig1.
        """
        if not self.fig1_window_is_open :
            print 'open_fig1()'
            plt.ion() # enables interactive mode
            self.fig1 = plt.figure(figsize=(6,5), dpi=80, facecolor='w',edgecolor='w',frameon=True) # parameters like in class Figure
            #plt.subplots_adjust(left=0.08, bottom=0.02, right=0.98, top=0.98, wspace=0.2, hspace=0.1)
            self.fig1.canvas.set_window_title("CSpad image") 
            self.fig1_window_is_open = True


    def close_fig1( self ):
        """Close fig1 and its window.
        """

        if self.fig1_window_is_open :
            #plt.ioff()
            #plt.close()
            self.fig1_window_is_open = False 
            print 'close_fig1()'

  
    def plotImage( self, arr2d1ev, mode=1  ):
        """Plot 2d image from input array.
        """

        self.open_fig1()

        plt.clf() # clear plot
        
        pantit='Image'
        plt.imshow(arr2d1ev, origin='upper', interpolation='nearest') # Just a histogram
        plt.title(pantit,color='r',fontsize=20) # pars like in class Text
        plt.xlabel('X pixels')
        plt.ylabel('Y pixels')
        
        #plt.ion() # turn interactive mode on
        #plt.margins(x=0.05,y=0.05,tight=True)
        #plt.rc('lines', linewidth=2, color='r') # Set the current default parameters
        
        #plt.savefig("my-image-hdf5.png")
        #plt.show()

        str_event = 'Event ' + str(cp.confpars.eventCurrent)
        plt.text(50, -10, str_event, fontsize=24)
        
        #if mode == 1 :   # Single event mode
        #    plt.show()  
        #else :           # Slide show 
        #    plt.draw()   # Draws, but does not block

#--------------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    sys.exit ( "Module is not supposed to be run as main module" )

#--------------------------------
