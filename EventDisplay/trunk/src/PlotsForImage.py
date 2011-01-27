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
        #self.fig1_window_is_open = False 

    #-------------------
    #  Public methods --
    #-------------------
  
    def plotImage( self, arr2d1ev, fig ):
        """Plot 2d image from input array."""

        fig.canvas.set_window_title("Image") 

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
