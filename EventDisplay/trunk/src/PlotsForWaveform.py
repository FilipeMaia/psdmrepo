#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotsForWaveform...
#
#------------------------------------------------------------------------

"""Plots for any 'waveform' record in the EventeDisplay project.

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
class PlotsForWaveform ( object ) :
    """Plots for any 'waveform' record in the EventeDisplay project."""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor, initialization"""
        pass

    #-------------------
    #  Public methods --
    #-------------------
  
    def plotWFWaveform( self, ds1ev, fig ):
        """Plot waveform from input array."""

        print 'plotWFWaveform'

        numberOfWF, par2, dimX = ds1ev.shape
        print 'numberOfWF, par2, dimX = ', numberOfWF, par2, dimX
        arrwf = ds1ev[0,0,...]
        
        print 'arrwf.shape', arrwf.shape

        fig.canvas.set_window_title(cp.confpars.current_item_name_for_title) 
        plt.clf() # clear plot
        fig.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)        

        plt.plot(arrwf, range(dimX), 'b-')
        
        pantit='Waveform, event ' + str(cp.confpars.eventCurrent)
        #self.axes = plt.imshow(arr2d1ev, origin='upper', interpolation='nearest') # Just a histogram
        #self.colb = plt.colorbar(self.axes, pad=0.01, fraction=0.10, shrink = 1) #, ticks=coltickslocs)
        #plt.clim(cp.confpars.imageImageAmin,cp.confpars.imageImageAmax)
        
        plt.title(pantit,color='r',fontsize=20) # pars like in class Text
        ##plt.xlabel('X pixels')
        ##plt.ylabel('Y pixels')
        
        ##plt.margins(x=0.05,y=0.05,tight=True)
        ##plt.rc('lines', linewidth=2, color='r') # Set the current default parameters
        
        ##str_event = 'Event ' + str(cp.confpars.eventCurrent)
        ##plt.text(-50, -10, str_event, fontsize=24)

        ##plt.savefig("my-image-hdf5.png")
        ##plt.show()        


    def plotWFSpectrum( self, arr2d1ev, fig ):
        """Spectrum of amplitudes in the 2d input array."""

        #plt.clf() # clear plot
        #fig.canvas.set_window_title(cp.confpars.current_item_name_for_title) 
        #pantit='Specrum, event ' + str(cp.confpars.eventCurrent)
        #plt.title(pantit,color='r',fontsize=20) # pars like in class Text
        #arrdimX,arrdimY = arr2d1ev.shape
        ##print 'arr2d1ev.shape=', arr2d1ev.shape, arrdimX, arrdimY 
        #print 'arr2d1ev=\n', arr2d1ev
        #arr1d1ev = copy(arr2d1ev)
        #arr1d1ev.resize(arrdimX*arrdimY)
        #print 'arr1d1ev=\n', arr1d1ev
        ##plt.hist(arr1d1ev,100)

        #cp.confpars.imageSpectrumRange=(15,45)
        ##cp.confpars.imageSpectrumNbins=30       
        ##cp.confpars.imageSpectrumRange=None        
        ##cp.confpars.imageSpectrumNbins=None        
        #plt.hist(arr1d1ev, bins=cp.confpars.imageSpectrumNbins, range=(cp.confpars.imageSpectrumAmin,cp.confpars.imageSpectrumAmax))
        ##plt.hist(arr1d1ev)


#--------------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    sys.exit ( "Module is not supposed to be run as main module" )

#--------------------------------
