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

        nwin = fig.nwin_waveform
        indwf1 = cp.confpars.waveformWindowParameters[nwin][7]
        indwf2 = cp.confpars.waveformWindowParameters[nwin][8]
        indwf3 = cp.confpars.waveformWindowParameters[nwin][9]
        indwf4 = cp.confpars.waveformWindowParameters[nwin][10]

        if indwf1 == None : self.arrwf1 = zeros( (dimX) ) # from numpy
        else :              self.arrwf1 = ds1ev[indwf1,0,...]

        if indwf2 == None : self.arrwf2 = zeros( (dimX) ) # from numpy
        else :              self.arrwf2 = ds1ev[indwf2,0,...]

        if indwf3 == None : self.arrwf3 = zeros( (dimX) ) # from numpy
        else :              self.arrwf3 = ds1ev[indwf3,0,...]

        if indwf4 == None : self.arrwf4 = zeros( (dimX) ) # from numpy
        else :              self.arrwf4 = ds1ev[indwf4,0,...]

        arrT = range(dimX)

        fig.canvas.set_window_title(cp.confpars.current_item_name_for_title) 
        plt.clf() # clear plot
        fig.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.94, wspace=0.1, hspace=0.1)        

        plt.plot( arrT, self.arrwf1, 'k-', arrT, self.arrwf2, 'r-', arrT, self.arrwf3, 'g-', arrT, self.arrwf4, 'b-')

        #plt.plot( range(dimX)arrT, arrwf, 'b-')
        
        str_title='Waveform, event ' + str(cp.confpars.eventCurrent)
        #plt.clim(cp.confpars.imageImageAmin,cp.confpars.imageImageAmax)
        
        plt.title(str_title,color='r',fontsize=20) # pars like in class Text
        ##plt.xlabel('X pixels')
        ##plt.ylabel('Y pixels')
        
#--------------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    sys.exit ( "Module is not supposed to be run as main module" )

#--------------------------------
