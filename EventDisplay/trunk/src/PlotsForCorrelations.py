#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotsForCorrelations...
#
#------------------------------------------------------------------------

"""Plots for correlations.

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
import numpy as np

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
class PlotsForCorrelations ( object ) :
    """Plots for correlations."""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor, initialization"""
        pass

    #-------------------
    #  Public methods --
    #-------------------
  

    def plotCorrelations ( self, fig, dsY, dsX=None ) :
        
        win = fig.nwin
        radioXPar = cp.confpars.correlationWindowParameters[win][2] 

        nYpoints = dsY.shape[0]


        print '\nNew Win:', win, ' Correlation plot for nYpoints =', nYpoints 


        #self.arr1ev = ds[cp.confpars.eventCurrent]

        print 'dsY=',dsY

        Yarr = np.zeros( (nYpoints), dtype=np.int16 )
        for ind in range (nYpoints) :
            #print ind, dsY[ind][5] 
            Yarr[ind] = dsY[ind][5]

        Xarr = range(nYpoints)

        plt.clf()
        plt.plot(Xarr, Yarr, 'b-')
        plt.show()





    def plotWFWaveform( self, ds1ev, fig ):
        """Plot waveform from input array."""

        print 'plotWFWaveform'
        print 'ds1ev.shape', ds1ev.shape
        numberOfWF, par2, dimX = ds1ev.shape
        #print 'numberOfWF, par2, dimX = ', numberOfWF, par2, dimX
        #print 'arrwf.shape', arrwf.shape

        nwin = fig.nwin
        indwf1 = cp.confpars.waveformWindowParameters[nwin][7]
        indwf2 = cp.confpars.waveformWindowParameters[nwin][8]
        indwf3 = cp.confpars.waveformWindowParameters[nwin][9]
        indwf4 = cp.confpars.waveformWindowParameters[nwin][10]

        arrT = range(dimX)
        #arrZ = zeros( (dimX) )

        par = []

        if indwf1 != None : par.append( (arrT, ds1ev[indwf1,0,...], 'k-') )
        if indwf2 != None : par.append( (arrT, ds1ev[indwf2,0,...], 'r-') )
        if indwf3 != None : par.append( (arrT, ds1ev[indwf3,0,...], 'g-') )
        if indwf4 != None : par.append( (arrT, ds1ev[indwf4,0,...], 'b-') )

        fig.canvas.set_window_title(cp.confpars.current_item_name_for_title) 
        plt.clf() # clear plot
        fig.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.94, wspace=0.1, hspace=0.1)        

        print 'Number of waves to draw =', len(par) 

        if len(par) == 1 :
            plt.plot( par[0][0],par[0][1],par[0][2] )

        elif len(par) == 2 :
            plt.plot( par[0][0],par[0][1],par[0][2],\
                      par[1][0],par[1][1],par[1][2] )

        elif len(par) == 3 :
            plt.plot( par[0][0],par[0][1],par[0][2],\
                      par[1][0],par[1][1],par[1][2],\
                      par[2][0],par[2][1],par[2][2] )
            
        elif len(par) == 4 :
            plt.plot( par[0][0],par[0][1],par[0][2],\
                      par[1][0],par[1][1],par[1][2],\
                      par[2][0],par[2][1],par[2][2],\
                      par[3][0],par[3][1],par[3][2] )
        else :
            print 'Wrong number of waves !!!', len(par) 

        autoRangeIsOn = cp.confpars.waveformWindowParameters[nwin][1]

        if autoRangeIsOn : pass
        else :
            plt.xlim(cp.confpars.waveformWindowParameters[nwin][4],\
                     cp.confpars.waveformWindowParameters[nwin][5]  )

            plt.ylim(cp.confpars.waveformWindowParameters[nwin][2],\
                     cp.confpars.waveformWindowParameters[nwin][3]  )
            


        
        str_title='Waveform, event ' + str(cp.confpars.eventCurrent)
        
        plt.title(str_title,color='r',fontsize=20) # pars like in class Text
        ##plt.xlabel('X pixels')
        ##plt.ylabel('Y pixels')
        
#--------------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    sys.exit ( "Module is not supposed to be run as main module" )

#--------------------------------
