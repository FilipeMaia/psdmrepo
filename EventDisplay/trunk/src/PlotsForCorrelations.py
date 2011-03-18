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
        Ydsname   = cp.confpars.correlationWindowParameters[win][0] 
        Xdsname   = cp.confpars.correlationWindowParameters[win][1] 
        radioXPar = cp.confpars.correlationWindowParameters[win][2] 
        YParName  = cp.confpars.correlationWindowParameters[win][7] 
        XParName  = cp.confpars.correlationWindowParameters[win][8] 
        #YParInd   = cp.confpars.correlationWindowParameters[win][9] 
        #XParInd   = cp.confpars.correlationWindowParameters[win][10] 

        nYpoints = dsY.shape[0]
        print '\nNew Win:', win, ' Correlation plot for nYpoints =', nYpoints 

        self.Yarr = dsY[YParName]
        print 'Y-Parameter array :\n', self.Yarr

        self.markerStyle = 'bs-'

        if   radioXPar == 0 : # for Index
            self.Xarr = range(nYpoints)
            print 'Index array from 0 to', nYpoints
            self.XTitle = 'Index'
            self.PlotTitle = 'Parameter vs Index'
            
        elif radioXPar == 1 : # for Time
            self.Xarr = 0.000000001 * dsX['nanoseconds'] + dsX['seconds']
            self.Xarr -= self.Xarr[0]
            print 'Time array :\n', self.Xarr 
            self.XTitle = 'Time (sec)'
            self.PlotTitle = 'Parameter vs Time'

        elif radioXPar == 2 : # for X-Parameter
            self.Xarr = dsX[XParName]
            print 'X-Parameter array :\n', self.Xarr
            self.XTitle = XParName
            self.PlotTitle = 'Correlations of two parameters'
            self.markerStyle = 'bo-'

        plt.clf()
        plt.plot(self.Xarr, self.Yarr, self.markerStyle, markersize=2)
        plt.ylabel(YParName)
        plt.xlabel(self.XTitle)
        plt.title(self.PlotTitle,color='r',fontsize=20) # pars like in class Text
        fig.canvas.set_window_title(Ydsname)
 
        plt.show()



        

        
#--------------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    sys.exit ( "Module is not supposed to be run as main module" )

#--------------------------------
