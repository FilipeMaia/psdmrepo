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
import PrintHDF5        as printh5

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
  

    def plotCorrelations ( self, fig, h5file ) :

        self.h5file = h5file
        self.fig    = fig
        
        win = fig.nwin
        self.Ydsname   = cp.confpars.correlationWindowParameters[win][0] 
        self.Xdsname   = cp.confpars.correlationWindowParameters[win][1] 
        self.radioXPar = cp.confpars.correlationWindowParameters[win][2] 
        self.Ymin      = cp.confpars.correlationWindowParameters[win][3]
        self.Ymax      = cp.confpars.correlationWindowParameters[win][4]
        self.Xmin      = cp.confpars.correlationWindowParameters[win][5]
        self.Xmax      = cp.confpars.correlationWindowParameters[win][6]
        self.YParName  = cp.confpars.correlationWindowParameters[win][7] 
        self.XParName  = cp.confpars.correlationWindowParameters[win][8] 
        self.YLimsIsOn = cp.confpars.correlationWindowParameters[win][9] 
        self.XLimsIsOn = cp.confpars.correlationWindowParameters[win][10] 

        if self.Ydsname == 'None' :
            print 'THE Ydsname=', self.Ydsname, ' IS SET INCORRECTLY. THE CORRELATION PLOT', self.nwin,' IS IGNORED'
            return


        self.dsY = h5file[self.Ydsname]
        print 'dsY.shape=',self.dsY.shape

        nYpoints = self.dsY.shape[0]
        print '\nNew Win:', win, ' Correlation plot for nYpoints =', nYpoints 

        self.Yarr = self.dsY[self.YParName]
        print 'Y-Parameter array :\n', self.Yarr

        self.markerStyle = 'bs-'

        if   self.radioXPar == 0 : # for Index
            self.Xarr = range(nYpoints)
            print 'Index array from 0 to', nYpoints
            self.XTitle = 'Index'
            self.PlotTitle = 'Parameter vs Index'
            
        elif self.radioXPar == 1 : # for Time
            self.Xdsname  = printh5.get_item_path_to_last_name(self.Ydsname) + '/time'
            print 'Xdsname =',self.Xdsname 
            self.dsX = h5file[self.Xdsname]
            self.Xarr = 0.000000001 * self.dsX['nanoseconds'] + self.dsX['seconds']
            self.Xarr -= self.Xarr[0]
            print 'Time array :\n', self.Xarr 
            self.XTitle = 'Time (sec)'
            self.PlotTitle = 'Parameter vs Time'

        elif self.radioXPar == 2 : # for X-Parameter
            if self.Xdsname == 'None' :
                print 'THE Xdsname=', self.Xdsname, ' IS SET INCORRECTLY. THE CORRELATION PLOT', win,' IS IGNORED' 
                return
            self.dsX = h5file[self.Xdsname]

            self.Xarr = self.dsX[self.XParName]

            if self.dsX.shape[0] != self.dsY.shape[0] :
                print 'Arrays of different length, X, Y shape=', self.dsX.shape[0], self.dsY.shape[0]
                #print 'THE CORRELATION PLOT', win,' IS IGNORED' 
                #return

                self.mapCorrelatingArraysByTime()

            print 'X-Parameter array :\n', self.Xarr
            self.XTitle = self.XParName
            self.PlotTitle = 'Correlations of two parameters'
            self.markerStyle = 'bo-'


        elif self.radioXPar == 3 : # for Y-Parameter histogram 
            self.PlotTitle = 'Y-parameter histogram'
            self.plotHistogram() 

        if self.radioXPar != 3 : self.plotWaveform() 

        plt.show()


    def plotWaveform( self ) :
        plt.clf()

        plt.xlim(auto=True)
        plt.ylim(auto=True)

        plt.plot(self.Xarr, self.Yarr, self.markerStyle, markersize=2)

        if self.XLimsIsOn : plt.xlim(self.Xmin,self.Xmax)
        if self.YLimsIsOn : plt.ylim(self.Ymin,self.Ymax)

        plt.ylabel(self.YParName)
        plt.xlabel(self.XTitle)
        plt.title(self.PlotTitle,color='r',fontsize=20) # pars like in class Text
        self.fig.canvas.set_window_title(self.Ydsname)




    def plotHistogram( self ) :

        plt.xlim(auto=True)
        axes = plt.hist(self.Yarr, bins=100, color='b')
        if self.YLimsIsOn : plt.xlim(self.Ymin,self.Ymax)
        plt.xlabel(self.YParName)
        plt.title(self.PlotTitle,color='r',fontsize=20) # pars like in class Text
        self.fig.canvas.set_window_title(self.Ydsname)        


    def mapCorrelatingArraysByTimeInit( self ) :

        #self.Xarr # is assumed to be available
        #self.Yarr # is assumed to be available
        XTimedsname  = printh5.get_item_path_to_last_name(self.Xdsname) + '/time'
        YTimedsname  = printh5.get_item_path_to_last_name(self.Ydsname) + '/time'
        print 'Xdsname =',self.Xdsname 
        print 'Ydsname =',self.Ydsname 
        self.dsXT = self.h5file[XTimedsname]
        self.dsYT = self.h5file[YTimedsname]
        self.XTarr = 0.000000001 * self.dsXT['nanoseconds'] + self.dsXT['seconds']
        self.YTarr = 0.000000001 * self.dsYT['nanoseconds'] + self.dsYT['seconds']
        print 'self.XTarr =', self.XTarr
        print 'self.YTarr =', self.YTarr
        self._nXpoints = self.dsX.shape[0]
        self._nYpoints = self.dsY.shape[0]
        self._indX=0
        self._indY=0
        self._tmapXlist = []
        self._tmapYlist = []
        print 'mapCorrelatingArraysByTimeInit :map arrays of different length by time, X, Y length=', self._nXpoints, self._nYpoints



    def mapCorrelatingArraysByTimeIterations( self ) :

        while self._indX < self._nXpoints and self._indY < self._nYpoints :

            if self.XTarr[self._indX] == self.YTarr[self._indY] :   # Time is the same
                self._tmapXlist.append(self.Xarr[self._indX])
                self._tmapYlist.append(self.Yarr[self._indY])
                self._indX += 1
                self._indY += 1

            elif self.XTarr[self._indX] > self.YTarr[self._indY] :  # Time X > Time Y
                self._indY += 1            

            else :                                                  # Time X < Time Y
                self._indX += 1            



    def mapCorrelatingArraysByTimeSummary( self ) :
        self.Xarr = np.array(self._tmapXlist)
        self.Yarr = np.array(self._tmapYlist)
        print 'Number of synchronized in time array elements =', self.Xarr.shape


    def mapCorrelatingArraysByTime ( self ) :
        self.mapCorrelatingArraysByTimeInit()
        self.mapCorrelatingArraysByTimeIterations()
        self.mapCorrelatingArraysByTimeSummary()
        pass


    

        
#--------------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    sys.exit ( "Module is not supposed to be run as main module" )

#--------------------------------
