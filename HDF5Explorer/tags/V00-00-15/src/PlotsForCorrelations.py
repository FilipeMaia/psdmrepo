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
        self.LogIsOn   = cp.confpars.correlationWindowParameters[win][11]
        self.YNBins    = cp.confpars.correlationWindowParameters[win][12]
        self.XNBins    = cp.confpars.correlationWindowParameters[win][13]
        self.YParIndex = cp.confpars.correlationWindowParameters[win][14] 
        self.XParIndex = cp.confpars.correlationWindowParameters[win][15] 
        
        if self.Ydsname == 'None' :
            print 'THE Ydsname=', self.Ydsname, ' IS SET INCORRECTLY. THE CORRELATION PLOT', win,' IS IGNORED'
            return

        if self.YParName == 'None' : 
            print 'THE YParName=', self.YParName, ' IS NOT SET. THE CORRELATION PLOT', win,' IS IGNORED'
            return

        try    : self.dsY = h5file[self.Ydsname]
        except :
            self.printWarningNonAvailable(self.Ydsname)
            return

        print 'dsY.shape=',self.dsY.shape

        if self.YParIndex == 'None' :
            self.Yarr = self.dsY[self.YParName]
        else :
            self.Yarr = self.dsY[self.YParName][self.YParIndex]  # Next level index for ipimb...       

        print 'Y-Parameter array :\n', self.Yarr
        self.nYpoints = self.dsY.shape[0]
        #self.nYpoints = self.Yarr.shape[0] # also works
        print '\nNew Win:', win, ' Correlation plot for nYpoints =', self.nYpoints 


        self.markerStyle = 'bs-'

        if   self.radioXPar == 0 : # for Index
            self.Xarr = range(self.nYpoints)
            print 'Index array from 0 to', self.nYpoints
            self.XTitle = 'Index'
            self.PlotTitle = 'Plot ' + str(win+1) + ': Parameter vs Index'
            self.plotWaveform()


        elif self.radioXPar == 1 : # for Time
            self.Xdsname  = printh5.get_item_path_to_last_name(self.Ydsname) + '/time'
            print 'Xdsname =',self.Xdsname 
            try    : self.dsX = h5file[self.Xdsname]
            except :
                self.printWarningNonAvailable(self.Xdsname)
                return

            self.Xarr = 0.000000001 * self.dsX['nanoseconds'] + self.dsX['seconds']
            self.Xarr -= self.Xarr[0]
            print 'Time array :\n', self.Xarr 
            self.XTitle = 'Time (sec)'
            self.PlotTitle = 'Plot ' + str(win+1) + ': Parameter vs Time'
            self.plotWaveform()

            
        elif self.radioXPar == 2 : # for X-Parameter
            if self.Xdsname == 'None' :
                print 'THE Xdsname=', self.Xdsname, ' IS SET INCORRECTLY. THE CORRELATION PLOT', win,' IS IGNORED' 
                return

            if self.XParName == 'None' : 
                print 'THE XParName=', self.XParName, ' IS NOT SET. THE CORRELATION PLOT', win,' IS IGNORED'
                return

            try    : self.dsX = h5file[self.Xdsname]
            except :
                self.printWarningNonAvailable(self.Xdsname)
                return

            if self.XParIndex == 'None' :
                self.Xarr = self.dsX[self.XParName]
            else :
                self.Xarr = self.dsX[self.XParName][self.XParIndex] # Next level index for ipimb...


            if self.dsX.shape[0] != self.dsY.shape[0] :
                print 'Arrays of different length, X, Y shape=', self.dsX.shape[0], self.dsY.shape[0]
                #print 'THE CORRELATION PLOT', win,' IS IGNORED' 
                #return

                self.mapCorrelatingArraysByTime()

            print 'X-Parameter array :\n', self.Xarr
            self.XTitle = self.XParName
            self.PlotTitle = 'Plot ' + str(win+1) + ': Correlations of two parameters'
            self.markerStyle = 'bo-'
            #self.plotWaveform()
            self.plot2DHistogram()

 
        elif self.radioXPar == 3 : # for Y-Parameter histogram 
            self.PlotTitle = 'Plot ' + str(win+1) + ': Y-parameter histogram'
            self.plot1DHistogram() 

        plt.show()

    def printWarningNonAvailable(self, parname) :
        print     'WARNING: The parameter ' + parname \
              + '\n         is not available in current hdf5 file...' \
              + ' Check settings for this plot.'

    def setLabelX( self, parname, parindex='None' ) :
        if parindex == 'None' : xtitle = parname
        else                  : xtitle = parname + ' : ' + parindex
        plt.xlabel(xtitle)


    def setLabelY( self, parname, parindex='None' ) :
        if parindex == 'None' : ytitle = parname
        else                  : ytitle = parname + ' : ' + parindex
        plt.ylabel(ytitle)


    def plotWaveform( self ) :
        plt.clf()

        plt.xlim(auto=True)
        plt.ylim(auto=True)

        plt.plot(self.Xarr, self.Yarr, self.markerStyle, markersize=2)

        if self.XLimsIsOn : plt.xlim(self.Xmin,self.Xmax)
        if self.YLimsIsOn : plt.ylim(self.Ymin,self.Ymax)

        self.setLabelY(self.YParName, self.YParIndex)
        plt.xlabel(self.XTitle)
        plt.title(self.PlotTitle,color='r',fontsize=20) # pars like in class Text
        self.fig.canvas.set_window_title(self.Ydsname)


    def plot1DHistogram( self ) :
        plt.clf()
        plt.xlim(auto=True)
        axes = plt.hist(self.Yarr, bins=self.YNBins, color='b')
        if self.YLimsIsOn : plt.xlim(self.Ymin,self.Ymax)
        self.setLabelX(self.YParName, self.YParIndex)
        plt.title(self.PlotTitle,color='r',fontsize=20) # pars like in class Text
        self.fig.canvas.set_window_title(self.Ydsname)        



    def plot2DHistogram( self ) :
        plt.clf()
        if self.YLimsIsOn :
            self._Yrange = [self.Ymin,self.Ymax]
        else :
            self._Yrange = [self.Yarr.min(),self.Yarr.max()]

        if self.XLimsIsOn :
            self._Xrange = [self.Xmin,self.Xmax]
        else :
            self._Xrange = [self.Xarr.min(),self.Xarr.max()]

        XYNBins = (self.XNBins, self.YNBins)
        XYRange = [self._Xrange, self._Yrange]
        XYExtent = (self._Xrange[0], self._Xrange[1], self._Yrange[0], self._Yrange[1])
        
        arr2d, xedges, yedges = np.histogram2d(self.Xarr, self.Yarr, bins=XYNBins, range=XYRange) #, normed=False, weights=None) 

        if self.LogIsOn : self.arrImage = log(arr2d)
        else :            self.arrImage =     arr2d

        axes = plt.imshow(np.rot90(self.arrImage), interpolation='nearest', extent=XYExtent, aspect='auto') #, origin='upper'

        colb = plt.colorbar(axes, pad=0.005, fraction=0.10, aspect=12, shrink=1) # pad=0.10, orientation=2, aspect = 8, ticks=coltickslocs

        #plt.xlabel(self.XParName)
        #plt.ylabel(self.YParName)
        self.setLabelX(self.XParName, self.XParIndex)
        self.setLabelY(self.YParName, self.YParIndex)
        plt.title(self.PlotTitle,color='r',fontsize=20) # pars like in class Text
        self.fig.canvas.set_window_title(self.Ydsname)        


    def mapCorrelatingArraysByTimeInit( self ) :

        #self.Xarr # is assumed to be available
        #self.Yarr # is assumed to be available
        XTimedsname  = printh5.get_item_path_to_last_name(self.Xdsname) + '/time'
        YTimedsname  = printh5.get_item_path_to_last_name(self.Ydsname) + '/time'

        print 'Xdsname =',self.Xdsname 
        print 'Ydsname =',self.Ydsname 

        try    : self.dsXT = self.h5file[XTimedsname]
        except :
            self.printWarningNonAvailable(XTimedsname)
            return

        try    : self.dsYT = self.h5file[YTimedsname]
        except :
            self.printWarningNonAvailable(YTimedsname)
            return

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
