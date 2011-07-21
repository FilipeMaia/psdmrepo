#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotsForCalibCycles...
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
import GlobalMethods    as gm

#---------------------
#  Class definition --
#---------------------
class PlotsForCalibCycles ( object ) :
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
  

    def plotCalibCycles0 ( self, fig, h5file ) :

        print 'plotCalibCycles(..) IS UNDER DEVELOPEMENT YET...' 


    def plotCalibCycles ( self, fig, h5file ) :

        self.h5file = h5file
        self.fig    = fig
        
        win = fig.nwin
        self.Ydsname   = cp.confpars.calibcycleWindowParameters[win][0] 
        self.Xdsname   = cp.confpars.calibcycleWindowParameters[win][1] 
        self.radioXPar = cp.confpars.calibcycleWindowParameters[win][2] 
        self.Ymin      = cp.confpars.calibcycleWindowParameters[win][3]
        self.Ymax      = cp.confpars.calibcycleWindowParameters[win][4]
        self.Xmin      = cp.confpars.calibcycleWindowParameters[win][5]
        self.Xmax      = cp.confpars.calibcycleWindowParameters[win][6]
        self.YParName  = cp.confpars.calibcycleWindowParameters[win][7] 
        self.XParName  = cp.confpars.calibcycleWindowParameters[win][8] 
        self.YLimsIsOn = cp.confpars.calibcycleWindowParameters[win][9] 
        self.XLimsIsOn = cp.confpars.calibcycleWindowParameters[win][10] 
        self.LogIsOn   = cp.confpars.calibcycleWindowParameters[win][11]
        self.YNBins    = cp.confpars.calibcycleWindowParameters[win][12]
        self.XNBins    = cp.confpars.calibcycleWindowParameters[win][13]
        self.YParIndex = cp.confpars.calibcycleWindowParameters[win][14] # Event/record index
        self.XParIndex = cp.confpars.calibcycleWindowParameters[win][15] # Event/record index 
        
        if self.Ydsname == 'None' :
            print 'THE Ydsname=', self.Ydsname, ' IS SET INCORRECTLY. THE CALIBCYCLES PLOT', win,' IS IGNORED'
            return

        if self.YParName == 'None' : 
            print 'THE YParName=', self.YParName, ' IS NOT SET. THE CALIBCYCLES PLOT', win,' IS IGNORED'
            return

        print 'self.Ydsname   = ', self.Ydsname 
        print 'self.YParName  = ', self.YParName

        s0, sN, isFoundInString = gm.getPatternEndsInTheString(self.Ydsname, pattern='Run:')
        print 'self.Ydsname[0:sN+4]  = ', self.Ydsname[0:sN+4]
        runGroupName = self.Ydsname[0:sN+4]
        print 'Run group name = ', runGroupName 


        g = h5file[runGroupName]
        number_of_calibcycles = len(g.items())
        print 'Number of calibcycles in the run group:', number_of_calibcycles


        s0, sN, isFoundInString = gm.getPatternEndsInTheString(self.Ydsname, pattern='CalibCycle:')
        print 's0, sN, isFoundInString =', s0, sN, isFoundInString
        print 'self.Ydsname[s0:sN]  = ', self.Ydsname[s0:sN] 
        print 'self.Ydsname[sN:sN+4]= ', self.Ydsname[sN:sN+4] 

        self.Ydsname_for_calibcycle = self.Ydsname

        if self.radioXPar == 1 : # for Time
            self.Xdsname = gm.get_item_path_to_last_name(self.Ydsname) + '/time'
            print 'Xdsname =',self.Xdsname 

        elif self.radioXPar == 2 : # for X-Parameter
            if self.Xdsname == 'None' :
                print 'THE Xdsname=', self.Xdsname, ' IS SET INCORRECTLY. THE CALIBCYCLE PLOT', win,' IS IGNORED' 
                return

            if self.XParName == 'None' : 
                print 'THE XParName=', self.XParName, ' IS NOT SET. THE CALIBCYCLE PLOT', win,' IS IGNORED'
                return

        self.Yarr = []
        self.Xarr = []

        # Loop over calibcycles and fill Y and X (if necessary) arrays
        for i in range(number_of_calibcycles) :

            str_calibc_number = '%04d' % i
            self.Ydsname_for_calibcycle = self.Ydsname[0:sN] + str_calibc_number + self.Ydsname[sN+4:]
            print self.Ydsname_for_calibcycle,

            ds  = h5file[self.Ydsname_for_calibcycle]

            if self.YParIndex == 'None' :
                self.arry = ds[self.YParName] 
            else :
                self.arry = ds[self.YParName][self.YParIndex]  # Next level index for ipimb...  

            val = np.mean(self.arry)   # We use the averaged over events value of the parameter 
            print '  <Ypar>=',val,
            self.Yarr.append(val)

            self.Xdsname_for_calibcycle = self.Xdsname[0:sN] + str_calibc_number + self.Xdsname[sN+4:]
            #print self.Xdsname_for_calibcycle,

            if self.radioXPar == 0 :   # for index
                print ' ' 

            elif self.radioXPar == 1 : # for Time

                self.dsX = h5file[self.Xdsname_for_calibcycle]
                timearr = 0.000000001 * self.dsX['nanoseconds'] + self.dsX['seconds']
                timemean=np.mean(timearr)
                self.Xarr.append( timemean )
                print '  <time>=', timemean

            elif self.radioXPar == 2 : # for X-Parameter

                dsx  = h5file[self.Xdsname_for_calibcycle]
                self.arrx = ds[self.XParName] 

                if self.XParIndex == 'None' :
                    self.arrx = ds[self.XParName] 
                else :
                    self.arrx = ds[self.XParName][self.XParIndex]  # Next level index for ipimb...  

                meanx = np.mean(self.arrx) # We use the averaged over events value of the parameter 
                print '  <Xpar>=', meanx
                self.Xarr.append(meanx)

            elif self.radioXPar == 3 : # for Y-Parameter histogram 
                print ' ' 

        self.Xarr = np.array(self.Xarr)
        self.Yarr = np.array(self.Yarr)

        nYpoints = self.Yarr.shape[0]
        print '\nNew Win:', win, ' CalibCycle plot for nYpoints =', nYpoints 

        self.markerStyle = 'bs-'

        if self.radioXPar == 0 : # for CalibCycle Index
            self.Xarr = range(nYpoints)
            print 'Index array from 0 to', nYpoints
            self.XTitle = 'Calibcycle index'
            self.PlotTitle = 'Plot ' + str(win+1) + ': <Parameter> vs calibcycle index'
            self.plotWaveform()

        elif self.radioXPar == 1 : # for Time
            self.Xarr -= self.Xarr[0]
            print 'Time array :\n', self.Xarr 
            self.XTitle = '<Time> (sec)'
            self.PlotTitle = 'Plot ' + str(win+1) + ': <Parameter> vs calibcycle <Time>'
            self.plotWaveform()

        elif self.radioXPar == 2 : # for X-Parameter
            print 'X-Parameter array :\n', self.Xarr
            self.XTitle = self.XParName
            self.PlotTitle = 'Plot ' + str(win+1) + ': Correlations of two <parameters> over calibcycles'
            self.markerStyle = 'bo-'
            #self.plotWaveform()
            self.plot2DHistogram()

        elif self.radioXPar == 3 : # for Y-Parameter histogram 
            self.PlotTitle = 'Plot ' + str(win+1) + ': <Y-parameter> histogram over calibcycles'
            self.plot1DHistogram() 


        plt.show()            

        ##-------------
        return
        ##-------------


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

        self.setLabelY(self.YParName,self.YParIndex)
        plt.xlabel(self.XTitle)
        plt.title(self.PlotTitle,color='r',fontsize=20) # pars like in class Text
        self.fig.canvas.set_window_title(self.Ydsname)


    def plot1DHistogram( self ) :
        plt.clf()
        plt.xlim(auto=True)
        arrh = self.excludeNanFromArray(self.Yarr)       
        axes = plt.hist(arrh, bins=self.YNBins, color='b')
        if self.YLimsIsOn : plt.xlim(self.Ymin,self.Ymax)
        self.setLabelX(self.YParName, self.YParIndex)
        plt.title(self.PlotTitle,color='r',fontsize=20) # pars like in class Text
        self.fig.canvas.set_window_title(self.Ydsname)        


    def excludeNanFromArray(self, arr) :
        print arr
        print arr.shape[0]
        newarr = []
        for val in arr :
            if np.isnan(val) : continue
            newarr.append(val)
        print 'WARNING: All NaN parameters are excluded...'
        return np.array(newarr)



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
