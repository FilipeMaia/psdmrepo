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
#import h5py

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
  
    def plotWFWaveform( self, ds1ev, fig, h5file=None ):
        """Plot waveform from input array."""

        print 'plotWFWaveform'
        print 'ds1ev.shape', ds1ev.shape
        numberOfWF, par2, self.dimX = ds1ev.shape
        #print 'numberOfWF, par2, dimX = ', numberOfWF, par2, dimX
        #print 'arrwf.shape', arrwf.shape
        self.nwin   = nwin = fig.nwin
        self.ds1ev  = ds1ev
        self.h5file = h5file

        if cp.confpars.waveformWindowParameters[self.nwin][1] > 3: # if vertical or horizontal units are requested
            self.getScaleConfigParameters()

        self.arrT = self.getTimeArray()
            
        par = []

        indwf1 = cp.confpars.waveformWindowParameters[nwin][7]
        indwf2 = cp.confpars.waveformWindowParameters[nwin][8]
        indwf3 = cp.confpars.waveformWindowParameters[nwin][9]
        indwf4 = cp.confpars.waveformWindowParameters[nwin][10]


        if indwf1 != None : par.append( (self.arrT, self.getAmplitudeArray(indwf1), 'k-') )
        if indwf2 != None : par.append( (self.arrT, self.getAmplitudeArray(indwf2), 'r-') )
        if indwf3 != None : par.append( (self.arrT, self.getAmplitudeArray(indwf3), 'g-') )
        if indwf4 != None : par.append( (self.arrT, self.getAmplitudeArray(indwf4), 'b-') )

        fig.canvas.set_window_title(cp.confpars.current_item_name_for_title) 
        plt.clf() # clear plot
        fig.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.94, wspace=0.1, hspace=0.1)        

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


        if self.getBitStatus( 1 ):
            plt.ylim(cp.confpars.waveformWindowParameters[nwin][2],\
                     cp.confpars.waveformWindowParameters[nwin][3]  )
            
        if self.getBitStatus( 2 ):
            plt.xlim(cp.confpars.waveformWindowParameters[nwin][4],\
                     cp.confpars.waveformWindowParameters[nwin][5]  )
        else :
            plt.xlim(self.arrT.min(), self.arrT.max())
        
        str_title='Waveform, event ' + str(cp.confpars.eventCurrent)
        
        plt.title(str_title,color='r',fontsize=20) # pars like in class Text

        XTitle = 'Index'
        YTitle = 'ADC units'
        if self.getBitStatus( 4 ) : YTitle = 'Amplitude (V)'     
        if self.getBitStatus( 8 ) : XTitle = 'Time (ns)'    
        plt.xlabel(XTitle)
        plt.ylabel(YTitle)



    def getAmplitudeArray(self,indwf) :
        arrA0 = self.ds1ev[indwf,0,...]
        if self.getBitStatus( 4 ): # Apply units
            arrA = array(arrA0,dtype=float)
            scale = self.vertFullScale / 2**16
            return arrA0 * scale + self.vertOffset
        else :                     # Do not apply units
            return arrA0
        

    def getTimeArray(self) :
        arrT0 = arange(self.dimX,dtype=float)
        #print 'arrT = ',arrT0

        if self.getBitStatus( 8 ): # Apply units
            return arrT0 * self.horizSampleInterval + self.horizDelayTime
        else :                     # Do not apply units
            return arrT0


    def getBitStatus(self,bit):
        return cp.confpars.waveformWindowParameters[self.nwin][1] & bit
        #rangeUnitsBits : 1-ALimits, 2-TLimits, 4-AUnits, 8-TUnits



    def getScaleConfigParameters(self):
        dsname = cp.confpars.waveformWindowParameters[self.nwin][0]
        #print 'Waveforms for dataset:', dsname
        dsgroup = '/Configure:0000/Acqiris::ConfigV1/' + printh5.get_item_second_to_last_name(dsname)
        dsname_config_vert  = dsgroup + '/vert'
        dsname_config_horiz = dsgroup + '/horiz'
        #print 'Configuration dataset vert : ', dsname_config_vert
        #print 'Configuration dataset horiz: ', dsname_config_horiz

        self.vertFullScale = 1 
        self.vertOffset    = 0
        self.horizSampleInterval = 1
        self.horizDelayTime      = 0

        try: 
            dsvert  = self.h5file[dsname_config_vert]
        except KeyError:
            #if not printh5.isDataset(dsvert) :
            print 80*'!'
            print 'WARNING:', dsname_config_vert, ' DATASET DOES NOT EXIST IN HDF5\n',\
                  'PROGRAM CAN NOT USE VERTICAL UNITS FROM CONFIGURATION; USE DEFAULT, SCALE=1, OFFSET=0'
            print 80*'!'
            return

        try: 
            dshoriz = self.h5file[dsname_config_horiz]
        except KeyError:
            #if not printh5.isDataset(dshoriz) :
            print 80*'!'
            print 'WARNING:', dsname_config_horiz, ' DATASET DOES NOT EXIST IN HDF5\n',\
                  'PROGRAM CAN NOT USE HORIZONTAL UNITS FROM CONFIGURATION; USE DEFAULT, SCALE=1, OFFSET=0'
            print 80*'!'
            return

        #print 'dsvert.dtype =', dsvert.dtype
        #print 'dsvert[0] =', dsvert[0]
        self.vertFullScale = dsvert[0][0] 
        self.vertOffset    = dsvert[0][1]
        self.vertCoupling  = dsvert[0][2]
        self.vertBandwidth = dsvert[0][3]

        print 'self.vertFullScale      =',self.vertFullScale         
        print 'self.vertOffset         =',self.vertOffset         
        print 'self.vertCoupling       =',self.vertCoupling         
        print 'self.vertBandwidth      =',self.vertBandwidth         

        #print 'dshoriz.dtype =',   dshoriz.dtype
        #print 'dshoriz.value =',   dshoriz.value
        self.horizSampleInterval = dshoriz.value[0] * 1e9 # time in ns 
        self.horizDelayTime      = dshoriz.value[1] * 1e9 # time in ns
        self.horizNSamples       = dshoriz.value[2]
        self.horizNSegments      = dshoriz.value[3]
        print 'self.horizSampleInterval=',self.horizSampleInterval
        print 'self.horizDelayTime     =',self.horizDelayTime     
        print 'self.horizNSamples      =',self.horizNSamples
        print 'self.horizNSegments     =',self.horizNSegments

#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------


    def plotWaveVsEvent(self, fig, h5file):
        #print "plotWaveVsEvent(...)"

        self.nwin   = nwin = fig.nwin
        self.h5file = h5file
        dsname = cp.confpars.waveformWindowParameters[self.nwin][0]
        #cp.confpars.current_item_name_for_title = printh5.get_item_name_for_title(dsname)
        cp.confpars.current_item_name_for_title = str(dsname)

        ds     = self.h5file[dsname]
        Nevents = ds.shape[0]     
        event = 0
        self.ds1ev = ds[event]
        numberOfWF, par2, self.dimX = self.ds1ev.shape
        #print 'Nevents, dimX = ', Nevents, self.dimX      

        # Use the 1st defined index for this wave.
        for par in range(10,6,-1):
            if               cp.confpars.waveformWindowParameters[nwin][par] != None :
                self.indwf = cp.confpars.waveformWindowParameters[nwin][par]

        if cp.confpars.waveformWindowParameters[self.nwin][1] > 3: # if vertical or horizontal units are requested
            self.getScaleConfigParameters()

        self.arrA = self.getAmplitudeArray(self.indwf)
        self.arrT = self.getTimeArray()

        # Make 2D float array of the wave vs event
        self.arrWaveVsEv = empty([Nevents, self.dimX], dtype=float)

        for event in range(Nevents) :
            #print 'event =', event
            self.ds1ev = ds[event]            
            self.arrWaveVsEv[event,...] = self.getAmplitudeArray(self.indwf)

        # Define A range
        ARange = None
        if self.getBitStatus( 1 ): # A limits is on
            ARange = (cp.confpars.waveformWindowParameters[nwin][2],\
                      cp.confpars.waveformWindowParameters[nwin][3] )


        #arr2d = self.arrWaveVsEv
        # Define T INDEX range
        self.TIndMin = 0
        self.TIndMax = self.arrT.shape[0] - 1 
        self.TMin = self.TIndMin
        self.TMax = self.TIndMax

        if self.getBitStatus( 2 ): # T limits is on            
            self.TMin = cp.confpars.waveformWindowParameters[nwin][4]
            self.TMax = cp.confpars.waveformWindowParameters[nwin][5]
        else:
            self.TMin = self.arrT.min()
            self.TMax = self.arrT.max()


        if self.getBitStatus( 8 ): # T Units
            self.TIndMin = (self.TMin - self.horizDelayTime) / self.horizSampleInterval
            self.TIndMax = (self.TMax - self.horizDelayTime) / self.horizSampleInterval
        else:
            self.TIndMin = self.TMin
            self.TIndMax = self.TMax

        Range  = (self.TMin, self.TMax, 0, Nevents)     
        arr2d   = self.arrWaveVsEv[...,self.TIndMin:self.TIndMax]

        # Begin plot
        fig.canvas.set_window_title(cp.confpars.current_item_name_for_title) 
        plt.clf() # clear plot
        fig.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.94, wspace=0.1, hspace=0.1)        
        axsp = fig.add_subplot(111)

        #plt.plot( self.arrT, self.arrA, 'k-' )

        axim = axsp.imshow(arr2d, origin='lower',extent=Range, interpolation='nearest', aspect='auto')
        mycb = fig.colorbar(axim, fraction=0.1, pad=0.1, shrink=1.0, aspect=40, orientation=1) # ticks=coltickslocs

        axim.set_clim(ARange) # (0,15000)

        XTitle = 'Index'
        YTitle = 'Event'
        ZTitle = 'ADC units'
        if self.getBitStatus( 4 ) : ZTitle = 'Amplitude (V)'     
        if self.getBitStatus( 8 ) : XTitle = 'Time (ns)'    
        plt.xlabel(XTitle)
        plt.ylabel(YTitle)
        #plt.zlabel(ZTitle)


#--------------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    sys.exit ( "Module is not supposed to be run as main module" )

#--------------------------------
