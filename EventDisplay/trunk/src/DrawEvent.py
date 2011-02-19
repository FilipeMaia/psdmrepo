#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module DrawEvent...
#
#------------------------------------------------------------------------

"""Reads info from HDF5 file and rendering it depending on configuration parameters.

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
import h5py    # access to hdf5 file structure
from numpy import *  # for use like       array(...)
import time

import matplotlib
matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
import matplotlib.pyplot as plt

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp
import ConfigCSpad      as cs
import PlotsForCSpad    as cspad
import PlotsForImage    as image
import PlotsForWaveform as wavef
import PrintHDF5        as printh5

#---------------------
#  Class definition --
#---------------------
class DrawEvent ( object ) :
    """Reads info from HDF5 file and rendering it depending on configuration parameters.

    @see BaseClass
    @see OtherClass
    """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor"""
        print 'DrawEvent () Initialization'
        cp.confpars.h5_file_is_open = False
        self.plotsCSpad             = cspad.PlotsForCSpad()
        self.plotsImage             = image.PlotsForImage()
        self.plotsWaveform          = wavef.PlotsForWaveform()
        self.fig_window_is_open     = False 
        self.list_of_open_figs      = []

        # CSpad V1 for runs ~546,547...
        self.dsnameCSpadV1 = "/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV1/XppGon.0:Cspad.0/data"

        # CSpad V2 for runs ~900
        self.dsnameCSpadV2    = "/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/XppGon.0:Cspad.0/data"
        self.dsnameCSpadV2CXI = "/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDs1.0:Cspad.0/data"
        self.dsnameCSpadV2Conf= "/Configure:0000/CsPad::ConfigV2/"                        #CxiDs1.0:Cspad.0/config - is added auto

    #-------------------
    #  Public methods --
    #-------------------

    def drawEvent ( self, mode=1 ) :
        """Draws current event"""

        t_drawEvent = time.clock()
        
        #if not cp.confpars.h5_file_is_open :
        self.openHDF5File() # t=0us

        print 'Event %d' % ( cp.confpars.eventCurrent )

        runNumber = self.h5file.attrs['runNumber'] # t=0us 
        #print 'Run number = %d' % (runNumber) 


        # Loop over checked data sets
        self.figNum = 0
        #print 'Number of checked items:', len(cp.confpars.list_of_checked_item_names)
        for dsname in cp.confpars.list_of_checked_item_names :

            ds     = self.h5file[dsname]
            arr1ev = ds[cp.confpars.eventCurrent]

            item_last_name = printh5.get_item_last_name(dsname)
            cp.confpars.current_item_name_for_title = printh5.get_item_name_for_title(dsname)
            print 'Plot item:', dsname, ' item name:', item_last_name
            #print 'Name for plot title:', cp.confpars.current_item_name_for_title

            if dsname == self.dsnameCSpadV1 :
                print 'Draw plots for CSpad V1'

                #arr1ev # (4, 8, 185, 388) <- format of this record

                self.figNum += 1 
                if cp.confpars.cspadImageIsOn : 
                    self.plotsCSpad.plotCSpadV1Image(arr1ev,self.set_fig(4),plot=8)
                else : self.close_fig(self.figNum)

                self.figNum += 1 
                if cp.confpars.cspadImageOfPairIsOn : 
                    self.plotsCSpad.plotCSpadV1Image(arr1ev,self.set_fig(1),plot=1)
                else : self.close_fig(self.figNum)

                self.figNum += 1 
                if cp.confpars.cspadImageQuadIsOn : 
                    self.plotsCSpad.plotCSpadV1Image(arr1ev,self.set_fig(4),plot='Quad')
                else : self.close_fig(self.figNum)

                self.figNum += 1 
                if cp.confpars.cspadImageDetIsOn : 
                    self.plotsCSpad.plotCSpadV1Image(arr1ev,self.set_fig(4),plot='Det')
                else : self.close_fig(self.figNum)

                self.figNum += 1 
                if cp.confpars.cspadSpectrumIsOn : 
                    self.plotsCSpad.plotCSpadV1Spectrum(arr1ev,self.set_fig(4),plot=16)
                else : self.close_fig(self.figNum)

                self.figNum += 1 
                if cp.confpars.cspadSpectrum08IsOn : 
                    self.plotsCSpad.plotCSpadV1Spectrum(arr1ev,self.set_fig(4),plot=8)
                else : self.close_fig(self.figNum)
                
            if dsname == self.dsnameCSpadV2 or dsname == self.dsnameCSpadV2CXI :
                #print 'Draw plots for CSpad V2'

                #arr1ev # (32, 185, 388) <- format of this record

                self.getCSpadConfiguration(dsname)


                self.figNum += 1 
                if cp.confpars.cspadImageIsOn : 
                    self.plotsCSpad.plotCSpadV2Image(arr1ev,self.set_fig(4),plot=8)
                else : self.close_fig(self.figNum)

                self.figNum += 1 
                if cp.confpars.cspadImageOfPairIsOn : 
                    self.plotsCSpad.plotCSpadV2Image(arr1ev,self.set_fig(1),plot=1)
                else : self.close_fig(self.figNum)

                self.figNum += 1 
                if cp.confpars.cspadImageQuadIsOn : 
                    self.plotsCSpad.plotCSpadV2Image(arr1ev,self.set_fig(4),plot='Quad')
                else : self.close_fig(self.figNum)

                self.figNum += 1 
                if cp.confpars.cspadSpectrumIsOn : 
                    self.plotsCSpad.plotCSpadV2Spectrum(arr1ev,self.set_fig(4),plot=16)
                else : self.close_fig(self.figNum)

                self.figNum += 1 
                if cp.confpars.cspadSpectrum08IsOn : 
                    self.plotsCSpad.plotCSpadV2Spectrum(arr1ev,self.set_fig(4),plot=8)
                else : self.close_fig(self.figNum)
                
                for nwin in range(cp.confpars.cspadImageNWindowsMax) :
                    self.figNum += 1 
                    if cp.confpars.cspadImageDetIsOn and nwin < cp.confpars.cspadImageNWindows : 
                        self.plotsCSpad.plotCSpadV2Image(arr1ev,self.set_fig(4),plot='Det')
                    else : self.close_fig(self.figNum)

            if item_last_name == 'image' :
                self.figNum += 1 
                if cp.confpars.imageImageIsOn : 
                    self.plotsImage.plotImage(arr1ev,self.set_fig('1x1'))
                else : self.close_fig(self.figNum)

                self.figNum += 1 
                if cp.confpars.imageSpectrumIsOn : 
                    self.plotsImage.plotImageSpectrum(arr1ev,self.set_fig('1x1'))
                else : self.close_fig(self.figNum)

                self.figNum += 1 
                if cp.confpars.imageImageSpecIsOn : 
                    self.plotsImage.plotImageAndSpectrum(arr1ev,self.set_fig('1x2'))
                else : self.close_fig(self.figNum)

            if item_last_name == 'waveforms' :
                print 'Here should be an emplementation of stuff for waveform'

                self.figNum += 1 
                if cp.confpars.waveformWaveformIsOn : 
                    self.plotsWaveform.plotWFWaveform(arr1ev,self.set_fig('2x1'))
                else : self.close_fig(self.figNum)

        self.fig_window_is_open = True

        self.showEvent(mode)
        self.closeHDF5File()
        print 'Time to drawEvent() (sec) = %f' % (time.clock() - t_drawEvent)


    def getCSpadConfiguration( self, dsname ):

        if cp.confpars.cspadImageDetIsOn    \
        or cp.confpars.cspadImageQuadIsOn   \
        or cp.confpars.cspadImageOfPairIsOn \
        or cp.confpars.cspadImageIsOn       \
        or cp.confpars.cspadSpectrumIsOn    \
        or cp.confpars.cspadSpectrum08IsOn  : 

            item_second_to_last_name = printh5.get_item_second_to_last_name(dsname)
            cspad_config_ds_name = self.dsnameCSpadV2Conf + item_second_to_last_name + '/config' 
            
            print '   CSpad configuration dataset name:', cspad_config_ds_name

            dsConf = self.h5file[cspad_config_ds_name]      # t=0.01us
            cs.confcspad.indPairsInQuads = dsConf.value[13] #
            #print "Indexes of pairs in quads =\n",cs.confcspad.indPairsInQuads 


    def showEvent ( self, mode=1 ) :
        """showEvent: plt.show() or draw() depending on mode"""

        t_start = time.clock()
        if mode == 1 :   # Single event mode
            plt.show()  
        else :           # Slide show 
            plt.draw()   # Draws, but does not block
        print 'Time to show or draw (sec) = %f' % (time.clock() - t_start)


    def stopDrawEvent ( self ) :
        """Operations in case of stop drawing event(s)"""
        print 'stopDrawEvent()'
        self.drawEvent() # mode=1 (by default) for the last plot


    def quitDrawEvent ( self ) :
        """Operations in case of quit drawing event(s)"""
        #self.plotsCSpad.close_fig1()
        #self.plotsImage.close_fig1()
        self.close_fig()
        plt.ioff()
        plt.close()
        self.closeHDF5File()
        print 'quitDrawEvent()'


    def openHDF5File( self ) :     
        fname = cp.confpars.dirName+'/'+cp.confpars.fileName
        #print 'openHDF5File() : %s' % (fname)
        self.h5file=  h5py.File(fname, 'r') # open read-only       
        cp.confpars.h5_file_is_open = True
        #printh5.print_file_info(self.h5file)


    def closeHDF5File( self ) :       
        if cp.confpars.h5_file_is_open :
            self.h5file.close()
            cp.confpars.h5_file_is_open = False
            #print 'closeHDF5File()'


    def set_fig( self, type=None ):
        """Set current fig."""

        ##self.figNum += 1 
        #if self.fig_window_is_open :
        if self.figNum in self.list_of_open_figs :
            self.fig = plt.figure(num=self.figNum)        
        else :
            self.fig = self.open_fig(type)
            self.set_window_position()
            self.list_of_open_figs.append(self.figNum)
        return self.fig

               
    def open_fig( self, type=None ):
        """Open window for figure."""
        #print 'open_fig()'

        plt.ion() # enables interactive mode
        if type == 1 :
            self.fig = plt.figure(num=self.figNum, figsize=(6,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        if type == '1x1' :
            self.fig = plt.figure(num=self.figNum, figsize=(6,6), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        if type == '1x2' :
            self.fig = plt.figure(num=self.figNum, figsize=(5,10), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        if type == '2x1' :
            self.fig = plt.figure(num=self.figNum, figsize=(10,5), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        if type == '3x4' :
            self.fig = plt.figure(num=self.figNum, figsize=(6,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        if type == 4 :
            self.fig = plt.figure(num=self.figNum, figsize=(10,10), dpi=80, facecolor='w',edgecolor='w',frameon=True)
            self.fig.subplots_adjust(left=0.08, bottom=0.02, right=0.98, top=0.98, wspace=0.2, hspace=0.1)
        else :
            self.fig = plt.figure(num=self.figNum)

        self.fig.myXmin = None
        self.fig.myXmax = None
        self.fig.myYmin = None
        self.fig.myYmax = None
        self.fig.myZoomIsOn = False

        return self.fig

    def set_window_position( self ):
        """Move window in desired position."""
        #print 'set_window_position()'
        #plt.get_current_fig_manager().window.move(890, 100) #### This works!
        fig_QMainWindow = plt.get_current_fig_manager().window
        posx = cp.confpars.posGUIMain[0] + 460 + 50*self.figNum
        posy = cp.confpars.posGUIMain[1]       + 20*(self.figNum-1)
        fig_QMainWindow.move(posx,posy)
        #fig_QMainWindow.move(820+50*self.figNum, 20*(self.figNum-1)) #### This works!


    def close_fig( self, figNum=None ):
        """Close fig and its window."""
        if figNum==None :
            plt.close('all') # closes all the figure windows
        else :
            plt.close(figNum)
            if figNum in self.list_of_open_figs : self.list_of_open_figs.remove(figNum)

        if  self.fig_window_is_open :
            self.fig_window_is_open = False 
            #print 'close_fig()'



#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
