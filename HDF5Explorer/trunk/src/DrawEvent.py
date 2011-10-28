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
import numpy as np
import scipy.misc as scimisc
import time

import matplotlib
matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
import matplotlib.pyplot as plt

from PyQt4 import QtGui

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters          as cp
import ConfigCSpad               as cs
import PlotsForCSpad             as cspad
import PlotsForImage             as image
import PlotsForWaveform          as wavef
import PlotsForCorrelations      as corrs
import PlotsForCalibCycles       as calib
import PlotsForCSpadProjections  as cspadproj
import PlotsForImageProjections  as imageproj
import PrintHDF5                 as printh5
import GlobalMethods             as gm
import CalibCycles               as cc

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
    def __init__ ( self, parent=None ) :
        """Constructor"""
        #print 'DrawEvent () Initialization'
        cp.confpars.h5_file_is_open = False
        self.plotsCSpad             = cspad.PlotsForCSpad()
        self.plotsImage             = image.PlotsForImage()
        self.plotsWaveform          = wavef.PlotsForWaveform()
        self.plotsCorrelations      = corrs.PlotsForCorrelations()
        self.plotsCalibCycles       = calib.PlotsForCalibCycles()
        self.plotsCSpadProj         = cspadproj.PlotsForCSpadProjections(self.plotsCSpad)
        self.plotsImageProj         = imageproj.PlotsForImageProjections()
        
        self.list_of_open_figs      = []
        self.parent                 = parent
        #self.fileNameWithAlreadySetCSpadConfiguration = None

        #self.arrInWindowMax         = 0 
        #self.arrInWindowSum         = 0

        # CSpad V1 for runs ~546,547...
        self.dsnameCSpadV1 = "/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV1/XppGon.0:Cspad.0/data"

       # CSpad V2 for runs ~900
       #self.dsnameCSpadV2    = "/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/XppGon.0:Cspad.0/data"
       #self.dsnameCSpadV2CXI = "/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDs1.0:Cspad.0/data"
       #self.dsnameCSpadV2Conf= "/Configure:0000/CsPad::ConfigV2/"                        #CxiDs1.0:Cspad.0/config - is added auto
       #self.dsnameCSpadV3Conf= "/Configure:0000/CsPad::ConfigV3/"                        #CxiDs1.0:Cspad.0/config - is added auto
        self.dsnameCSpadVXConf= "/Configure:0000/CsPad::ConfigV"

    #-------------------
    #  Public methods --
    #-------------------

    def dimIsFixed(self, dsname) :
        #print 'Check if the item "_dim_fix_flag_201103" is in the group', dsname
        path  = gm.get_item_path_to_last_name(dsname)
        group = self.h5file[path]
        if '_dim_fix_flag_201103' in group.keys() : return True
        else :                                      return False 

    def selectionIsPassed(self) :

        if not cp.confpars.selectionIsOn : return True           # Is passed if selection is OFF

        self.nwin = None

        for win in range(cp.confpars.selectionNWindows) :        # Loop over all windows

            dsname = cp.confpars.selectionWindowParameters[win][6]
            if dsname == 'None' :
                print '\n',70*'!','\nWARNING: SELECTION IS REQUESTED, BUT ITS DATASET IS NOT SET.', \
                      '\nFIX THIS PROBLEM IN SELECTION GUI\n',70*'!','\n'
                return True

            ds     = self.h5file[dsname]
            self.arr1ev = ds[cp.confpars.eventCurrent]

            if gm.ImageIsInTheName(dsname) :      # Check for IMAGE

                if not self.dimIsFixed(dsname) :
                    self.arr1ev.shape = (self.arr1ev.shape[1],self.arr1ev.shape[0])
                #self.arr2d = self.arr1ev
                self.arr2d = self.applyImageCorrections( self.arr1ev, dsname )

            elif gm.CSpadIsInTheName(dsname) :    # Check for CSpad 

                self.getCSpadConfiguration(dsname)
                self.arr2d = self.plotsCSpad.getImageArrayForDet( self.arr1ev )

            if self.selectionInWindowIsPassed(win,self.arr2d) : return True # All selections are included as OR
                            
        return False


    def selectionInWindowIsPassed(self, win, arr2d) :

        Thr   = cp.confpars.selectionWindowParameters[win][0]
        inBin = cp.confpars.selectionWindowParameters[win][1]
        iXmin = cp.confpars.selectionWindowParameters[win][2]
        iXmax = cp.confpars.selectionWindowParameters[win][3]
        iYmin = cp.confpars.selectionWindowParameters[win][4]
        iYmax = cp.confpars.selectionWindowParameters[win][5]

        #print 'win, Thr, inBin, iXmin, iXmax, iYmin, iYmax, dsname =',\
        #      win, Thr, inBin, iXmin, iXmax, iYmin, iYmax,\
        #      cp.confpars.selectionWindowParameters[win][6]

        if iYmin > arr2d.shape[0] : iYmin = arr2d.shape[0]
        if iYmax > arr2d.shape[0] : iYmax = arr2d.shape[0]
        if iXmin > arr2d.shape[1] : iXmin = arr2d.shape[1]
        if iXmax > arr2d.shape[1] : iXmax = arr2d.shape[1]
        
        arrInWindow = arr2d[iYmin:iYmax,iXmin:iXmax]


        self.arrInWindowMax = arrInWindow.max()
        self.arrInWindowSum = arrInWindow.sum()

        #print ' arrInWindow =',arrInWindow
        #print ' arrInWindowMax =',self.arrInWindowMax
        #print ' arrInWindowSum =',self.arrInWindowSum

        if inBin :
            if self.arrInWindowMax > Thr : return True
        else :
            if self.arrInWindowSum > Thr : return True

        return False


    def averageOverEvents(self, mode=1) :
        print 'averageOverEvents'

        t_start = time.clock()
        self.openHDF5File() # t=0us
        if not cp.confpars.h5_file_is_open : return
        
        self.eventStart = cp.confpars.eventCurrent
        self.eventEnd   = cp.confpars.eventCurrent + 1000 # This is have changed at 1st call loopOverDataSets

        print 'selectionIsOn =', cp.confpars.selectionIsOn 

        self.numEventsSelected = 0
        self.loopIsContinued   = True

        # Loop over events
        while self.loopIsContinued :

            #print 'TEST:self.loopIsContinued     = ', self.loopIsContinued

            selectionIsPassed = self.selectionIsPassed() 
            self.printEventSelectionStatistics()

            self.loopOverDataSets() # Initialization & Accumulation

            cp.confpars.eventCurrent += 1
            if selectionIsPassed : self.numEventsSelected   += 1
            self.loopIsContinued = self.numEventsSelected < cp.confpars.numEventsAverage and cp.confpars.eventCurrent < self.eventEnd 

            #cp.confpars.eventCurrent -= 1

        self.loopOverDataSets(option=1) # Normalization per 1 event

        self.drawAveragedArrays(mode)

        print 'Time to averageOverEvents() (sec) = %f' % (time.clock() - t_start)
        self.closeHDF5File()


    def printEventSelectionStatistics(self) :
        if cp.confpars.eventCurrent % 10 == 0 :
            if cp.confpars.selectionIsOn :
                print ' Current event =', cp.confpars.eventCurrent,\
                      ' Selected =',      self.numEventsSelected,  \
                      ' arrInWindow.max() =', self.arrInWindowMax, \
                      ' arrInWindow.sum() =', self.arrInWindowSum
            else :
                print ' Current event =', cp.confpars.eventCurrent


    def drawAveragedArrays(self, mode=1) :
        """This method makes initialization and draws averaged arrays"""

        self.nwin   = None
        self.figNum = 0
        self.loopOverDataSets(option=5) # Draw normalized datasets
        self.showEvent(mode)


    def loopOverDataSets ( self, option=None ) :
        """Loop over datasets and accumulating arrays"""

        if option == None :                                   # Initialization of lists
            if cp.confpars.eventCurrent == self.eventStart :  # for the start event
                self.ave1ev    = []                         
                self.avedsname = []

        indds=-1
        for dsname in cp.confpars.list_of_checked_item_names :

            item_last_name   = gm.get_item_last_name(dsname)
            itemIsForAverage = gm.ImageIsInTheName(dsname) or \
                               item_last_name=='waveforms' or \
                               gm.CSpadIsInTheName(dsname)

            if not itemIsForAverage : continue

            ds = self.h5file[dsname]

            if cp.confpars.eventCurrent >= ds.shape[0] :
                print 80*'=', \
                      '\nWARNING! CURRENT EVENT NUMBER', cp.confpars.eventCurrent, \
                      ' EXCEEDS THE ARRAY SHAPE INDEX', ds.shape[0], \
                      '\nfor dataset:', dsname, \
                      '\nTHIS EVENT IS NOT INCLUDED IN AVERAGE!'
                continue

            indds += 1 
            if  option == None :

                self.arr1ev = ds[cp.confpars.eventCurrent]

                if item_last_name == 'image' : 
                    if not self.dimIsFixed(dsname) :
                        self.arr1ev.shape = (self.arr1ev.shape[1],self.arr1ev.shape[0])

                if cp.confpars.eventCurrent != self.eventStart :

                    self.ave1ev[indds] += self.arr1ev              # Accumulation
                    #print 'self.ave1ev[indds]\n', self.ave1ev[indds]

                else :                                           # Initialization
                    self.ave           = np.zeros(self.arr1ev.shape, dtype=np.float32)
                    self.ave1ev        .append(self.ave)
                    self.ave1ev[indds] += self.arr1ev   
                    self.avedsname     .append(dsname)
                    self.eventEnd      = ds.shape[0]
                    print 'Total numbr of events in averaged sample [', indds, '] =', self.eventEnd

            elif option == 1 :                                   # Normalization per 1 event
                if self.numEventsSelected > 0 :
                    self.ave1ev[indds] /= self.numEventsSelected
                    #print 50*'=' + '\nself.ave1ev[indds]\n', self.ave1ev[indds]

            elif option == 5 :                                   # Draw averaged dataset

                #print 'Plot the dataset', dsname, '\naveraged over .numEventsSelected=', self.numEventsSelected

                self.plotsCSpad.resetEventWithAlreadyGeneratedCSpadDetImage()
                self.drawArrayForDSName(self.avedsname[indds], self.ave1ev[indds])
                self.saveArrayForDSNameInFile(self.avedsname[indds], self.ave1ev[indds]) 


    def drawNextEvent ( self, mode=1 ) :
        """Draws the next (selected) event"""

        self.openHDF5File() # t=0us
        if not cp.confpars.h5_file_is_open : return

        self.numEventsSelected = 0
        while True :
            cp.confpars.eventCurrent += cp.confpars.span
            if self.selectionIsPassed() : 
                self.drawEventFromOpenFile(mode) # Draw everything for current event
                break
            self.printEventSelectionStatistics()

        self.closeHDF5File()


    def drawPreviousEvent ( self, mode=1 ) :
        """Draws the previous (selected) event"""

        self.openHDF5File() # t=0us
        if not cp.confpars.h5_file_is_open : return

        self.numEventsSelected = 0
        while True :
            cp.confpars.eventCurrent -= cp.confpars.span
            if cp.confpars.eventCurrent<0 :
                cp.confpars.eventCurrent=0
                print 'Event = 0, there is no more previous event in this file.'
                break
            if self.selectionIsPassed() : 
                self.drawEventFromOpenFile(mode) # Draw everything for current event
                break
            self.printEventSelectionStatistics()

        self.closeHDF5File()



    def startSlideShow ( self ) :
        """Start show (selected) events in permanent mode"""

        self.openHDF5File() # t=0us
        if not cp.confpars.h5_file_is_open : return

        eventStart = cp.confpars.eventCurrent
        eventEnd   = cp.confpars.eventCurrent + 1000*cp.confpars.span
        self.numEventsSelected = 0

        while (self.parent.SHowIsOn) :
            if cp.confpars.eventCurrent>eventEnd : break
            QtGui.QApplication.processEvents()
            if not self.parent.SHowIsOn : break
            if self.selectionIsPassed() :
                self.drawEventFromOpenFile(mode=0) # mode for slide show (draw())
                self.parent.numbEdit.setText( str(cp.confpars.eventCurrent) )

            self.printEventSelectionStatistics()
            cp.confpars.eventCurrent+=cp.confpars.span

        self.closeHDF5File()


    def stopSlideShow ( self ) :
        """Operations in case of stop drawing event(s)"""
        print 'stopSlideShow()'
        #self.drawEventFromOpenFile() # mode=1 (by default) for the last plot
        self.showEvent (mode=1) # apply show() mode for the last event
        self.closeHDF5File()


    def drawEvent ( self, mode=1 ) :
        """Draws current event"""

        self.openHDF5File() # t=0us
        if not cp.confpars.h5_file_is_open : return
        self.drawEventFromOpenFile (mode)
        self.closeHDF5File()


    def drawEventFromOpenFile ( self, mode=1 ) :
        """Draws current event when the file is already open"""

        t_drawEvent = time.clock()
        print 'Event %d' % ( cp.confpars.eventCurrent )

        print 'selectionIsPassed', self.selectionIsPassed() 

        # Loop over checked data sets
        self.nwin   = None
        self.figNum = 0

        for dsname in cp.confpars.list_of_checked_item_names :

            ds     = self.h5file[dsname]
            if cp.confpars.eventCurrent >= ds.shape[0] :
                print 'WARNING! CURRENT EVENT NUMBER ', cp.confpars.eventCurrent, \
                      ' EXCEEDS THE ARRAY SHAPE INDEX ', ds.shape[0], \
                      '\nfor dataset: ', dsname, \
                      '\nPLOTS ARE IGNORED!'
                return

            self.arr1ev = ds[cp.confpars.eventCurrent]

            if gm.get_item_last_name(dsname) == 'image' : 
                if not self.dimIsFixed(dsname) :
                    self.arr1ev.shape = (self.arr1ev.shape[1],self.arr1ev.shape[0])

            self.drawArrayForDSName(dsname,self.arr1ev)

        self.showEvent(mode)
        print 'Time to drawEvent() (sec) = %f' % (time.clock() - t_drawEvent)


    def saveArrayForDSNameInFile(self, dsname, arr1ev) :

        cspadIsInTheName = gm.CSpadIsInTheName(dsname)
        item_last_name   = gm.get_item_last_name(dsname)
        #cp.confpars.current_item_name_for_title = printh5.get_item_name_for_title(dsname)

        if cspadIsInTheName :

            print 'saveArrayForDSNameInFile'
            #arr1ev # (32, 185, 388) <- format of this record
            #print 'arr1ev.shape=', arr1ev.shape
            self.getCSpadConfiguration(dsname)
            arr2d = self.plotsCSpad.getImageArrayForDet( arr1ev )
            #print 'arr2d.shape=', arr2d.shape
            #print 'arr1ev=/n', arr1ev
            #print 'arr2d[100:200,100:200]=\n', arr2d[100:200,100:200]
            gm.saveNumpyArrayInFile(arr2d,  fname='cspad-ave.txt' , format='%f') # , format='%i')
           #gm.saveNumpyArrayInFile(arr1ev, fname='cspad-arr.txt' , format='%i') # , format='%f')

            quad = cp.confpars.cspadQuad
            arrQuad = self.plotsCSpad.getImageArrayForQuad( arr1ev, quadNum=quad )
            gm.saveNumpyArrayInFile(arrQuad, fname='cspad-ave-quad-' + str(quad) + '.txt' , format='%f') # , format='%i')


        if gm.ImageIsInTheName(dsname) :

            cameraName = gm.get_item_second_to_last_name(dsname)

            str_evt = '-ev%06d' % (self.eventStart) # cp.confpars.eventCurrent
            str_ave = '-av%d'   % (self.numEventsSelected) # cp.confpars.numEventsAverage
            fname_common = 'camera-' + str(cameraName) + str_ave + str_evt

            gm.saveNumpyArrayInFile(arr1ev, fname=fname_common + '.txt' , format='%f') # , format='%i')

            tiff_fname = fname_common + '.tiff'
            print 'Save image in TIFF file  ', tiff_fname
            scimisc.imsave(tiff_fname, arr1ev)


    def drawArrayForDSName(self, dsname, arr1ev) :

        cspadIsInTheName = gm.CSpadIsInTheName(dsname)
        imageIsInTheName = gm.ImageIsInTheName(dsname)
        item_last_name   = gm.get_item_last_name(dsname)
        cp.confpars.current_item_name_for_title = printh5.get_item_name_for_title(dsname)

        
        itemIsForDrawing = imageIsInTheName            or \
                           item_last_name=='waveforms' or \
                           cspadIsInTheName
        if not itemIsForDrawing : return


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

            
        #if dsname == self.dsnameCSpadV2 or dsname == self.dsnameCSpadV2CXI :
            #print 'Draw plots for CSpad V2'

        if cspadIsInTheName :

            #arr1ev # (32, 185, 388) <- format of this record
            self.getCSpadConfiguration(dsname)

            for self.nwin in range(cp.confpars.cspadNWindows) :

                cp.confpars.fillCSPadConfigParsNamedFromWin(self.nwin)

                if       cp.confpars.cspadWindowParameters[self.nwin][0] == dsname \
                      or cp.confpars.cspadWindowParameters[self.nwin][0] == 'All'  :

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

                    self.figNum += 1 
                    if cp.confpars.cspadSpectrumDetIsOn : 
                        self.plotsCSpad.plotCSpadV2Spectrum(arr1ev,self.set_fig('1x1'),plot='DetSpec')
                    else : self.close_fig(self.figNum)

                    self.figNum += 1 
                    if cp.confpars.cspadImageDetIsOn : 
                        self.plotsCSpad.plotCSpadV2Image(arr1ev,self.set_fig(4),plot='Det')
                    else : self.close_fig(self.figNum)


            self.figNum = 10*cp.confpars.cspadNWindowsMax

            self.figNum += 1 
            if cp.confpars.cspadProjXIsOn : 
                self.plotsCSpadProj.plotProjX(arr1ev,self.set_fig('1x1'))
            else : self.close_fig(self.figNum)

            self.figNum += 1 
            if cp.confpars.cspadProjYIsOn : 
                self.plotsCSpadProj.plotProjY(arr1ev,self.set_fig('1x1'))
            else : self.close_fig(self.figNum)

            self.figNum += 1 
            if cp.confpars.cspadProjRIsOn : 
                self.plotsCSpadProj.plotProjR(arr1ev,self.set_fig('1x1'))
            else : self.close_fig(self.figNum)

            self.figNum += 1 
            if cp.confpars.cspadProjPhiIsOn : 
                self.plotsCSpadProj.plotProjPhi(arr1ev,self.set_fig('1x1'))
            else : self.close_fig(self.figNum)
            

        if imageIsInTheName :

            arr2d = self.applyImageCorrections( arr1ev, dsname )

            for self.nwin in range(cp.confpars.imageNWindows) :

                if       cp.confpars.imageWindowParameters[self.nwin][0] == dsname \
                      or cp.confpars.imageWindowParameters[self.nwin][0] == 'All'  :

                    self.figNum += 1 
                    if cp.confpars.imageImageIsOn : 
                        self.plotsImage.plotImage(arr2d,self.set_fig('1x1',dsname))
                    else : self.close_fig(self.figNum)

                    self.figNum += 1 
                    if cp.confpars.imageSpectrumIsOn : 
                        self.plotsImage.plotImageSpectrum(arr2d,self.set_fig('1x1',dsname))
                    else : self.close_fig(self.figNum)

                    self.figNum += 1 
                    if cp.confpars.imageImageSpecIsOn : 
                        self.plotsImage.plotImageAndSpectrum(arr2d,self.set_fig('2x3',dsname))
                    else : self.close_fig(self.figNum)


            self.figNum += 1 
            if cp.confpars.imageProjXIsOn : 
                self.plotsImageProj.plotProjX(arr2d,self.set_fig('1x1'))
            else : self.close_fig(self.figNum)

            self.figNum += 1 
            if cp.confpars.imageProjYIsOn : 
                self.plotsImageProj.plotProjY(arr2d,self.set_fig('1x1'))
            else : self.close_fig(self.figNum)

            self.figNum += 1 
            if cp.confpars.imageProjRIsOn : 
                self.plotsImageProj.plotProjR(arr2d,self.set_fig('1x1'))
            else : self.close_fig(self.figNum)

            self.figNum += 1 
            if cp.confpars.imageProjPhiIsOn : 
                self.plotsImageProj.plotProjPhi(arr2d,self.set_fig('1x1'))
            else : self.close_fig(self.figNum)


        if item_last_name == 'waveforms' :

            for self.nwin in range(cp.confpars.waveformNWindows) :

                if dsname == cp.confpars.waveformWindowParameters[self.nwin][0] :

                    self.figNum += 1 
                    if cp.confpars.waveformWaveformIsOn : 
                        self.plotsWaveform.plotWFWaveform(arr1ev,self.set_fig('2x1'),self.h5file)
                    else : self.close_fig(self.figNum)


    def applyImageCorrections( self, arr2d1ev, dsname ):
        """Apply corrections to the image array:
        1) uint -> float32;
        2) subtract offset from data
        """
        offset_dsname = gm.get_item_path_to_last_name(dsname) + '/data'
        ds = self.h5file[offset_dsname]
        offset = ds['offset'][cp.confpars.eventCurrent] # !!! Indexes are reversed in order to use 'offset' as index
        if self.nwin == None :
            return np.array(arr2d1ev, dtype=float32) - offset
        else :
            return np.array(arr2d1ev, dtype=float32) - offset - cp.confpars.imageWindowParameters[self.nwin][10]


    def getCSpadConfiguration( self, dsname ):

        if gm.CSpadMiniElementIsInTheName(dsname) :
            print 'getCSpadConfiguration(...): This is a CSpadMiniElement. Special configuration is not required'
            cs.confcspad.isCSPad2x2 = True
            return

        #print 'Get quad numbers for each event from the dataset:' 
        el_dsname = gm.get_item_path_to_last_name(dsname) + '/element'
        #print el_dsname
        self.ds_element = self.h5file[el_dsname]

        cs.confcspad.quad_nums_in_event = self.ds_element[cp.confpars.eventCurrent]['quad']

        print 'quad_nums_in_event = ', cs.confcspad.quad_nums_in_event

        #if cp.confpars.fileName == self.fileNameWithAlreadySetCSpadConfiguration : return

        if cp.confpars.cspadImageDetIsOn    \
        or cp.confpars.cspadImageQuadIsOn   \
        or cp.confpars.cspadImageOfPairIsOn \
        or cp.confpars.cspadImageIsOn       \
        or cp.confpars.cspadProjXIsOn       \
        or cp.confpars.cspadProjYIsOn       \
        or cp.confpars.cspadProjRIsOn       \
        or cp.confpars.cspadProjPhiIsOn     \
        or cp.confpars.cspadSpectrumIsOn    \
        or cp.confpars.cspadSpectrumDetIsOn \
        or cp.confpars.cspadSpectrum08IsOn  : 

            item_second_to_last_name = printh5.get_item_second_to_last_name(dsname)
            #cspad_config_ds_name = self.dsnameCSpadV2Conf + item_second_to_last_name + '/config' 
            #cspad_config_ds_name = self.dsnameCSpadV3Conf + item_second_to_last_name + '/config' 

            #Find the configuration dataset name for CSPad
            self.dsConf = None
            for vers in range(100) :
                self.cspad_config_ds_name = self.dsnameCSpadVXConf + str(vers) + '/' + item_second_to_last_name + '/config' 
                try:
                    self.dsConf = self.h5file[self.cspad_config_ds_name]      # t=0.01us
                    break
                except KeyError:
                    #print 'Try another configuration dataset name'
                    continue

            if self.dsConf == None :
                print 80*'!'
                print 'WARNING: The CSPad configuration dataset is not found. Default versin will be used'
                print 80*'!'
                return

            print 'Configuration dataset name:', self.cspad_config_ds_name

            cs.confcspad.indPairsInQuads = self.dsConf['sections'] # For V2 it is dsConf.value[13], for V3 it is 15th  ...
            print "Indexes of pairs in quads =\n",cs.confcspad.indPairsInQuads 
            #self.fileNameWithAlreadySetCSpadConfiguration = cp.confpars.fileName




    def showEvent ( self, mode=1 ) :
        """showEvent: plt.show() or draw() depending on mode"""

        t_start = time.clock()
        if mode == 1 :   # Single event mode
            plt.show()  
        else :           # Slide show 
            plt.draw()   # Draws, but does not block
        print 'Time to show or draw (sec) = %f' % (time.clock() - t_start)


    def quitDrawEvent ( self ) :
        """Operations in case of quit drawing event(s)"""
        #self.plotsCSpad.close_fig1()
        #self.plotsImage.close_fig1()
        self.close_fig()
        plt.ioff()
        plt.close()
        self.closeHDF5File()
        #print 'quitDrawEvent()'


    def openHDF5File( self ) :     
        if not cp.confpars.h5_file_is_open :
            fname = cp.confpars.dirName+'/'+cp.confpars.fileName
            #print 'openHDF5File() : %s' % (fname)

            try: 
                self.h5file=  h5py.File(fname, 'r') # open read-only       
            except IOError:
                print 'IOError: No such file:', fname
                cp.confpars.h5_file_is_open = False
                return

            cp.confpars.h5_file_is_open = True
            #printh5.print_file_info(self.h5file)

            self.getTimeHDF5File()

            #if cp.confpars.fileName == self.fileNameWithAlreadySetCSpadConfiguration : return
            #else : cs.confcspad.setCSpadParameters()
            cs.confcspad.setCSpadParameters()


    def getTimeHDF5File( self ) :     
        conf_group = self.h5file['/Configure:0000']
        cs.confcspad.run_start_seconds = conf_group.attrs['start.seconds']

        print 'Run start (seconds) =', cs.confcspad.run_start_seconds
        tloc = time.localtime(cs.confcspad.run_start_seconds) # converts sec to the tuple struct_time in local
        print 'Run start local time :', time.strftime('%Y-%m-%d %H:%M:%S',tloc)
        #t1_sec = int( time.mktime((2011, 6, 23, 23, 22, 18, 0, 0, 0)) ) # converts date-tile to seconds


    def closeHDF5File( self ) :       
        if cp.confpars.h5_file_is_open :
            self.h5file.close()
            cp.confpars.h5_file_is_open = False
            #print 'closeHDF5File()'


    def set_fig( self, type=None, dsname=None ):
        """Set current fig."""

        ##self.figNum += 1 
        if self.figNum in self.list_of_open_figs :
            self.fig = plt.figure(num=self.figNum)        
        else :
            self.fig = self.open_fig(type,dsname)
            self.set_window_position()
            self.list_of_open_figs.append(self.figNum)
        return self.fig

               
    def open_fig( self, type=None, dsname=None ):
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

        if type == '2x3' :
            self.fig = plt.figure(num=self.figNum, figsize=(6,9), dpi=80, facecolor='w',edgecolor='w',frameon=True)
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

        self.fig.canvas.mpl_connect('close_event', self.processCloseEvent)

        self.fig.myXmin = None
        self.fig.myXmax = None
        self.fig.myYmin = None
        self.fig.myYmax = None
        self.fig.myCmin = None
        self.fig.myCmax = None
        self.fig.myarr  = None
        self.fig.mydsname = dsname
        #self.fig.myFigNum = self.number

        self.fig.myZoomIsOn = False
        self.fig.nwin   = self.nwin

        print 'Open figure number=', self.figNum, ' for window=', self.fig.nwin
        return self.fig


    def set_window_position( self ):
        """Move window in desired position."""
        #print 'set_window_position()'
        #plt.get_current_fig_manager().window.move(890, 100) #### This works!
        fig_QMainWindow = plt.get_current_fig_manager().window

        if self.figNum<100 :
            self.figOffsetNum = 0
            self.figOffsetX   = 460
        else :
            self.figOffsetNum = 100
            self.figOffsetX   = 500

        posx = cp.confpars.posGUIMain[0] + self.figOffsetX + 50*(self.figNum-self.figOffsetNum)
        posy = cp.confpars.posGUIMain[1]                   + 20*(self.figNum-self.figOffsetNum-1)
        fig_QMainWindow.move(posx,posy)
        #fig_QMainWindow.move(820+50*self.figNum, 20*(self.figNum-1)) #### This works!


    def processCloseEvent( self, event ):
        """Figure will be closed automatically, but it is necesary to remove its number from the list..."""
        fig    = event.canvas.figure # plt.gcf() does not work, because closed canva may be non active
        figNum = fig.number 
        print 'CloseEvent for figure number = ', figNum
        if figNum in self.list_of_open_figs : self.list_of_open_figs.remove(figNum)


    def close_fig( self, figNum=None ):
        """Close fig and its window."""
        if figNum==None :
            plt.close('all') # closes all the figure windows
        else :
            plt.close(figNum)
            if figNum in self.list_of_open_figs : self.list_of_open_figs.remove(figNum)

#-----------------------------------------

    def drawCorrelationPlots ( self ) :
        """Draw Correlation Plots"""

        if not cp.confpars.correlationsIsOn :
            print 'Check the "Correlations" checkbox in the "What to display?" GUI\n' +\
                  'and set the correlation plot(s) parameters.'
            return
        
        self.openHDF5File()
        if not cp.confpars.h5_file_is_open : return
        self.drawCorrelationPlotsFromOpenFile()
        self.closeHDF5File()

    def drawCorrelationPlotsFromOpenFile ( self ) :

        self.figNum = 100

        for self.nwin in range(cp.confpars.correlationNWindows) :

            self.figNum += 1 
            if cp.confpars.correlationsIsOn : 
                self.plotsCorrelations.plotCorrelations(self.set_fig('2x1'), self.h5file)
            else : self.close_fig(self.figNum)

#-----------------------------------------

    def drawCalibCyclePlots ( self ) :
        """Draw CalibCycle Plots"""

        if not cp.confpars.calibcycleIsOn :
            print 'Check the "CalibCycle" checkbox in the "What to display?" GUI\n' +\
                  'and set the correlation plot(s) parameters.'
            return
        
        self.openHDF5File()
        if not cp.confpars.h5_file_is_open : return
        self.drawCalibCyclePlotsFromOpenFile()
        self.closeHDF5File()

    def drawCalibCyclePlotsFromOpenFile ( self ) :

        self.figNum = 200

        print 'drawCalibCyclePlotsFromOpenFile for nwin=', cp.confpars.calibcycleNWindows

        for self.nwin in range(cp.confpars.calibcycleNWindows) :

            self.figNum += 1 
            if cp.confpars.calibcycleIsOn : 
                self.plotsCalibCycles.plotCalibCycles(self.set_fig('2x1'), self.h5file)
            else : self.close_fig(self.figNum)

#-----------------------------------------
#-----------------------------------------
#-----------------------------------------

    def drawWaveVsEventPlots ( self, dNcc=0 ) :
        """Draw waveform vs event plots"""

        if not cp.confpars.waveformWaveVsEvIsOn :
            print 'Check the "WF vs Event" checkbox in the "What to display?" GUI\n' +\
                  'and set the waveform plot parameters.'
            return
        
        self.openHDF5File()
        if not cp.confpars.h5_file_is_open : return

        dsname = cp.confpars.waveformWindowParameters[0][0]
        #Create the CalibCycles object and initialize it from open file
        self.CalibCOps = cc.CalibCycles()
        NccTotal = self.CalibCOps.extractNumberOfCalibCyclesFromOpenFile(self.h5file, dsname)
        print 'Total number of CalibCycles =', NccTotal
        self.drawWaveVsEventPlotsFromOpenFile(dNcc)        
        self.closeHDF5File()

    def drawWaveVsEventPlotsFromOpenFile ( self, dNcc=0 ) :

        self.figNum = 300

        print 'drawWaveVsEvPlotsFromOpenFile for nwin=', cp.confpars.waveformNWindows, ' dNcc=', dNcc

        for self.nwin in range(cp.confpars.waveformNWindows) :

            self.setDSNameForCalibCycle ( self.nwin, dNcc )

            self.figNum += 1 
            if cp.confpars.waveformWaveVsEvIsOn : 
                self.plotsWaveform.plotWaveVsEvent(self.set_fig('2x1'), self.h5file)
            else : self.close_fig(self.figNum)


    def setDSNameForCalibCycle ( self, nwin, dNcc=0 ) :
        if dNcc != 0 :
            dsname = cp.confpars.waveformWindowParameters[nwin][0]
            cp.confpars.waveformWindowParameters[nwin][0] = self.CalibCOps.getDSNameForCalibCycleDN(dsname, dNcc)
            print 'Plot for other CalibCycle:', cp.confpars.waveformWindowParameters[nwin][0]



#-----------------------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
