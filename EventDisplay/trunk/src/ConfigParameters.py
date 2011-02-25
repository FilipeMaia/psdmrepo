#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigParameters...
#
#------------------------------------------------------------------------

"""This module contains all configuration parameters for EventDisplay.

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
import os
#from PyQt4 import QtGui, QtCore
#---------------------------------
#  Imports of base class module --
#---------------------------------


#-----------------------------
# Imports for other modules --
#-----------------------------


#---------------------
#  Class definition --
#---------------------
class ConfigParameters ( object ) :
    """This class contains all configuration parameters

    @see BaseClass
    @see OtherClass
    """

    #--------------------
    #  Class variables --
    #--------------------
    publicStaticVariable = 0 
    __privateStaticVariable = "A string"

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor"""
        self.setDefaultParameters()


    def setDefaultParameters ( self ) :
        """Set default configuration parameters hardwired in this module"""

        print 'setDefaultParameters'

        # define default initial values of the configuration parameters

        #self.dirName         = '/reg/neh/home/dubrovin/LCLS/test_h5py'
        #self.fileName        = 'test.h5'
        #self.dirName              = '/reg/d/psdm/XPP/xppcom10/hdf5'
        #self.fileName             = 'xppcom10-r0546.h5'
        self.dirName              = '/reg/d/psdm/CXI/cxi80410/hdf5'
        self.fileName             = 'cxi80410-r0430.h5'
        self.eventCurrent         = 1
        self.span                 = 1
        self.numEventsAverage     = 50
        self.selectionIsOn        = False
        
        self.list_of_checked_item_names=[]
        self.list_of_checked_item_names.append('/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/CxiDs1.0:Cspad.0/data')
        self.list_of_checked_item_names.append('/Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/CxiSc1.0:Tm6740.0/image')       

        # Status parameters which do not need to be saved
        self.confParsDirName      = '.'
        self.confParsFileName     = 'evtdispconfig'

        self.h5_file_is_open      = False
        self.wtdWindowIsOpen      = False
        self.wtdIMWindowIsOpen    = False
        self.wtdCSWindowIsOpen    = False
        self.wtdWFWindowIsOpen    = False
        self.treeWindowIsOpen     = False
        self.treeViewIsExpanded   = False
        self.configGUIIsOpen      = False
        self.selectionGUIIsOpen   = False
        self.selectionWindowIsOpen= False
        self.posGUIMain           = (370,10)

        self.readParsFromFileAtStart = True
        self.current_item_name_for_title = 'Current extended item name for title'

        # Default parameters for CSpad plots
        self.cspadQuad            = 1
        self.cspadPair            = 1

        self.cspadAmplitudeRaMin  = 0
        self.cspadAmplitudeRange  = 2000

        self.cspadImageNWindows   = 1
        self.cspadImageNWindowsMax= 8 # Maximal number of windows for CSpad image which can be opened

        self.cspadImageOfPairIsOn = True
        self.cspadImageIsOn       = False
        self.cspadImageQuadIsOn   = False
        self.cspadImageDetIsOn    = True
        self.cspadSpectrumIsOn    = False
        self.cspadSpectrum08IsOn  = False
        self.cspadProjXYIsOn      = False
        self.cspadProjRIsOn       = False
        self.cspadProjPhiIsOn     = False

        self.cspadImageAmin       = 0   
        self.cspadImageAmax       = 2000
        self.cspadSpectrumAmin    = 0   
        self.cspadSpectrumAmax    = 2000
        self.cspadSpectrumRange   = None
        self.cspadSpectrumNbins   = 50
        self.cspadSpectrumBinWidth= 1
        self.cspadBinWidthIsOn    = True


        # Default parameters for Image plots
        self.imageImageIsOn       = False
        self.imageSpectrumIsOn    = False
        self.imageImageSpecIsOn   = True
        self.imageProjXYIsOn      = False
        self.imageProjRIsOn       = False
        self.imageProjPhiIsOn     = False

        self.imageImageAmin       = 0    #  15
        self.imageImageAmax       = 100  #  45

        self.imageSpectrumRange   = None # (15,45)
        self.imageSpectrumAmin    = 0    #  15
        self.imageSpectrumAmax    = 100  #  45
        self.imageSpectrumNbins   = 50   #  30
        self.imageAmplitudeRaMin  = 0
        self.imageAmplitudeRange  = 500

        # Default parameters for Waveform plots

        self.waveformWaveformIsOn  = True

        self.waveformNWindows      = 2
        self.waveformNWindowsMax   = 10 # Maximal number of windows for waveforms which can be opened

        self.waveformWindowParameters = []
        for win in range(self.waveformNWindowsMax) :
            self.waveformWindowParameters.append(['None', True, 0, 1000, 0, 1000, 0, None, None, None, None])
                        #[dataset, autoRangeIsOn, Amin, Amax, Tmin, Tmax, NumberOfWFInDS, WF1, WF2, WF3, WF4]


        # Default parameters for Selection algorithms
        self.selectionNWindows      = 2
        self.selectionNWindowsMax   = 10 # Maximal number of windows for selection algorithms

        self.selectionWindowParameters = []
        for win in range(self.selectionNWindowsMax) :
            self.selectionWindowParameters.append([0, True, 0, 1000, 0, 1000])
                                                 #[Theshold, InBin, Xmin, Xmax, Ymin, Ymax]


        self.perEventDistIsOn  = True
        self.correlationsIsOn  = True

        self.projCenterX       = 850
        self.projCenterY       = 850

        self.projR_BinWidthIsOn= True
        self.projR_SliWidthIsOn= True

        self.projR_NBins       = 100
        self.projR_BinWidth    = 10
        self.projR_NSlices     = 8
        self.projR_SliWidth    = 45
              
        self.projR_Rmin        = 0
        self.projR_Rmax        = 1000
        self.projR_Phimin      = 0
        self.projR_Phimax      = 360

        self.projPhi_BinWidthIsOn= True
        self.projPhi_SliWidthIsOn= True

        self.projPhi_NBins       = 36
        self.projPhi_BinWidth    = 10
        self.projPhi_NSlices     = 10
        self.projPhi_SliWidth    = 100
              
        self.projPhi_Rmin        = 0
        self.projPhi_Rmax        = 1000
        self.projPhi_Phimin      = 0
        self.projPhi_Phimax      = 360

    #-------------------
    #  Public methods --
    #-------------------

    def Print ( self ) :
        """Prints current values of configuration parameters
        """
        print '\nConfigParameters'
        print 'HDF5 file : %s' % ( self.dirName + '/' + self.fileName )
        print 'Event : %d and span : %d \n' % ( self.eventCurrent, self.span )
        print 'Number of items to plot =', len(self.list_of_checked_item_names)
        for name in self.list_of_checked_item_names :
            print str(name)

        print 'CSPAD_QUAD_NUMBER',         self.cspadQuad
        print 'CSPAD_PAIR_NUMBER',         self.cspadPair
        print 'CSPAD_IMAGE_IS_ON',         self.cspadImageIsOn       
        print 'CSPAD_IMAGE_OF_PAIR_IS_ON', self.cspadImageOfPairIsOn
        print 'CSPAD_IMAGE_QUAD_IS_ON',    self.cspadImageQuadIsOn
        print 'CSPAD_IMAGE_DET_IS_ON',     self.cspadImageDetIsOn
        print 'CSPAD_SPECT_IS_ON',         self.cspadSpectrumIsOn    
        print 'CSPAD_SPE08_IS_ON',         self.cspadSpectrum08IsOn    
        print 'CSPAD_PROJ_XY_IS_ON',       self.cspadProjXYIsOn    
        print 'CSPAD_PROJ_R_IS_ON',        self.cspadProjRIsOn    
        print 'CSPAD_PROJ_PHI_IS_ON',      self.cspadProjPhiIsOn    

        print 'CSPAD_IMAGE_AMIN',          self.cspadImageAmin
        print 'CSPAD_IMAGE_AMAX',          self.cspadImageAmax
        print 'CSPAD_SPECT_AMIN',          self.cspadSpectrumAmin
        print 'CSPAD_SPECT_AMAX',          self.cspadSpectrumAmax

        print 'IMAGE_IMAGE_IS_ON',         self.imageImageIsOn       
        print 'IMAGE_IMAGE_SPEC_IS_ON',    self.imageImageSpecIsOn       
        print 'IMAGE_SPECT_IS_ON',         self.imageSpectrumIsOn    
        print 'IMAGE_PROJ_XY_IS_ON',       self.imageProjXYIsOn    
        print 'IMAGE_PROJ_R_IS_ON',        self.imageProjRIsOn    
        print 'IMAGE_PROJ_PHI_IS_ON',      self.imageProjPhiIsOn    

        print 'READ_PARS_AT_START',        self.readParsFromFileAtStart

        print 'WAVEF_WAVEF_IS_ON',         self.waveformWaveformIsOn    

        print 'WAVEF_N_WINDOWS_MAX',       self.waveformNWindowsMax 
        print 'WAVEF_N_WINDOWS',           self.waveformNWindows 

        for win in range(self.waveformNWindows) :

            print 'WAVEF_WINDOW_NUMBER',   win 
            print 'WAVEF_DATASET',         self.waveformWindowParameters[win][0] 
            print 'WAVEF_AUTO_RANGE_IS_ON',self.waveformWindowParameters[win][1] 
            print 'WAVEF_AMIN',            self.waveformWindowParameters[win][2] 
            print 'WAVEF_AMAX',            self.waveformWindowParameters[win][3] 
            print 'WAVEF_TMIN',            self.waveformWindowParameters[win][4] 
            print 'WAVEF_TMAX',            self.waveformWindowParameters[win][5] 
            print 'WAVEF_N_WF_IN_DATA_SET',self.waveformWindowParameters[win][6] 
            print 'WAVEF_IND_WF_IN_BLACK', self.waveformWindowParameters[win][7] 
            print 'WAVEF_IND_WF_IN_RED',   self.waveformWindowParameters[win][8] 
            print 'WAVEF_IND_WF_IN_GREEN', self.waveformWindowParameters[win][9] 
            print 'WAVEF_IND_WF_IN_BLUE',  self.waveformWindowParameters[win][10] 

        print 'SELEC_N_WINDOWS_MAX',       self.selectionNWindowsMax 
        print 'SELEC_N_WINDOWS',           self.selectionNWindows 

        for win in range(self.selectionNWindows) :

            print 'SELEC_WINDOW_NUMBER',   win 
            print 'SELEC_THRESHOLD',       self.selectionWindowParameters[win][0] 
            print 'SELEC_IN_BIN',          self.selectionWindowParameters[win][1] 
            print 'SELEC_XMIN',            self.selectionWindowParameters[win][2] 
            print 'SELEC_XMAX',            self.selectionWindowParameters[win][3] 
            print 'SELEC_YMIN',            self.selectionWindowParameters[win][4] 
            print 'SELEC_YMAX',            self.selectionWindowParameters[win][5] 

        print 'NUM_EVENTS_FOR_AVERAGE',    self.numEventsAverage
        print 'SELECTION_IS_ON',           self.selectionIsOn

        print 'PER_EVENT_DIST_IS_ON',      self.perEventDistIsOn
        print 'CORRELATIONS_IS_ON',        self.correlationsIsOn

        print 70*'='


    def readParameters(self, fname=None) :
        self.__setConfigParsFileName(fname)        
        print 'Read parameters from file:', self._fname
        dicBool = {'false':False, 'true':True}
        if os.path.exists(self._fname) :
            f=open(self._fname,'r')
            self.list_of_checked_item_names = []
            for line in f :
                if len(line) == 1 : continue # line is empty
                key = line.split()[0]
                val = line.split()[1]
                if   key == 'HDF5_FILE_NAME'           : self.dirName,self.fileName = os.path.split(val)
                elif key == 'N_CHECKED_ITEMS'          : number_of_items = int(val)
                elif key == 'ITEM_NAME'                : self.list_of_checked_item_names.append(val) 
                elif key == 'CURRENT_EVENT'            : self.eventCurrent = int(val)
                elif key == 'SPAN'                     : self.span = int(val)
                elif key == 'NUM_EVENTS_FOR_AVERAGE'   : self.numEventsAverage        = int(val)
                elif key == 'SELECTION_IS_ON'          : self.selectionIsOn           = dicBool[val.lower()]

                elif key == 'CSPAD_IMAGE_IS_ON'        : self.cspadImageIsOn          = dicBool[val.lower()]
                elif key == 'CSPAD_IMAGE_OF_PAIR_IS_ON': self.cspadImageOfPairIsOn    = dicBool[val.lower()]
                elif key == 'CSPAD_IMAGE_QUAD_IS_ON'   : self.cspadImageQuadIsOn      = dicBool[val.lower()]
                elif key == 'CSPAD_IMAGE_DET_IS_ON'    : self.cspadImageDetIsOn       = dicBool[val.lower()]
                elif key == 'CSPAD_SPECT_IS_ON'        : self.cspadSpectrumIsOn       = dicBool[val.lower()]
                elif key == 'CSPAD_SPE08_IS_ON'        : self.cspadSpectrum08IsOn     = dicBool[val.lower()]
                elif key == 'CSPAD_PROJ_XY_IS_ON'      : self.cspadProjXYIsOn         = dicBool[val.lower()]
                elif key == 'CSPAD_PROJ_R_IS_ON'       : self.cspadProjRIsOn          = dicBool[val.lower()]
                elif key == 'CSPAD_PROJ_PHI_IS_ON'     : self.cspadProjPhiIsOn        = dicBool[val.lower()]
                elif key == 'IMAGE_IMAGE_IS_ON'        : self.imageImageIsOn          = dicBool[val.lower()]
                elif key == 'IMAGE_IMAGE_SPEC_IS_ON'   : self.imageImageSpecIsOn      = dicBool[val.lower()]
                elif key == 'IMAGE_SPECT_IS_ON'        : self.imageSpectrumIsOn       = dicBool[val.lower()]
                elif key == 'IMAGE_PROJ_XY_IS_ON'      : self.imageProjXYIsOn         = dicBool[val.lower()]
                elif key == 'IMAGE_PROJ_R_IS_ON'       : self.imageProjRIsOn          = dicBool[val.lower()]
                elif key == 'IMAGE_PROJ_PHI_IS_ON'     : self.imageProjPhiIsOn        = dicBool[val.lower()]

                elif key == 'WAVEF_WAVEF_IS_ON'        : self.waveformWaveformIsOn    = dicBool[val.lower()]
                elif key == 'READ_PARS_AT_START'       : self.readParsFromFileAtStart = dicBool[val.lower()]
                elif key == 'CSPAD_QUAD_NUMBER'        : self.cspadQuad               = int(val)
                elif key == 'CSPAD_PAIR_NUMBER'        : self.cspadPair               = int(val)
                elif key == 'CSPAD_IMAGE_AMIN'         : self.cspadImageAmin          = int(val)
                elif key == 'CSPAD_IMAGE_AMAX'         : self.cspadImageAmax          = int(val)
                elif key == 'CSPAD_SPECT_AMIN'         : self.cspadSpectrumAmin       = int(val)
                elif key == 'CSPAD_SPECT_AMAX'         : self.cspadSpectrumAmax       = int(val)

                elif key == 'PER_EVENT_DIST_IS_ON'     : self.perEventDistIsOn        = dicBool[val.lower()]
                elif key == 'CORRELATIONS_IS_ON'       : self.correlationsIsOn        = dicBool[val.lower()]

                elif key == 'WAVEF_N_WINDOWS_MAX'      : self.waveformNWindowsMax     = int(val)
                elif key == 'WAVEF_N_WINDOWS'          : self.waveformNWindows        = int(val)

                elif key == 'WAVEF_WINDOW_NUMBER'      : win                          = int(val)
                elif key == 'WAVEF_DATASET'            : self.waveformWindowParameters[win][0] = val
                elif key == 'WAVEF_AUTO_RANGE_IS_ON'   : self.waveformWindowParameters[win][1] = dicBool[val.lower()]
                elif key == 'WAVEF_AMIN'               : self.waveformWindowParameters[win][2] = int(val)
                elif key == 'WAVEF_AMAX'               : self.waveformWindowParameters[win][3] = int(val)
                elif key == 'WAVEF_TMIN'               : self.waveformWindowParameters[win][4] = int(val)
                elif key == 'WAVEF_TMAX'               : self.waveformWindowParameters[win][5] = int(val)
                elif key == 'WAVEF_N_WF_IN_DATA_SET'   : self.waveformWindowParameters[win][6] = self.getValIntOrNone(val)
                elif key == 'WAVEF_IND_WF_IN_BLACK'    : self.waveformWindowParameters[win][7] = self.getValIntOrNone(val)
                elif key == 'WAVEF_IND_WF_IN_RED'      : self.waveformWindowParameters[win][8] = self.getValIntOrNone(val)
                elif key == 'WAVEF_IND_WF_IN_GREEN'    : self.waveformWindowParameters[win][9] = self.getValIntOrNone(val)
                elif key == 'WAVEF_IND_WF_IN_BLUE'     : self.waveformWindowParameters[win][10]= self.getValIntOrNone(val)


                elif key == 'SELEC_N_WINDOWS_MAX'      : self.selectionNWindowsMax     = int(val)
                elif key == 'SELEC_N_WINDOWS'          : self.selectionNWindows        = int(val)

                elif key == 'SELEC_WINDOW_NUMBER'      : win                           = int(val)
                elif key == 'SELEC_THRESHOLD'          : self.selectionWindowParameters[win][0] = int(val)
                elif key == 'SELEC_IN_BIN'             : self.selectionWindowParameters[win][1] = dicBool[val.lower()]
                elif key == 'SELEC_XMIN'               : self.selectionWindowParameters[win][2] = int(val)
                elif key == 'SELEC_XMAX'               : self.selectionWindowParameters[win][3] = int(val)
                elif key == 'SELEC_YMIN'               : self.selectionWindowParameters[win][4] = int(val)
                elif key == 'SELEC_YMAX'               : self.selectionWindowParameters[win][5] = int(val)

                else : print 'The record : %s %s \n is UNKNOWN in readParameters()' % (key, val) 
            f.close()
        else :
            print 'The file %s does not exist' % (fname)
            print 'WILL USE DEFAULT CONFIGURATION PARAMETERS'


    def getValIntOrNone(self,val) :
        if val == 'None' : return None
        else :             return int(val)


    def writeParameters(self, fname=None) :
        self.__setConfigParsFileName(fname)        
        print 'Write parameters in file:', self._fname
        space = '    '
        f=open(self._fname,'w')
        f.write('HDF5_FILE_NAME'            + space + self.dirName + '/' + self.fileName + '\n')
        f.write('N_CHECKED_ITEMS'           + space + str(len(self.list_of_checked_item_names)) + '\n')
        for name in self.list_of_checked_item_names :
            f.write('ITEM_NAME'             + space + str(name)                         + '\n')
        f.write('CURRENT_EVENT'             + space + str(self.eventCurrent)            + '\n')
        f.write('SPAN'                      + space + str(self.span)                    + '\n')
        f.write('CSPAD_IMAGE_IS_ON'         + space + str(self.cspadImageIsOn)          + '\n')
        f.write('CSPAD_IMAGE_OF_PAIR_IS_ON' + space + str(self.cspadImageOfPairIsOn)    + '\n')
        f.write('CSPAD_IMAGE_QUAD_IS_ON'    + space + str(self.cspadImageQuadIsOn)      + '\n')
        f.write('CSPAD_IMAGE_DET_IS_ON'     + space + str(self.cspadImageDetIsOn)       + '\n')
        f.write('CSPAD_SPECT_IS_ON'         + space + str(self.cspadSpectrumIsOn)       + '\n')
        f.write('CSPAD_SPE08_IS_ON'         + space + str(self.cspadSpectrum08IsOn)     + '\n')
        f.write('CSPAD_PROJ_XY_IS_ON'       + space + str(self.cspadProjXYIsOn)         + '\n')
        f.write('CSPAD_PROJ_R_IS_ON'        + space + str(self.cspadProjRIsOn)          + '\n')
        f.write('CSPAD_PROJ_PHI_IS_ON'      + space + str(self.cspadProjPhiIsOn)        + '\n')
        f.write('IMAGE_IMAGE_IS_ON'         + space + str(self.imageImageIsOn)          + '\n')
        f.write('IMAGE_IMAGE_SPEC_IS_ON'    + space + str(self.imageImageSpecIsOn)      + '\n')
        f.write('IMAGE_SPECT_IS_ON'         + space + str(self.imageSpectrumIsOn)       + '\n')
        f.write('IMAGE_PROJ_XY_IS_ON'       + space + str(self.imageProjXYIsOn)         + '\n')
        f.write('IMAGE_PROJ_R_IS_ON'        + space + str(self.imageProjRIsOn)          + '\n')
        f.write('IMAGE_PROJ_PHI_IS_ON'      + space + str(self.imageProjPhiIsOn)        + '\n')
        f.write('WAVEF_WAVEF_IS_ON'         + space + str(self.waveformWaveformIsOn)    + '\n')
        f.write('READ_PARS_AT_START'        + space + str(self.readParsFromFileAtStart) + '\n')
        f.write('CSPAD_QUAD_NUMBER'         + space + str(self.cspadQuad)               + '\n')
        f.write('CSPAD_PAIR_NUMBER'         + space + str(self.cspadPair)               + '\n')
        f.write('CSPAD_IMAGE_AMIN'          + space + str(self.cspadImageAmin)          + '\n')
        f.write('CSPAD_IMAGE_AMAX'          + space + str(self.cspadImageAmax)          + '\n')
        f.write('CSPAD_SPECT_AMIN'          + space + str(self.cspadSpectrumAmin)       + '\n')
        f.write('CSPAD_SPECT_AMAX'          + space + str(self.cspadSpectrumAmax)       + '\n')
        f.write('PER_EVENT_DIST_IS_ON'      + space + str(self.perEventDistIsOn)        + '\n')
        f.write('CORRELATIONS_IS_ON'        + space + str(self.correlationsIsOn)        + '\n')

        f.write('\n')
        f.write('WAVEF_N_WINDOWS_MAX'       + space + str(self.waveformNWindowsMax)     + '\n')
        f.write('WAVEF_N_WINDOWS'           + space + str(self.waveformNWindows)        + '\n')

        for win in range(self.waveformNWindows) :
            f.write('\n')
            f.write('WAVEF_WINDOW_NUMBER'   + space + str(win)                                          + '\n')
            f.write('WAVEF_DATASET'         + space + str(self.waveformWindowParameters[win][0] )       + '\n')
            f.write('WAVEF_AUTO_RANGE_IS_ON'+ space + str(self.waveformWindowParameters[win][1] )       + '\n')
            f.write('WAVEF_AMIN'            + space + str(self.waveformWindowParameters[win][2] )       + '\n')
            f.write('WAVEF_AMAX'            + space + str(self.waveformWindowParameters[win][3] )       + '\n')
            f.write('WAVEF_TMIN'            + space + str(self.waveformWindowParameters[win][4] )       + '\n')
            f.write('WAVEF_TMAX'            + space + str(self.waveformWindowParameters[win][5] )       + '\n')
            f.write('WAVEF_N_WF_IN_DATA_SET'+ space + str(self.waveformWindowParameters[win][6] )       + '\n')
            f.write('WAVEF_IND_WF_IN_BLACK' + space + str(self.waveformWindowParameters[win][7] )       + '\n')
            f.write('WAVEF_IND_WF_IN_RED'   + space + str(self.waveformWindowParameters[win][8] )       + '\n')
            f.write('WAVEF_IND_WF_IN_GREEN' + space + str(self.waveformWindowParameters[win][9] )       + '\n')
            f.write('WAVEF_IND_WF_IN_BLUE'  + space + str(self.waveformWindowParameters[win][10])       + '\n')

        f.write('SELEC_N_WINDOWS_MAX'       + space + str(self.selectionNWindowsMax)     + '\n')
        f.write('SELEC_N_WINDOWS'           + space + str(self.selectionNWindows)        + '\n')

        for win in range(self.selectionNWindows) :
            f.write('\n')
            f.write('SELEC_WINDOW_NUMBER'   + space + str(win)                                           + '\n')
            f.write('SELEC_THRESHOLD'       + space + str(self.selectionWindowParameters[win][0] )       + '\n')
            f.write('SELEC_IN_BIN'          + space + str(self.selectionWindowParameters[win][1] )       + '\n')
            f.write('SELEC_XMIN'            + space + str(self.selectionWindowParameters[win][2] )       + '\n')
            f.write('SELEC_XMAX'            + space + str(self.selectionWindowParameters[win][3] )       + '\n')
            f.write('SELEC_YMIN'            + space + str(self.selectionWindowParameters[win][4] )       + '\n')
            f.write('SELEC_YMAX'            + space + str(self.selectionWindowParameters[win][5] )       + '\n')
            f.write('NUM_EVENTS_FOR_AVERAGE'+ space + str( self.numEventsAverage )                       + '\n')
            f.write('SELECTION_IS_ON'       + space + str( self.selectionIsOn )                          + '\n')
        f.close()


    def __setConfigParsFileName(self, fname=None) :
        if fname == None :
            self._fname = self.confParsDirName + '/' + self.confParsFileName
        else :
            self._fname = fname

#---------------------------------------
# Makes a single object of this class --
#---------------------------------------

confpars = ConfigParameters()

#----------------------------------------------
# In case someone decides to run this module --
#----------------------------------------------
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )

#----------------------------------------------
