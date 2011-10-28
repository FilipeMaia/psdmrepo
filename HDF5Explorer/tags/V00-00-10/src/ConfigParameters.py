#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigParameters...
#
#------------------------------------------------------------------------

"""This module contains all configuration parameters for HDF5Explorer.

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
        self.setRunTimeParametersInit()
        self.setDefaultParameters()


    def setRunTimeParametersInit ( self ) :

        self.h5_file_is_open         = False
        self.wtdWindowIsOpen         = False
        self.wtdIMWindowIsOpen       = False
        self.wtdCSWindowIsOpen       = False
        self.wtdWFWindowIsOpen       = False
        self.treeWindowIsOpen        = False
        self.treeViewIsExpanded      = False
        self.configGUIIsOpen         = False
        self.selectionGUIIsOpen      = False
        self.selectionWindowIsOpen   = False
        self.correlationGUIIsOpen    = False
        self.calibcycleGUIIsOpen     = False
        self.playerGUIIsOpen         = False
        self.bkgdGUIIsOpen           = False
        self.gainGUIIsOpen           = False

        self.step01IsDone            = False
        self.step02IsDone            = False
        self.step03IsDone            = False
        self.step04IsDone            = False


    def setDefaultParameters ( self ) :
        """Set default configuration parameters hardwired in this module"""

        print 'setDefaultParameters'

        # define default initial values of the configuration parameters

        #self.dirName         = '/reg/neh/home/dubrovin/LCLS/test_h5py'
        #self.fileName        = 'test.h5'
        #self.dirName              = '/reg/d/psdm/XPP/xppcom10/hdf5'
        #self.fileName             = 'xppcom10-r0546.h5'

        self.dirName              = '/reg/d/psdm/CXI/cxi80410/hdf5'
        self.fileName             = 'cxi80410-r0730.h5'

        self.eventCurrent         = 0
        self.span                 = 1
        self.numEventsAverage     = 50
        self.selectionIsOn        = False
        
        self.list_of_checked_item_names=[]
        self.list_of_checked_item_names.append('/Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/CxiSc1.0:Tm6740.1/image')       
        #self.list_of_checked_item_names.append('/Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/CxiSc1.0:Tm6740.2/image')
        #self.list_of_checked_item_names.append('/Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/CxiSc1.0:Tm6740.3/image')

        # Status parameters which do not need to be saved
        self.confParsDirName         = '.'
        self.confParsFileName        = 'hdf5expconfig'

        self.posGUIMain              = (370,10)

        self.readParsFromFileAtStart = True
        self.current_item_name_for_title = 'Current extended item name for title'

        # Default parameters for CSpad plots

        self.cspadNWindows   = 1
        self.cspadNWindowsMax= 8 # Maximal number of windows for CSpad which can be opened

        self.cspadWindowParameters = []
        for win in range(self.cspadNWindowsMax) :
            self.cspadWindowParameters.append(['All', 0, 100, 0, 100, 100, 1, True, True, False, 1, 1])
            #[dataset, ImAmin, ImAmax, SpAmin, SpAmax, SpNBins, SpBinWidth, ImALimsIsOn, SpALimsIsOn, SpBinWidthIsOn, quad, pair]
            #[0,       1,      2,      3,      4,      5,       6,          7,           8,           9,              10,   11]

        self.cspadImageOfPairIsOn = False
        self.cspadImageIsOn       = False
        self.cspadImageQuadIsOn   = False
        self.cspadImageDetIsOn    = False
        self.cspadSpectrumIsOn    = False
        self.cspadSpectrumDetIsOn = False
        self.cspadSpectrum08IsOn  = False
        self.cspadProjXIsOn       = False
        self.cspadProjYIsOn       = False
        self.cspadProjRIsOn       = False
        self.cspadProjPhiIsOn     = False
        self.cspadApplyTiltAngle  = False # Is used in PlotsForCSpad.py

        self.cspadCurrentDSName   = 'None'
        self.cspadAmplitudeRaMin  = 0
        self.cspadAmplitudeRange  = 2000
        self.cspadImageAmin       = 0   
        self.cspadImageAmax       = 2000
        self.cspadSpectrumAmin    = 0   
        self.cspadSpectrumAmax    = 2000
        self.cspadSpectrumNbins   = 50
        self.cspadSpectrumBinWidth= 1
        self.cspadImageLimsIsOn   = True
        self.cspadSpectLimsIsOn   = True
        self.cspadBinWidthIsOn    = True
        self.cspadQuad            = 1
        self.cspadPair            = 1

        # Default parameters for Image plots
        self.imageImageIsOn       = False
        self.imageSpectrumIsOn    = False
        self.imageImageSpecIsOn   = True
        self.imageProjXIsOn       = False
        self.imageProjYIsOn       = False
        self.imageProjRIsOn       = False
        self.imageProjPhiIsOn     = False

        self.imageAmplitudeRaMin  = 0
        self.imageAmplitudeRange  = 500

        self.imageImageAmin       = 0    #  15
        self.imageImageAmax       = 100  #  45
        self.imageSpectrumAmin    = 0    #  15
        self.imageSpectrumAmax    = 100  #  45
        self.imageSpectrumNbins   = 50   #  30
        self.imageSpectrumBinWidth= 1 
        self.imageBinWidthIsOn    = True
        self.imageDataset         = 'All'
       #self.imageSpectrumRange   = None # (15,45)

        # Default parameters for Image plots

        self.imageNWindows      = 1
        self.imageNWindowsMax   = 10 # Maximal number of windows for waveforms which can be opened

        self.imageWindowParameters = []
        for win in range(self.imageNWindowsMax) :
            self.imageWindowParameters.append(['All', 0, 100, 0, 100, 100, 1, False, False, False, 0])
            #[dataset, ImAmin, ImAmax, SpAmin, SpAmax, SpNBins, SpBinWidth, ImALimsIsOn, SpALimsIsOn, SpBinWidthIsOn, ImOffset]
            #[0,       1,      2,      3,      4,      5,       6,          7,           8,           9,              10]


        # Default parameters for Waveform plots
        self.waveformWaveformIsOn  = False
        self.waveformWaveVsEvIsOn  = False

        self.waveformNWindows      = 1
        self.waveformNWindowsMax   = 10 # Maximal number of windows for waveforms which can be opened

        self.waveformWindowParameters = []
        for win in range(self.waveformNWindowsMax) :
            self.waveformWindowParameters.append(['None', 0, 0, 1000, 0, 1000, 0, None, None, None, None])
                        #[dataset, rangeUnitsBits, Amin, Amax, Tmin, Tmax, NumberOfWFInDS, WF1, WF2, WF3, WF4]
        #rangeUnitsBits : 1-ALimits, 2-TLimits, 4-AUnits, 8-TUnits


        # Default parameters for Selection algorithms
        self.selectionNWindows      = 1
        self.selectionNWindowsMax   = 10 # Maximal number of windows for selection algorithms

        self.selectionWindowParameters = []
        for win in range(self.selectionNWindowsMax) :
            self.selectionWindowParameters.append([0, True, 0, 1000, 0, 1000, 'None'])
                                                 #[Theshold, InBin, Xmin, Xmax, Ymin, Ymax]


        # Default parameters for Correlation algorithms
        self.correlationsIsOn         = False

        self.correlationNWindows      = 1
        self.correlationNWindowsMax   = 10 # Maximal number of windows for selection algorithms

        self.correlationWindowParameters = []
        for win in range(self.correlationNWindowsMax) :
            self.correlationWindowParameters.append(['None','None',    0,    0, 1000,    0, 1000,'None','None', False,     False,     False,     40,     40,    'None', 'None'])
                                                   #[ Ydsn,  Xdsn, Radio, Ymin, Ymax, Xmin, Xmax, Ypar,  Xpar,  YLimsIsOn, XLimsIsOn, LogZIsOn, YNBins, XNBins, YparInd, XparInd]
                                                   #[    0,     1,     2,    3,    4,    5,    6,    7,     8,     9,      10,        11,       12,     13      14,      15]



        # Default parameters for CalibCycle algorithms
        self.calibcycleIsOn          = False

        self.calibcycleNWindows      = 1
        self.calibcycleNWindowsMax   = 10 # Maximal number of windows for selection algorithms

        self.calibcycleWindowParameters = []
        for win in range(self.calibcycleNWindowsMax) :
            self.calibcycleWindowParameters.append(['None','None',    0,    0, 1000,    0, 1000,'None','None', False,     False,     False,     40,     40,    'None', 'None'])
                                                  #[ Ydsn,  Xdsn, Radio, Ymin, Ymax, Xmin, Xmax, Ypar,  Xpar,  YLimsIsOn, XLimsIsOn, LogZIsOn, YNBins, XNBins, YparInd, XparInd]
                                                  #[    0,     1,     2,    3,    4,    5,    6,    7,     8,     9,      10,        11,       12,     13,     14,      15]

        self.projCenterX         = 850
        self.projCenterY         = 850

        self.projR_BinWidthIsOn  = True
        self.projR_SliWidthIsOn  = True
        self.projR_NBins         = 100
        self.projR_BinWidth      = 10
        self.projR_NSlices       = 8
        self.projR_SliWidth      = 45
        self.projR_Rmin          = 0
        self.projR_Rmax          = 1000
        self.projR_Phimin        = 0
        self.projR_Phimax        = 360

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


        self.projX_BinWidthIsOn  = True
        self.projX_SliWidthIsOn  = True
        self.projX_NBins         = 100  
        self.projX_BinWidth      = 20  
        self.projX_NSlices       = 10  
        self.projX_SliWidth      = 200 
        self.projX_Xmin          = 0   
        self.projX_Xmax          = 400
        self.projX_Ymin          = 0   
        self.projX_Ymax          = 400
                                 
        self.projY_BinWidthIsOn  = True
        self.projY_SliWidthIsOn  = True
        self.projY_NBins         = 100 
        self.projY_BinWidth      = 20  
        self.projY_NSlices       = 10  
        self.projY_SliWidth      = 200 
        self.projY_Xmin          = 0   
        self.projY_Xmax          = 400
        self.projY_Ymin          = 0   
        self.projY_Ymax          = 400

        # Default parameters for the background subtraction
        self.arr_bkgd             = 0
        self.bkgdSubtractionIsOn  = False
        self.bkgdDirName          = '.'
        self.bkgdFileName         = 'cspad-bkgd.txt'

        # Default parameters for the gain correction
        self.arr_gain             = 0
        self.gainCorrectionIsOn   = False
        self.gainDirName          = '.'
        self.gainFileName         = 'cspad-gain.txt'

        self.aveDirName           = '.'
        self.aveFileName          = 'cspad-ave.txt'

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

        print 'PLAYER_GUI_IS_OPEN',        self.playerGUIIsOpen

        print 'STEP_01_IS_DONE',           self.step01IsDone
        print 'STEP_02_IS_DONE',           self.step02IsDone
        print 'STEP_03_IS_DONE',           self.step03IsDone
        print 'STEP_04_IS_DONE',           self.step04IsDone
        
        print 'CSPAD_IMAGE_IS_ON',         self.cspadImageIsOn       
        print 'CSPAD_IMAGE_OF_PAIR_IS_ON', self.cspadImageOfPairIsOn
        print 'CSPAD_IMAGE_QUAD_IS_ON',    self.cspadImageQuadIsOn
        print 'CSPAD_IMAGE_DET_IS_ON',     self.cspadImageDetIsOn
        print 'CSPAD_SPECT_IS_ON',         self.cspadSpectrumIsOn    
        print 'CSPAD_SPECT_DET_IS_ON',     self.cspadSpectrumDetIsOn    
        print 'CSPAD_SPE08_IS_ON',         self.cspadSpectrum08IsOn    
        print 'CSPAD_PROJ_X_IS_ON',        self.cspadProjXIsOn    
        print 'CSPAD_PROJ_Y_IS_ON',        self.cspadProjYIsOn    
        print 'CSPAD_PROJ_R_IS_ON',        self.cspadProjRIsOn    
        print 'CSPAD_PROJ_PHI_IS_ON',      self.cspadProjPhiIsOn    
        print 'CSPAD_APPLY_TILT_ANGLE',    self.cspadApplyTiltAngle
        
        print 'CSPAD_RANGE_AMIN',          self.cspadAmplitudeRaMin   
        print 'CSPAD_RANGE_AMAX',          self.cspadAmplitudeRange            
        print 'CSPAD_IMAGE_AMIN',          self.cspadImageAmin
        print 'CSPAD_IMAGE_AMAX',          self.cspadImageAmax
        print 'CSPAD_SPECT_AMIN',          self.cspadSpectrumAmin
        print 'CSPAD_SPECT_AMAX',          self.cspadSpectrumAmax
        print 'CSPAD_SPECT_NBINS',         self.cspadSpectrumNbins    
        print 'CSPAD_SPECT_BIN_WIDTH',     self.cspadSpectrumBinWidth 
        print 'CSPAD_IM_LIMITS_IS_ON',     self.cspadImageLimsIsOn
        print 'CSPAD_SP_LIMITS_IS_ON',     self.cspadSpectLimsIsOn
        print 'CSPAD_BIN_WIDTH_IS_ON',     self.cspadBinWidthIsOn     
        print 'CSPAD_QUAD_NUMBER',         self.cspadQuad
        print 'CSPAD_PAIR_NUMBER',         self.cspadPair

        print 'CSPAD_N_WINDOWS_MAX',       self.cspadNWindowsMax 
        print 'CSPAD_N_WINDOWS',           self.cspadNWindows 

        for win in range(self.cspadNWindows) :

            print 'CSPAD_WINDOW_NUMBER',   win 
            print 'CSPAD_DATASET',         self.cspadWindowParameters[win][0] 
            print 'CSPAD_IMAGE_AMIN',      self.cspadWindowParameters[win][1] 
            print 'CSPAD_IMAGE_AMAX',      self.cspadWindowParameters[win][2] 
            print 'CSPAD_SPECT_AMIN',      self.cspadWindowParameters[win][3] 
            print 'CSPAD_SPECT_AMAX',      self.cspadWindowParameters[win][4]         
            print 'CSPAD_SPECT_NBINS',     self.cspadWindowParameters[win][5] 
            print 'CSPAD_SPECT_BIN_WIDTH', self.cspadWindowParameters[win][6] 
            print 'CSPAD_IM_LIMITS_IS_ON', self.cspadWindowParameters[win][7] 
            print 'CSPAD_SP_LIMITS_IS_ON', self.cspadWindowParameters[win][8] 
            print 'CSPAD_BIN_WIDTH_IS_ON', self.cspadWindowParameters[win][9] 
            print 'CSPAD_QUAD_NUMBER',     self.cspadWindowParameters[win][10]
            print 'CSPAD_PAIR_NUMBER',     self.cspadWindowParameters[win][11]


        print 'IMAGE_IMAGE_IS_ON',         self.imageImageIsOn       
        print 'IMAGE_IMAGE_SPEC_IS_ON',    self.imageImageSpecIsOn       
        print 'IMAGE_SPECT_IS_ON',         self.imageSpectrumIsOn    
        print 'IMAGE_PROJ_X_IS_ON',        self.imageProjXIsOn    
        print 'IMAGE_PROJ_Y_IS_ON',        self.imageProjYIsOn    
        print 'IMAGE_PROJ_R_IS_ON',        self.imageProjRIsOn    
        print 'IMAGE_PROJ_PHI_IS_ON',      self.imageProjPhiIsOn    

        print 'IMAGE_N_WINDOWS_MAX',       self.imageNWindowsMax 
        print 'IMAGE_N_WINDOWS',           self.imageNWindows 

        for win in range(self.imageNWindows) :

            print 'IMAGE_WINDOW_NUMBER',   win 
            print 'IMAGE_DATASET',         self.imageWindowParameters[win][0] 
            print 'IMAGE_IMAGE_AMIN',      self.imageWindowParameters[win][1] 
            print 'IMAGE_IMAGE_AMAX',      self.imageWindowParameters[win][2] 
            print 'IMAGE_SPECT_AMIN',      self.imageWindowParameters[win][3] 
            print 'IMAGE_SPECT_AMAX',      self.imageWindowParameters[win][4]         
            print 'IMAGE_SPECT_NBINS',     self.imageWindowParameters[win][5] 
            print 'IMAGE_SPECT_BIN_WIDTH', self.imageWindowParameters[win][6] 
            print 'IMAGE_IM_LIMITS_IS_ON', self.imageWindowParameters[win][7] 
            print 'IMAGE_SP_LIMITS_IS_ON', self.imageWindowParameters[win][8] 
            print 'IMAGE_BIN_WIDTH_IS_ON', self.imageWindowParameters[win][9] 
            print 'IMAGE_OFFSET',          self.imageWindowParameters[win][10] 


        print 'READ_PARS_AT_START',        self.readParsFromFileAtStart

        print 'WAVEF_WAVEF_IS_ON',         self.waveformWaveformIsOn    
        print 'WAVEF_WFVSEV_IS_ON',        self.waveformWaveVsEvIsOn    

        print 'WAVEF_N_WINDOWS_MAX',       self.waveformNWindowsMax 
        print 'WAVEF_N_WINDOWS',           self.waveformNWindows 

        for win in range(self.waveformNWindows) :

            print 'WAVEF_WINDOW_NUMBER',   win 
            print 'WAVEF_DATASET',         self.waveformWindowParameters[win][0] 
            print 'WAVEF_RANGE_UNITS_BITS',self.waveformWindowParameters[win][1] 
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
            print 'SELEC_DATASET',         self.selectionWindowParameters[win][6] 

        print 'CORR_N_WINDOWS_MAX',       self.correlationNWindowsMax 
        print 'CORR_N_WINDOWS',           self.correlationNWindows 

        for win in range(self.correlationNWindows) :

            print 'CORR_WINDOW_NUMBER',    win 
            print 'CORR_YDATASET',         self.correlationWindowParameters[win][0] 
            print 'CORR_XDATASET',         self.correlationWindowParameters[win][1] 
            print 'CORR_XPARRADIO',        self.correlationWindowParameters[win][2] 
            print 'CORR_YMIN',             self.correlationWindowParameters[win][3] 
            print 'CORR_YMAX',             self.correlationWindowParameters[win][4] 
            print 'CORR_XMIN',             self.correlationWindowParameters[win][5] 
            print 'CORR_XMAX',             self.correlationWindowParameters[win][6] 
            print 'CORR_YPARNAME',         self.correlationWindowParameters[win][7] 
            print 'CORR_XPARNAME',         self.correlationWindowParameters[win][8] 
            print 'CORR_YLIMS_IS_ON',      self.correlationWindowParameters[win][9] 
            print 'CORR_XLIMS_IS_ON',      self.correlationWindowParameters[win][10] 
            print 'CORR_LOGZ_IS_ON',       self.correlationWindowParameters[win][11] 
            print 'CORR_YNBINS',           self.correlationWindowParameters[win][12]
            print 'CORR_XNBINS',           self.correlationWindowParameters[win][13]
            print 'CORR_YPARINDEX',        self.correlationWindowParameters[win][14] 
            print 'CORR_XPARINDEX',        self.correlationWindowParameters[win][15] 
            
        print 'CALIBC_N_WINDOWS_MAX',       self.calibcycleNWindowsMax 
        print 'CALIBC_N_WINDOWS',           self.calibcycleNWindows 

        for win in range(self.calibcycleNWindows) :

            print 'CALIBC_WINDOW_NUMBER',    win 
            print 'CALIBC_YDATASET',         self.calibcycleWindowParameters[win][0] 
            print 'CALIBC_XDATASET',         self.calibcycleWindowParameters[win][1] 
            print 'CALIBC_XPARRADIO',        self.calibcycleWindowParameters[win][2] 
            print 'CALIBC_YMIN',             self.calibcycleWindowParameters[win][3] 
            print 'CALIBC_YMAX',             self.calibcycleWindowParameters[win][4] 
            print 'CALIBC_XMIN',             self.calibcycleWindowParameters[win][5] 
            print 'CALIBC_XMAX',             self.calibcycleWindowParameters[win][6] 
            print 'CALIBC_YPARNAME',         self.calibcycleWindowParameters[win][7] 
            print 'CALIBC_XPARNAME',         self.calibcycleWindowParameters[win][8] 
            print 'CALIBC_YLIMS_IS_ON',      self.calibcycleWindowParameters[win][9] 
            print 'CALIBC_XLIMS_IS_ON',      self.calibcycleWindowParameters[win][10] 
            print 'CALIBC_LOGZ_IS_ON',       self.calibcycleWindowParameters[win][11] 
            print 'CALIBC_YNBINS',           self.calibcycleWindowParameters[win][12]
            print 'CALIBC_XNBINS',           self.calibcycleWindowParameters[win][13]
            print 'CALIBC_YPARINDEX',        self.calibcycleWindowParameters[win][14] 
            print 'CALIBC_XPARINDEX',        self.calibcycleWindowParameters[win][15] 
            
        print 'NUM_EVENTS_FOR_AVERAGE',    self.numEventsAverage
        print 'SELECTION_IS_ON',           self.selectionIsOn

       #print 'PER_EVENT_DIST_IS_ON',      self.perEventDistIsOn
        print 'CORRELATIONS_IS_ON',        self.correlationsIsOn
        print 'CALIBCYCLE_IS_ON',          self.calibcycleIsOn

        print 'PROJ_CENTER_X',             self.projCenterX         
        print 'PROJ_CENTER_Y',             self.projCenterY         
                                          
        print 'PROJ_R_BIN_WIDTH_IS_ON',    self.projR_BinWidthIsOn  
        print 'PROJ_R_SLI_WIDTH_IS_ON',    self.projR_SliWidthIsOn                                            
        print 'PROJ_R_N_BINS',             self.projR_NBins         
        print 'PROJ_R_BIN_WIDTH',          self.projR_BinWidth      
        print 'PROJ_R_NSLICES',            self.projR_NSlices       
        print 'PROJ_R_SLICE_WIDTH',        self.projR_SliWidth                                                
        print 'PROJ_R_RMIN',               self.projR_Rmin          
        print 'PROJ_R_RMAX',               self.projR_Rmax          
        print 'PROJ_R_PHIMIN',             self.projR_Phimin        
        print 'PROJ_R_PHIMAX',             self.projR_Phimax        
                                          
        print 'PROJ_PHI_BIN_WIDTH_IS_ON',  self.projPhi_BinWidthIsOn
        print 'PROJ_PHI_SLI_WIDTH_IS_ON',  self.projPhi_SliWidthIsOn             
        print 'PROJ_PHI_N_BINS',           self.projPhi_NBins       
        print 'PROJ_PHI_BIN_WIDTH',        self.projPhi_BinWidth    
        print 'PROJ_PHI_NSLICES',          self.projPhi_NSlices     
        print 'PROJ_PHI_SLICE_WIDTH',      self.projPhi_SliWidth                                                                     
        print 'PROJ_PHI_RMIN',             self.projPhi_Rmin        
        print 'PROJ_PHI_RMAX',             self.projPhi_Rmax        
        print 'PROJ_PHI_PHIMIN',           self.projPhi_Phimin      
        print 'PROJ_PHI_PHIMAX',           self.projPhi_Phimax      

        print 'PROJ_X_BIN_WIDTH_IS_ON'   , self.projX_BinWidthIsOn  
        print 'PROJ_X_SLI_WIDTH_IS_ON'   , self.projX_SliWidthIsOn                                          
        print 'PROJ_X_N_BINS'            , self.projX_NBins         
        print 'PROJ_X_BIN_WIDTH'         , self.projX_BinWidth      
        print 'PROJ_X_NSLICES'           , self.projX_NSlices       
        print 'PROJ_X_SLICE_WIDTH'       , self.projX_SliWidth                                              
        print 'PROJ_X_XMIN'              , self.projX_Xmin          
        print 'PROJ_X_XMAX'              , self.projX_Xmax          
        print 'PROJ_X_YMIN'              , self.projX_Ymin          
        print 'PROJ_X_YMAX'              , self.projX_Ymax          
                                                                    
        print 'PROJ_Y_BIN_WIDTH_IS_ON'   , self.projY_BinWidthIsOn  
        print 'PROJ_Y_SLI_WIDTH_IS_ON'   , self.projY_SliWidthIsOn                                                                  
        print 'PROJ_Y_N_BINS'            , self.projY_NBins         
        print 'PROJ_Y_BIN_WIDTH'         , self.projY_BinWidth      
        print 'PROJ_Y_NSLICES'           , self.projY_NSlices       
        print 'PROJ_Y_SLICE_WIDTH'       , self.projY_SliWidth                                                                     
        print 'PROJ_Y_XMIN'              , self.projY_Xmin          
        print 'PROJ_Y_XMAX'              , self.projY_Xmax          
        print 'PROJ_Y_YMIN'              , self.projY_Ymin          
        print 'PROJ_Y_YMAX'              , self.projY_Ymax          

        print 'BKGD_SUBTRACTION_IS_ON'   , self.bkgdSubtractionIsOn 
        print 'BKGD_DIR_NAME'            , self.bkgdDirName         
        print 'BKGD_FILE_NAME'           , self.bkgdFileName        

        print 'GAIN_CORRECTION_IS_ON'    , self.gainCorrectionIsOn  
        print 'GAIN_DIR_NAME'            , self.gainDirName         
        print 'GAIN_FILE_NAME'           , self.gainFileName        

        print 'AVERAGE_DIR_NAME'         , self.aveDirName          
        print 'AVERAGE_FILE_NAME'        , self.aveFileName         

        print 70*'='


    def readParameters(self, fname=None) :
        self.__setConfigParsFileName(fname)        
        print 'Read parameters from file:', self._fname
        dicBool = {'false':False, 'true':True}
        win = 0
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
                elif key == 'PLAYER_GUI_IS_OPEN'       : self.playerGUIIsOpen         = dicBool[val.lower()]
                elif key == 'STEP_01_IS_DONE'          : self.step01IsDone            = dicBool[val.lower()]
                elif key == 'STEP_02_IS_DONE'          : self.step02IsDone            = dicBool[val.lower()]
                elif key == 'STEP_03_IS_DONE'          : self.step03IsDone            = dicBool[val.lower()]
                elif key == 'STEP_04_IS_DONE'          : self.step04IsDone            = dicBool[val.lower()]

                elif key == 'CSPAD_IMAGE_IS_ON'        : self.cspadImageIsOn          = dicBool[val.lower()]
                elif key == 'CSPAD_IMAGE_OF_PAIR_IS_ON': self.cspadImageOfPairIsOn    = dicBool[val.lower()]
                elif key == 'CSPAD_IMAGE_QUAD_IS_ON'   : self.cspadImageQuadIsOn      = dicBool[val.lower()]
                elif key == 'CSPAD_IMAGE_DET_IS_ON'    : self.cspadImageDetIsOn       = dicBool[val.lower()]
                elif key == 'CSPAD_SPECT_IS_ON'        : self.cspadSpectrumIsOn       = dicBool[val.lower()]
                elif key == 'CSPAD_SPECT_DET_IS_ON'    : self.cspadSpectrumDetIsOn    = dicBool[val.lower()]
                elif key == 'CSPAD_SPE08_IS_ON'        : self.cspadSpectrum08IsOn     = dicBool[val.lower()]
                elif key == 'CSPAD_PROJ_X_IS_ON'       : self.cspadProjXIsOn          = dicBool[val.lower()]
                elif key == 'CSPAD_PROJ_Y_IS_ON'       : self.cspadProjYIsOn          = dicBool[val.lower()]
                elif key == 'CSPAD_PROJ_R_IS_ON'       : self.cspadProjRIsOn          = dicBool[val.lower()]
                elif key == 'CSPAD_PROJ_PHI_IS_ON'     : self.cspadProjPhiIsOn        = dicBool[val.lower()]
                elif key == 'CSPAD_APPLY_TILT_ANGLE'   : self.cspadApplyTiltAngle     = dicBool[val.lower()]

                elif key == 'IMAGE_IMAGE_IS_ON'        : self.imageImageIsOn          = dicBool[val.lower()]
                elif key == 'IMAGE_IMAGE_SPEC_IS_ON'   : self.imageImageSpecIsOn      = dicBool[val.lower()]
                elif key == 'IMAGE_SPECT_IS_ON'        : self.imageSpectrumIsOn       = dicBool[val.lower()]
                elif key == 'IMAGE_PROJ_X_IS_ON'       : self.imageProjXIsOn          = dicBool[val.lower()]
                elif key == 'IMAGE_PROJ_Y_IS_ON'       : self.imageProjYIsOn          = dicBool[val.lower()]
                elif key == 'IMAGE_PROJ_R_IS_ON'       : self.imageProjRIsOn          = dicBool[val.lower()]
                elif key == 'IMAGE_PROJ_PHI_IS_ON'     : self.imageProjPhiIsOn        = dicBool[val.lower()]

                elif key == 'WAVEF_WAVEF_IS_ON'        : self.waveformWaveformIsOn    = dicBool[val.lower()]
                elif key == 'WAVEF_WFVSEV_IS_ON'       : self.waveformWaveVsEvIsOn    = dicBool[val.lower()]
                elif key == 'READ_PARS_AT_START'       : self.readParsFromFileAtStart = dicBool[val.lower()]

                #elif key == 'CSPAD_QUAD_NUMBER'        : self.cspadQuad               = int(val)
                #elif key == 'CSPAD_PAIR_NUMBER'        : self.cspadPair               = int(val)

                #elif key == 'CSPAD_RANGE_AMIN'         : self.cspadAmplitudeRaMin     = int(val)
                #elif key == 'CSPAD_RANGE_AMAX'         : self.cspadAmplitudeRange     = int(val)
                #elif key == 'CSPAD_IMAGE_AMIN'         : self.cspadImageAmin          = int(val)
                #elif key == 'CSPAD_IMAGE_AMAX'         : self.cspadImageAmax          = int(val)
                #elif key == 'CSPAD_SPECT_AMIN'         : self.cspadSpectrumAmin       = int(val)
                #elif key == 'CSPAD_SPECT_AMAX'         : self.cspadSpectrumAmax       = int(val)
                #elif key == 'CSPAD_SPECT_NBINS'        : self.cspadSpectrumNbins      = int(val)
                #elif key == 'CSPAD_SPECT_BIN_WIDTH'    : self.cspadSpectrumBinWidth   = int(val)
                #elif key == 'CSPAD_BIN_WIDTH_IS_ON'    : self.cspadBinWidthIsOn       = dicBool[val.lower()]

                elif key == 'CSPAD_N_WINDOWS_MAX'      : self.cspadNWindowsMax        = int(val)
                elif key == 'CSPAD_N_WINDOWS'          : self.cspadNWindows           = int(val)

                elif key == 'CSPAD_WINDOW_NUMBER'      : win                          = int(val)
                elif key == 'CSPAD_DATASET'            : self.cspadWindowParameters[win][0] = val
                elif key == 'CSPAD_IMAGE_AMIN'         : self.cspadWindowParameters[win][1] = int(val)
                elif key == 'CSPAD_IMAGE_AMAX'         : self.cspadWindowParameters[win][2] = int(val)
                elif key == 'CSPAD_SPECT_AMIN'         : self.cspadWindowParameters[win][3] = int(val)
                elif key == 'CSPAD_SPECT_AMAX'         : self.cspadWindowParameters[win][4] = int(val)
                elif key == 'CSPAD_SPECT_NBINS'        : self.cspadWindowParameters[win][5] = int(val)
                elif key == 'CSPAD_SPECT_BIN_WIDTH'    : self.cspadWindowParameters[win][6] = int(val)
                elif key == 'CSPAD_IM_LIMITS_IS_ON'    : self.cspadWindowParameters[win][7] = dicBool[val.lower()]
                elif key == 'CSPAD_SP_LIMITS_IS_ON'    : self.cspadWindowParameters[win][8] = dicBool[val.lower()]
                elif key == 'CSPAD_BIN_WIDTH_IS_ON'    : self.cspadWindowParameters[win][9] = dicBool[val.lower()]
                elif key == 'CSPAD_QUAD_NUMBER'        : self.cspadWindowParameters[win][10]= int(val)
                elif key == 'CSPAD_PAIR_NUMBER'        : self.cspadWindowParameters[win][11]= int(val)

                elif key == 'IMAGE_N_WINDOWS_MAX'      : self.imageNWindowsMax        = int(val)
                elif key == 'IMAGE_N_WINDOWS'          : self.imageNWindows           = int(val)

                elif key == 'IMAGE_WINDOW_NUMBER'      : win                          = int(val)
                elif key == 'IMAGE_DATASET'            : self.imageWindowParameters[win][0] = val
                elif key == 'IMAGE_IMAGE_AMIN'         : self.imageWindowParameters[win][1] = int(val)
                elif key == 'IMAGE_IMAGE_AMAX'         : self.imageWindowParameters[win][2] = int(val)
                elif key == 'IMAGE_SPECT_AMIN'         : self.imageWindowParameters[win][3] = int(val)
                elif key == 'IMAGE_SPECT_AMAX'         : self.imageWindowParameters[win][4] = int(val)
                elif key == 'IMAGE_SPECT_NBINS'        : self.imageWindowParameters[win][5] = int(val)
                elif key == 'IMAGE_SPECT_BIN_WIDTH'    : self.imageWindowParameters[win][6] = int(val)
                elif key == 'IMAGE_IM_LIMITS_IS_ON'    : self.imageWindowParameters[win][7] = dicBool[val.lower()]
                elif key == 'IMAGE_SP_LIMITS_IS_ON'    : self.imageWindowParameters[win][8] = dicBool[val.lower()]
                elif key == 'IMAGE_BIN_WIDTH_IS_ON'    : self.imageWindowParameters[win][9] = dicBool[val.lower()]
                elif key == 'IMAGE_OFFSET'             : self.imageWindowParameters[win][10]= int(val)


               #elif key == 'PER_EVENT_DIST_IS_ON'     : self.perEventDistIsOn        = dicBool[val.lower()]
                elif key == 'CORRELATIONS_IS_ON'       : self.correlationsIsOn        = dicBool[val.lower()]
                elif key == 'CALIBCYCLE_IS_ON'         : self.calibcycleIsOn          = dicBool[val.lower()]

                elif key == 'WAVEF_N_WINDOWS_MAX'      : self.waveformNWindowsMax     = int(val)
                elif key == 'WAVEF_N_WINDOWS'          : self.waveformNWindows        = int(val)

                elif key == 'WAVEF_WINDOW_NUMBER'      : win                          = int(val)
                elif key == 'WAVEF_DATASET'            : self.waveformWindowParameters[win][0] = val
                elif key == 'WAVEF_RANGE_UNITS_BITS'   : self.waveformWindowParameters[win][1] = int(val)
                elif key == 'WAVEF_AMIN'               : self.waveformWindowParameters[win][2] = float(val)
                elif key == 'WAVEF_AMAX'               : self.waveformWindowParameters[win][3] = float(val)
                elif key == 'WAVEF_TMIN'               : self.waveformWindowParameters[win][4] = float(val)
                elif key == 'WAVEF_TMAX'               : self.waveformWindowParameters[win][5] = float(val)
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
                elif key == 'SELEC_DATASET'            : self.selectionWindowParameters[win][6] = val


                elif key == 'CORR_N_WINDOWS_MAX'       : self.correlationNWindowsMax   = int(val)
                elif key == 'CORR_N_WINDOWS'           : self.correlationNWindows      = int(val)

                elif key == 'CORR_WINDOW_NUMBER'       : win                           = int(val)
                elif key == 'CORR_YDATASET'            : self.correlationWindowParameters[win][0] = val
                elif key == 'CORR_XDATASET'            : self.correlationWindowParameters[win][1] = val
                elif key == 'CORR_XPARRADIO'           : self.correlationWindowParameters[win][2] = int(val)
                elif key == 'CORR_YMIN'                : self.correlationWindowParameters[win][3] = self.getValIntOrNone(val)
                elif key == 'CORR_YMAX'                : self.correlationWindowParameters[win][4] = self.getValIntOrNone(val)
                elif key == 'CORR_XMIN'                : self.correlationWindowParameters[win][5] = self.getValIntOrNone(val)      
                elif key == 'CORR_XMAX'                : self.correlationWindowParameters[win][6] = self.getValIntOrNone(val)      
                elif key == 'CORR_YPARNAME'            : self.correlationWindowParameters[win][7] = val  
                elif key == 'CORR_XPARNAME'            : self.correlationWindowParameters[win][8] = val  
                elif key == 'CORR_YLIMS_IS_ON'         : self.correlationWindowParameters[win][9] = dicBool[val.lower()] 
                elif key == 'CORR_XLIMS_IS_ON'         : self.correlationWindowParameters[win][10]= dicBool[val.lower()] 
                elif key == 'CORR_LOGZ_IS_ON'          : self.correlationWindowParameters[win][11]= dicBool[val.lower()] 
                elif key == 'CORR_YNBINS'              : self.correlationWindowParameters[win][12]= self.getValIntOrNone(val)
                elif key == 'CORR_XNBINS'              : self.correlationWindowParameters[win][13]= self.getValIntOrNone(val)
                elif key == 'CORR_YPARINDEX'           : self.correlationWindowParameters[win][14]= val
                elif key == 'CORR_XPARINDEX'           : self.correlationWindowParameters[win][15]= val


                elif key == 'CALIBC_N_WINDOWS_MAX'     : self.calibcycleNWindowsMax   = int(val)
                elif key == 'CALIBC_N_WINDOWS'         : self.calibcycleNWindows      = int(val)

                elif key == 'CALIBC_WINDOW_NUMBER'     : win                = int(val)
                elif key == 'CALIBC_YDATASET'          : self.calibcycleWindowParameters[win][0] = val
                elif key == 'CALIBC_XDATASET'          : self.calibcycleWindowParameters[win][1] = val
                elif key == 'CALIBC_XPARRADIO'         : self.calibcycleWindowParameters[win][2] = int(val)
                elif key == 'CALIBC_YMIN'              : self.calibcycleWindowParameters[win][3] = self.getValIntOrNone(val)
                elif key == 'CALIBC_YMAX'              : self.calibcycleWindowParameters[win][4] = self.getValIntOrNone(val)
                elif key == 'CALIBC_XMIN'              : self.calibcycleWindowParameters[win][5] = self.getValIntOrNone(val)      
                elif key == 'CALIBC_XMAX'              : self.calibcycleWindowParameters[win][6] = self.getValIntOrNone(val)      
                elif key == 'CALIBC_YPARNAME'          : self.calibcycleWindowParameters[win][7] = val  
                elif key == 'CALIBC_XPARNAME'          : self.calibcycleWindowParameters[win][8] = val  
                elif key == 'CALIBC_YLIMS_IS_ON'       : self.calibcycleWindowParameters[win][9] = dicBool[val.lower()] 
                elif key == 'CALIBC_XLIMS_IS_ON'       : self.calibcycleWindowParameters[win][10]= dicBool[val.lower()] 
                elif key == 'CALIBC_LOGZ_IS_ON'        : self.calibcycleWindowParameters[win][11]= dicBool[val.lower()] 
                elif key == 'CALIBC_YNBINS'            : self.calibcycleWindowParameters[win][12]= self.getValIntOrNone(val)
                elif key == 'CALIBC_XNBINS'            : self.calibcycleWindowParameters[win][13]= self.getValIntOrNone(val)
                elif key == 'CALIBC_YPARINDEX'         : self.calibcycleWindowParameters[win][14]= val
                elif key == 'CALIBC_XPARINDEX'         : self.calibcycleWindowParameters[win][15]= val
                
                elif key == 'PROJ_CENTER_X'            : self.projCenterX           = float(val)
                elif key == 'PROJ_CENTER_Y'            : self.projCenterY           = float(val)
                                                                                   
                elif key == 'PROJ_R_BIN_WIDTH_IS_ON'   : self.projR_BinWidthIsOn    = dicBool[val.lower()]
                elif key == 'PROJ_R_SLI_WIDTH_IS_ON'   : self.projR_SliWidthIsOn    = dicBool[val.lower()]
                elif key == 'PROJ_R_N_BINS'            : self.projR_NBins           = int(val)
                elif key == 'PROJ_R_BIN_WIDTH'         : self.projR_BinWidth        = int(val)
                elif key == 'PROJ_R_NSLICES'           : self.projR_NSlices         = int(val)
                elif key == 'PROJ_R_SLICE_WIDTH'       : self.projR_SliWidth        = int(val)
                elif key == 'PROJ_R_RMIN'              : self.projR_Rmin            = int(val)
                elif key == 'PROJ_R_RMAX'              : self.projR_Rmax            = int(val)
                elif key == 'PROJ_R_PHIMIN'            : self.projR_Phimin          = int(val)
                elif key == 'PROJ_R_PHIMAX'            : self.projR_Phimax          = int(val)
                                                                                   
                elif key == 'PROJ_PHI_BIN_WIDTH_IS_ON' : self.projPhi_BinWidthIsOn  = dicBool[val.lower()]
                elif key == 'PROJ_PHI_SLI_WIDTH_IS_ON' : self.projPhi_SliWidthIsOn  = dicBool[val.lower()]
                elif key == 'PROJ_PHI_N_BINS'          : self.projPhi_NBins         = int(val)
                elif key == 'PROJ_PHI_BIN_WIDTH'       : self.projPhi_BinWidth      = int(val)
                elif key == 'PROJ_PHI_NSLICES'         : self.projPhi_NSlices       = int(val)
                elif key == 'PROJ_PHI_SLICE_WIDTH'     : self.projPhi_SliWidth      = int(val)
                elif key == 'PROJ_PHI_RMIN'            : self.projPhi_Rmin          = int(val)
                elif key == 'PROJ_PHI_RMAX'            : self.projPhi_Rmax          = int(val)
                elif key == 'PROJ_PHI_PHIMIN'          : self.projPhi_Phimin        = int(val)
                elif key == 'PROJ_PHI_PHIMAX'          : self.projPhi_Phimax        = int(val)

                elif key == 'PROJ_X_BIN_WIDTH_IS_ON'   : self.projX_BinWidthIsOn    = dicBool[val.lower()]
                elif key == 'PROJ_X_SLI_WIDTH_IS_ON'   : self.projX_SliWidthIsOn    = dicBool[val.lower()]
                elif key == 'PROJ_X_N_BINS'            : self.projX_NBins           = int(val)
                elif key == 'PROJ_X_BIN_WIDTH'         : self.projX_BinWidth        = int(val)
                elif key == 'PROJ_X_NSLICES'           : self.projX_NSlices         = int(val)
                elif key == 'PROJ_X_SLICE_WIDTH'       : self.projX_SliWidth        = int(val)
                elif key == 'PROJ_X_XMIN'              : self.projX_Xmin            = int(val)
                elif key == 'PROJ_X_XMAX'              : self.projX_Xmax            = int(val)
                elif key == 'PROJ_X_YMIN'              : self.projX_Ymin            = int(val)
                elif key == 'PROJ_X_YMAX'              : self.projX_Ymax            = int(val)
                                                                                  
                elif key == 'PROJ_Y_BIN_WIDTH_IS_ON'   : self.projY_BinWidthIsOn    = dicBool[val.lower()]
                elif key == 'PROJ_Y_SLI_WIDTH_IS_ON'   : self.projY_SliWidthIsOn    = dicBool[val.lower()]
                elif key == 'PROJ_Y_N_BINS'            : self.projY_NBins           = int(val)
                elif key == 'PROJ_Y_BIN_WIDTH'         : self.projY_BinWidth        = int(val)
                elif key == 'PROJ_Y_NSLICES'           : self.projY_NSlices         = int(val)
                elif key == 'PROJ_Y_SLICE_WIDTH'       : self.projY_SliWidth        = int(val)
                elif key == 'PROJ_Y_XMIN'              : self.projY_Xmin            = int(val)
                elif key == 'PROJ_Y_XMAX'              : self.projY_Xmax            = int(val)
                elif key == 'PROJ_Y_YMIN'              : self.projY_Ymin            = int(val)
                elif key == 'PROJ_Y_YMAX'              : self.projY_Ymax            = int(val)

               #elif key == 'BKGD_SUBTRACTION_IS_ON'   : self.bkgdSubtractionIsOn   = dicBool[val.lower()]
                elif key == 'BKGD_DIR_NAME'            : self.bkgdDirName           = val
                elif key == 'BKGD_FILE_NAME'           : self.bkgdFileName          = val
                                                                                  
               #elif key == 'GAIN_CORRECTION_IS_ON'    : self.gainCorrectionIsOn    = dicBool[val.lower()]
                elif key == 'GAIN_DIR_NAME'            : self.gainDirName           = val
                elif key == 'GAIN_FILE_NAME'           : self.gainFileName          = val
                                                                                  
                elif key == 'AVERAGE_DIR_NAME'         : self.aveDirName            = val
                elif key == 'AVERAGE_FILE_NAME'        : self.aveFileName           = val

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
        f.write('PLAYER_GUI_IS_OPEN'        + space + str(self.playerGUIIsOpen)         + '\n')
        f.write('STEP_01_IS_DONE'           + space + str(self.step01IsDone)            + '\n')
        f.write('STEP_02_IS_DONE'           + space + str(self.step02IsDone)            + '\n')
        f.write('STEP_03_IS_DONE'           + space + str(self.step03IsDone)            + '\n')
        f.write('STEP_04_IS_DONE'           + space + str(self.step04IsDone)            + '\n')

        f.write('CSPAD_IMAGE_IS_ON'         + space + str(self.cspadImageIsOn)          + '\n')
        f.write('CSPAD_IMAGE_OF_PAIR_IS_ON' + space + str(self.cspadImageOfPairIsOn)    + '\n')
        f.write('CSPAD_IMAGE_QUAD_IS_ON'    + space + str(self.cspadImageQuadIsOn)      + '\n')
        f.write('CSPAD_IMAGE_DET_IS_ON'     + space + str(self.cspadImageDetIsOn)       + '\n')
        f.write('CSPAD_SPECT_IS_ON'         + space + str(self.cspadSpectrumIsOn)       + '\n')
        f.write('CSPAD_SPECT_DET_IS_ON'     + space + str(self.cspadSpectrumDetIsOn)    + '\n')
        f.write('CSPAD_SPE08_IS_ON'         + space + str(self.cspadSpectrum08IsOn)     + '\n')
        f.write('CSPAD_PROJ_X_IS_ON'        + space + str(self.cspadProjXIsOn)          + '\n')
        f.write('CSPAD_PROJ_Y_IS_ON'        + space + str(self.cspadProjYIsOn)          + '\n')
        f.write('CSPAD_PROJ_R_IS_ON'        + space + str(self.cspadProjRIsOn)          + '\n')
        f.write('CSPAD_PROJ_PHI_IS_ON'      + space + str(self.cspadProjPhiIsOn)        + '\n')
        f.write('CSPAD_APPLY_TILT_ANGLE'    + space + str(self.cspadApplyTiltAngle)     + '\n')

        f.write('IMAGE_IMAGE_IS_ON'         + space + str(self.imageImageIsOn)          + '\n')
        f.write('IMAGE_IMAGE_SPEC_IS_ON'    + space + str(self.imageImageSpecIsOn)      + '\n')
        f.write('IMAGE_SPECT_IS_ON'         + space + str(self.imageSpectrumIsOn)       + '\n')
        f.write('IMAGE_PROJ_X_IS_ON'        + space + str(self.imageProjXIsOn)          + '\n')
        f.write('IMAGE_PROJ_Y_IS_ON'        + space + str(self.imageProjYIsOn)          + '\n')
        f.write('IMAGE_PROJ_R_IS_ON'        + space + str(self.imageProjRIsOn)          + '\n')
        f.write('IMAGE_PROJ_PHI_IS_ON'      + space + str(self.imageProjPhiIsOn)        + '\n')
        f.write('WAVEF_WAVEF_IS_ON'         + space + str(self.waveformWaveformIsOn)    + '\n')
        f.write('WAVEF_WFVSEV_IS_ON'        + space + str(self.waveformWaveVsEvIsOn)    + '\n')
        f.write('READ_PARS_AT_START'        + space + str(self.readParsFromFileAtStart) + '\n')

        #f.write('CSPAD_QUAD_NUMBER'         + space + str(self.cspadQuad)               + '\n')
        #f.write('CSPAD_PAIR_NUMBER'         + space + str(self.cspadPair)               + '\n')
        #f.write('CSPAD_RANGE_AMIN'          + space + str(self.cspadAmplitudeRaMin)     + '\n')
        #f.write('CSPAD_RANGE_AMAX'          + space + str(self.cspadAmplitudeRange)     + '\n')
        #f.write('CSPAD_IMAGE_AMIN'          + space + str(self.cspadImageAmin)          + '\n')
        #f.write('CSPAD_IMAGE_AMAX'          + space + str(self.cspadImageAmax)          + '\n')
        #f.write('CSPAD_SPECT_AMIN'          + space + str(self.cspadSpectrumAmin)       + '\n')
        #f.write('CSPAD_SPECT_AMAX'          + space + str(self.cspadSpectrumAmax)       + '\n')
        #f.write('CSPAD_SPECT_NBINS'         + space + str(self.cspadSpectrumNbins)      + '\n')
        #f.write('CSPAD_SPECT_BIN_WIDTH'     + space + str(self.cspadSpectrumBinWidth)   + '\n')
        #f.write('CSPAD_BIN_WIDTH_IS_ON'     + space + str(self.cspadBinWidthIsOn)       + '\n')


        f.write('\n')
        f.write('CSPAD_N_WINDOWS_MAX'       + space + str(self.cspadNWindowsMax)     + '\n')
        f.write('CSPAD_N_WINDOWS'           + space + str(self.cspadNWindows)        + '\n')

        for win in range(self.cspadNWindows) :
            f.write('\n')
            f.write('CSPAD_WINDOW_NUMBER'  + space + str(win)                                    + '\n')
            f.write('CSPAD_DATASET'        + space + str(self.cspadWindowParameters[win][0] )    + '\n')
            f.write('CSPAD_IMAGE_AMIN'     + space + str(self.cspadWindowParameters[win][1] )    + '\n')
            f.write('CSPAD_IMAGE_AMAX'     + space + str(self.cspadWindowParameters[win][2] )    + '\n')
            f.write('CSPAD_SPECT_AMIN'     + space + str(self.cspadWindowParameters[win][3] )    + '\n')
            f.write('CSPAD_SPECT_AMAX'     + space + str(self.cspadWindowParameters[win][4] )    + '\n')
            f.write('CSPAD_SPECT_NBINS'    + space + str(self.cspadWindowParameters[win][5] )    + '\n')
            f.write('CSPAD_SPECT_BIN_WIDTH'+ space + str(self.cspadWindowParameters[win][6] )    + '\n')
            f.write('CSPAD_IM_LIMITS_IS_ON'+ space + str(self.cspadWindowParameters[win][7] )    + '\n')
            f.write('CSPAD_SP_LIMITS_IS_ON'+ space + str(self.cspadWindowParameters[win][8] )    + '\n')
            f.write('CSPAD_BIN_WIDTH_IS_ON'+ space + str(self.cspadWindowParameters[win][9] )    + '\n')
            f.write('CSPAD_QUAD_NUMBER'    + space + str(self.cspadWindowParameters[win][10])    + '\n')
            f.write('CSPAD_PAIR_NUMBER'    + space + str(self.cspadWindowParameters[win][11])    + '\n')


        f.write('\n')
        f.write('IMAGE_N_WINDOWS_MAX'       + space + str(self.imageNWindowsMax)     + '\n')
        f.write('IMAGE_N_WINDOWS'           + space + str(self.imageNWindows)        + '\n')

        for win in range(self.imageNWindows) :
            f.write('\n')
            f.write('IMAGE_WINDOW_NUMBER'  + space + str(win)                                    + '\n')
            f.write('IMAGE_DATASET'        + space + str(self.imageWindowParameters[win][0] )    + '\n')
            f.write('IMAGE_IMAGE_AMIN'     + space + str(self.imageWindowParameters[win][1] )    + '\n')
            f.write('IMAGE_IMAGE_AMAX'     + space + str(self.imageWindowParameters[win][2] )    + '\n')
            f.write('IMAGE_SPECT_AMIN'     + space + str(self.imageWindowParameters[win][3] )    + '\n')
            f.write('IMAGE_SPECT_AMAX'     + space + str(self.imageWindowParameters[win][4] )    + '\n')
            f.write('IMAGE_SPECT_NBINS'    + space + str(self.imageWindowParameters[win][5] )    + '\n')
            f.write('IMAGE_SPECT_BIN_WIDTH'+ space + str(self.imageWindowParameters[win][6] )    + '\n')
            f.write('IMAGE_IM_LIMITS_IS_ON'+ space + str(self.imageWindowParameters[win][7] )    + '\n')
            f.write('IMAGE_SP_LIMITS_IS_ON'+ space + str(self.imageWindowParameters[win][8] )    + '\n')
            f.write('IMAGE_BIN_WIDTH_IS_ON'+ space + str(self.imageWindowParameters[win][9] )    + '\n')
            f.write('IMAGE_OFFSET'         + space + str(self.imageWindowParameters[win][10])    + '\n')


       #f.write('PER_EVENT_DIST_IS_ON'      + space + str(self.perEventDistIsOn)        + '\n')
        f.write('CORRELATIONS_IS_ON'        + space + str(self.correlationsIsOn)        + '\n')
        f.write('CALIBCYCLE_IS_ON'          + space + str(self.calibcycleIsOn)          + '\n')

        f.write('NUM_EVENTS_FOR_AVERAGE'+ space + str( self.numEventsAverage )          + '\n')
        f.write('SELECTION_IS_ON'       + space + str( self.selectionIsOn )             + '\n')

        f.write('\n')
        f.write('WAVEF_N_WINDOWS_MAX'       + space + str(self.waveformNWindowsMax)     + '\n')
        f.write('WAVEF_N_WINDOWS'           + space + str(self.waveformNWindows)        + '\n')

        for win in range(self.waveformNWindows) :
            f.write('\n')
            f.write('WAVEF_WINDOW_NUMBER'   + space + str(win)                                       + '\n')
            f.write('WAVEF_DATASET'         + space + str(self.waveformWindowParameters[win][0] )    + '\n')
            f.write('WAVEF_RANGE_UNITS_BITS'+ space + str(self.waveformWindowParameters[win][1] )    + '\n')
            f.write('WAVEF_AMIN'            + space + str(self.waveformWindowParameters[win][2] )    + '\n')
            f.write('WAVEF_AMAX'            + space + str(self.waveformWindowParameters[win][3] )    + '\n')
            f.write('WAVEF_TMIN'            + space + str(self.waveformWindowParameters[win][4] )    + '\n')
            f.write('WAVEF_TMAX'            + space + str(self.waveformWindowParameters[win][5] )    + '\n')
            f.write('WAVEF_N_WF_IN_DATA_SET'+ space + str(self.waveformWindowParameters[win][6] )    + '\n')
            f.write('WAVEF_IND_WF_IN_BLACK' + space + str(self.waveformWindowParameters[win][7] )    + '\n')
            f.write('WAVEF_IND_WF_IN_RED'   + space + str(self.waveformWindowParameters[win][8] )    + '\n')
            f.write('WAVEF_IND_WF_IN_GREEN' + space + str(self.waveformWindowParameters[win][9] )    + '\n')
            f.write('WAVEF_IND_WF_IN_BLUE'  + space + str(self.waveformWindowParameters[win][10])    + '\n')

        f.write('SELEC_N_WINDOWS_MAX'       + space + str(self.selectionNWindowsMax)     + '\n')
        f.write('SELEC_N_WINDOWS'           + space + str(self.selectionNWindows)        + '\n')

        for win in range(self.selectionNWindows) :
            f.write('\n')
            f.write('SELEC_WINDOW_NUMBER'   + space + str(win)                                       + '\n')
            f.write('SELEC_THRESHOLD'       + space + str(self.selectionWindowParameters[win][0] )   + '\n')
            f.write('SELEC_IN_BIN'          + space + str(self.selectionWindowParameters[win][1] )   + '\n')
            f.write('SELEC_XMIN'            + space + str(self.selectionWindowParameters[win][2] )   + '\n')
            f.write('SELEC_XMAX'            + space + str(self.selectionWindowParameters[win][3] )   + '\n')
            f.write('SELEC_YMIN'            + space + str(self.selectionWindowParameters[win][4] )   + '\n')
            f.write('SELEC_YMAX'            + space + str(self.selectionWindowParameters[win][5] )   + '\n')
            f.write('SELEC_DATASET'         + space + str(self.selectionWindowParameters[win][6] )   + '\n')

        f.write('CORR_N_WINDOWS_MAX'        + space + str(self.correlationNWindowsMax)   + '\n')
        f.write('CORR_N_WINDOWS'            + space + str(self.correlationNWindows)      + '\n')

        for win in range(self.correlationNWindows) :
            f.write('\n')
            f.write('CORR_WINDOW_NUMBER'    + space + str(win)                                       + '\n')
            f.write('CORR_YDATASET'         + space + str(self.correlationWindowParameters[win][0] ) + '\n')
            f.write('CORR_XDATASET'         + space + str(self.correlationWindowParameters[win][1] ) + '\n')
            f.write('CORR_XPARRADIO'        + space + str(self.correlationWindowParameters[win][2] ) + '\n')
            f.write('CORR_YMIN'             + space + str(self.correlationWindowParameters[win][3] ) + '\n')
            f.write('CORR_YMAX'             + space + str(self.correlationWindowParameters[win][4] ) + '\n')
            f.write('CORR_XMIN'             + space + str(self.correlationWindowParameters[win][5] ) + '\n') 
            f.write('CORR_XMAX'             + space + str(self.correlationWindowParameters[win][6] ) + '\n') 
            f.write('CORR_YPARNAME'         + space + str(self.correlationWindowParameters[win][7] ) + '\n') 
            f.write('CORR_XPARNAME'         + space + str(self.correlationWindowParameters[win][8] ) + '\n') 
            f.write('CORR_YLIMS_IS_ON'      + space + str(self.correlationWindowParameters[win][9] ) + '\n') 
            f.write('CORR_XLIMS_IS_ON'      + space + str(self.correlationWindowParameters[win][10]) + '\n') 
            f.write('CORR_LOGZ_IS_ON'       + space + str(self.correlationWindowParameters[win][11]) + '\n') 
            f.write('CORR_YNBINS'           + space + str(self.correlationWindowParameters[win][12]) + '\n') 
            f.write('CORR_XNBINS'           + space + str(self.correlationWindowParameters[win][13]) + '\n') 
            f.write('CORR_YPARINDEX'        + space + str(self.correlationWindowParameters[win][14]) + '\n') 
            f.write('CORR_XPARINDEX'        + space + str(self.correlationWindowParameters[win][15]) + '\n') 


        f.write('CALIBC_N_WINDOWS_MAX'        + space + str(self.calibcycleNWindowsMax)   + '\n')
        f.write('CALIBC_N_WINDOWS'            + space + str(self.calibcycleNWindows)      + '\n')

        for win in range(self.calibcycleNWindows) :
            f.write('\n')
            f.write('CALIBC_WINDOW_NUMBER'    + space + str(win)                          + '\n')
            f.write('CALIBC_YDATASET'         + space + str(self.calibcycleWindowParameters[win][0] ) + '\n')
            f.write('CALIBC_XDATASET'         + space + str(self.calibcycleWindowParameters[win][1] ) + '\n')
            f.write('CALIBC_XPARRADIO'        + space + str(self.calibcycleWindowParameters[win][2] ) + '\n')
            f.write('CALIBC_YMIN'             + space + str(self.calibcycleWindowParameters[win][3] ) + '\n')
            f.write('CALIBC_YMAX'             + space + str(self.calibcycleWindowParameters[win][4] ) + '\n')
            f.write('CALIBC_XMIN'             + space + str(self.calibcycleWindowParameters[win][5] ) + '\n') 
            f.write('CALIBC_XMAX'             + space + str(self.calibcycleWindowParameters[win][6] ) + '\n') 
            f.write('CALIBC_YPARNAME'         + space + str(self.calibcycleWindowParameters[win][7] ) + '\n') 
            f.write('CALIBC_XPARNAME'         + space + str(self.calibcycleWindowParameters[win][8] ) + '\n') 
            f.write('CALIBC_YLIMS_IS_ON'      + space + str(self.calibcycleWindowParameters[win][9] ) + '\n') 
            f.write('CALIBC_XLIMS_IS_ON'      + space + str(self.calibcycleWindowParameters[win][10]) + '\n') 
            f.write('CALIBC_LOGZ_IS_ON'       + space + str(self.calibcycleWindowParameters[win][11]) + '\n') 
            f.write('CALIBC_YNBINS'           + space + str(self.calibcycleWindowParameters[win][12]) + '\n') 
            f.write('CALIBC_XNBINS'           + space + str(self.calibcycleWindowParameters[win][13]) + '\n') 
            f.write('CALIBC_YPARINDEX'        + space + str(self.calibcycleWindowParameters[win][14]) + '\n') 
            f.write('CALIBC_XPARINDEX'        + space + str(self.calibcycleWindowParameters[win][15]) + '\n') 

        f.write('PROJ_CENTER_X'                     + space + str(self.projCenterX         )       + '\n')
        f.write('PROJ_CENTER_Y'                     + space + str(self.projCenterY         )       + '\n')
                                                                                           
        f.write('PROJ_R_BIN_WIDTH_IS_ON'            + space + str(self.projR_BinWidthIsOn  )       + '\n')
        f.write('PROJ_R_SLI_WIDTH_IS_ON'            + space + str(self.projR_SliWidthIsOn  )       + '\n')
        f.write('PROJ_R_N_BINS'                     + space + str(self.projR_NBins         )       + '\n')
        f.write('PROJ_R_BIN_WIDTH'                  + space + str(self.projR_BinWidth      )       + '\n')
        f.write('PROJ_R_NSLICES'                    + space + str(self.projR_NSlices       )       + '\n')
        f.write('PROJ_R_SLICE_WIDTH'                + space + str(self.projR_SliWidth      )       + '\n')
        f.write('PROJ_R_RMIN'                       + space + str(self.projR_Rmin          )       + '\n')
        f.write('PROJ_R_RMAX'                       + space + str(self.projR_Rmax          )       + '\n')
        f.write('PROJ_R_PHIMIN'                     + space + str(self.projR_Phimin        )       + '\n')
        f.write('PROJ_R_PHIMAX'                     + space + str(self.projR_Phimax        )       + '\n')
                                                                                           
        f.write('PROJ_PHI_BIN_WIDTH_IS_ON'          + space + str(self.projPhi_BinWidthIsOn)       + '\n')
        f.write('PROJ_PHI_SLI_WIDTH_IS_ON'          + space + str(self.projPhi_SliWidthIsOn)       + '\n')
        f.write('PROJ_PHI_N_BINS'                   + space + str(self.projPhi_NBins       )       + '\n')
        f.write('PROJ_PHI_BIN_WIDTH'                + space + str(self.projPhi_BinWidth    )       + '\n')
        f.write('PROJ_PHI_NSLICES'                  + space + str(self.projPhi_NSlices     )       + '\n')
        f.write('PROJ_PHI_SLICE_WIDTH'              + space + str(self.projPhi_SliWidth    )       + '\n')
        f.write('PROJ_PHI_RMIN'                     + space + str(self.projPhi_Rmin        )       + '\n')
        f.write('PROJ_PHI_RMAX'                     + space + str(self.projPhi_Rmax        )       + '\n')
        f.write('PROJ_PHI_PHIMIN'                   + space + str(self.projPhi_Phimin      )       + '\n')
        f.write('PROJ_PHI_PHIMAX'                   + space + str(self.projPhi_Phimax      )       + '\n')

        f.write('PROJ_X_BIN_WIDTH_IS_ON'            + space + str(self.projX_BinWidthIsOn  )       + '\n')
        f.write('PROJ_X_SLI_WIDTH_IS_ON'            + space + str(self.projX_SliWidthIsOn  )       + '\n')
        f.write('PROJ_X_N_BINS'                     + space + str(self.projX_NBins         )       + '\n')
        f.write('PROJ_X_BIN_WIDTH'                  + space + str(self.projX_BinWidth      )       + '\n')
        f.write('PROJ_X_NSLICES'                    + space + str(self.projX_NSlices       )       + '\n')
        f.write('PROJ_X_SLICE_WIDTH'                + space + str(self.projX_SliWidth      )       + '\n')
        f.write('PROJ_X_XMIN'                       + space + str(self.projX_Xmin          )       + '\n')
        f.write('PROJ_X_XMAX'                       + space + str(self.projX_Xmax          )       + '\n')
        f.write('PROJ_X_YMIN'                       + space + str(self.projX_Ymin          )       + '\n')
        f.write('PROJ_X_YMAX'                       + space + str(self.projX_Ymax          )       + '\n')
                                                                                           
        f.write('PROJ_Y_BIN_WIDTH_IS_ON'            + space + str(self.projY_BinWidthIsOn  )       + '\n')
        f.write('PROJ_Y_SLI_WIDTH_IS_ON'            + space + str(self.projY_SliWidthIsOn  )       + '\n')
        f.write('PROJ_Y_N_BINS'                     + space + str(self.projY_NBins         )       + '\n')
        f.write('PROJ_Y_BIN_WIDTH'                  + space + str(self.projY_BinWidth      )       + '\n')
        f.write('PROJ_Y_NSLICES'                    + space + str(self.projY_NSlices       )       + '\n')
        f.write('PROJ_Y_SLICE_WIDTH'                + space + str(self.projY_SliWidth      )       + '\n')
        f.write('PROJ_Y_XMIN'                       + space + str(self.projY_Xmin          )       + '\n')
        f.write('PROJ_Y_XMAX'                       + space + str(self.projY_Xmax          )       + '\n')
        f.write('PROJ_Y_YMIN'                       + space + str(self.projY_Ymin          )       + '\n')
        f.write('PROJ_Y_YMAX'                       + space + str(self.projY_Ymax          )       + '\n')


        f.write('BKGD_SUBTRACTION_IS_ON'            + space + str(self.bkgdSubtractionIsOn )       + '\n')
        f.write('BKGD_DIR_NAME'                     + space + str(self.bkgdDirName         )       + '\n')
        f.write('BKGD_FILE_NAME'                    + space + str(self.bkgdFileName        )       + '\n')
                                                                                           
        f.write('GAIN_CORRECTION_IS_ON'             + space + str(self.gainCorrectionIsOn  )       + '\n')
        f.write('GAIN_DIR_NAME'                     + space + str(self.gainDirName         )       + '\n')
        f.write('GAIN_FILE_NAME'                    + space + str(self.gainFileName        )       + '\n')
                                                                                           
        f.write('AVERAGE_DIR_NAME'                  + space + str(self.aveDirName          )       + '\n')
        f.write('AVERAGE_FILE_NAME'                 + space + str(self.aveFileName         )       + '\n')

        f.close()


    def __setConfigParsFileName(self, fname=None) :
        if fname == None :
            self._fname = self.confParsDirName + '/' + self.confParsFileName
        else :
            self._fname = fname


    def fillCSPadConfigParsNamedFromWin(self, window):
        self.cspadCurrentDSName    = self.cspadWindowParameters[window][0]
        self.cspadImageAmin        = self.cspadWindowParameters[window][1]
        self.cspadImageAmax        = self.cspadWindowParameters[window][2]
        self.cspadSpectrumAmin     = self.cspadWindowParameters[window][3]
        self.cspadSpectrumAmax     = self.cspadWindowParameters[window][4]
        self.cspadSpectrumNbins    = self.cspadWindowParameters[window][5]
        self.cspadSpectrumBinWidth = self.cspadWindowParameters[window][6]       
        self.cspadImageLimsIsOn    = self.cspadWindowParameters[window][7]
        self.cspadSpectLimsIsOn    = self.cspadWindowParameters[window][8]
        self.cspadBinWidthIsOn     = self.cspadWindowParameters[window][9]
        self.cspadQuad             = self.cspadWindowParameters[window][10]
        self.cspadPair             = self.cspadWindowParameters[window][11]


    def fillCSPadConfigParsWinFromNamed(self, window):
        self.cspadWindowParameters[window][0] = self.cspadCurrentDSName
        self.cspadWindowParameters[window][1] = self.cspadImageAmin       
        self.cspadWindowParameters[window][2] = self.cspadImageAmax       
        self.cspadWindowParameters[window][3] = self.cspadSpectrumAmin    
        self.cspadWindowParameters[window][4] = self.cspadSpectrumAmax    
        self.cspadWindowParameters[window][5] = self.cspadSpectrumNbins   
        self.cspadWindowParameters[window][6] = self.cspadSpectrumBinWidth       
        self.cspadWindowParameters[window][7] = self.cspadImageLimsIsOn
        self.cspadWindowParameters[window][8] = self.cspadSpectLimsIsOn
        self.cspadWindowParameters[window][9] = self.cspadBinWidthIsOn    
        self.cspadWindowParameters[window][10]= self.cspadQuad
        self.cspadWindowParameters[window][11]= self.cspadPair


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
