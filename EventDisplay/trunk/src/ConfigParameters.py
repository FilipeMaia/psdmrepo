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

        self.cspadImageAmin       = 0   
        self.cspadImageAmax       = 2000
        self.cspadSpectrumAmin    = 0   
        self.cspadSpectrumAmax    = 2000
        self.cspadSpectrumNbins   = 50
        self.cspadSpectrumRange   = None


        # Default parameters for Image plots
        self.imageImageIsOn       = False
        self.imageSpectrumIsOn    = False
        self.imageImageSpecIsOn   = True

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
        self.waveformAutoRangeIsOn = True
        self.waveformWaveformAmin  = 0
        self.waveformWaveformAmax  = 2000
        self.waveformWaveformTmin  = 0
        self.waveformWaveformTmax  = 10000
        self.waveformNWindows      = 3
        self.waveformNWindowsMax   = 5 # Maximal number of windows for waveforms which can be opened


        
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

        print 'CSPAD_IMAGE_AMIN',          self.cspadImageAmin
        print 'CSPAD_IMAGE_AMAX',          self.cspadImageAmax
        print 'CSPAD_SPECT_AMIN',          self.cspadSpectrumAmin
        print 'CSPAD_SPECT_AMAX',          self.cspadSpectrumAmax

        print 'IMAGE_IMAGE_IS_ON',         self.imageImageIsOn       
        print 'IMAGE_IMAGE_SPEC_IS_ON',    self.imageImageSpecIsOn       
        print 'IMAGE_SPECT_IS_ON',         self.imageSpectrumIsOn    
        print 'WAVEF_WAVEF_IS_ON',         self.waveformWaveformIsOn    

        print 'READ_PARS_AT_START',        self.readParsFromFileAtStart
        print 70*'='


    def readParameters(self, fname=None) :
        self.__setConfigParsFileName(fname)        
        print 'Read parameters from file:', self._fname
        dicBool = {'false':False, 'true':True}
        if os.path.exists(self._fname) :
            f=open(self._fname,'r')
            self.list_of_checked_item_names = []
            for line in f :
                key = line.split()[0]
                val = line.split()[1]
                if   key == 'HDF5_FILE_NAME'           : self.dirName,self.fileName = os.path.split(val)
                elif key == 'N_CHECKED_ITEMS'          : number_of_items = int(val)
                elif key == 'ITEM_NAME'                : self.list_of_checked_item_names.append(val) 
                elif key == 'CURRENT_EVENT'            : self.eventCurrent = int(val)
                elif key == 'SPAN'                     : self.span = int(val)
                elif key == 'CSPAD_IMAGE_IS_ON'        : self.cspadImageIsOn          = dicBool[val.lower()]
                elif key == 'CSPAD_IMAGE_OF_PAIR_IS_ON': self.cspadImageOfPairIsOn    = dicBool[val.lower()]
                elif key == 'CSPAD_IMAGE_QUAD_IS_ON'   : self.cspadImageQuadIsOn      = dicBool[val.lower()]
                elif key == 'CSPAD_IMAGE_DET_IS_ON'    : self.cspadImageDetIsOn       = dicBool[val.lower()]
                elif key == 'CSPAD_SPECT_IS_ON'        : self.cspadSpectrumIsOn       = dicBool[val.lower()]
                elif key == 'CSPAD_SPE08_IS_ON'        : self.cspadSpectrum08IsOn     = dicBool[val.lower()]
                elif key == 'IMAGE_IMAGE_IS_ON'        : self.imageImageIsOn          = dicBool[val.lower()]
                elif key == 'IMAGE_IMAGE_SPEC_IS_ON'   : self.imageImageSpecIsOn      = dicBool[val.lower()]
                elif key == 'IMAGE_SPECT_IS_ON'        : self.imageSpectrumIsOn       = dicBool[val.lower()]
                elif key == 'WAVEF_WAVEF_IS_ON'        : self.waveformWaveformIsOn    = dicBool[val.lower()]
                elif key == 'READ_PARS_AT_START'       : self.readParsFromFileAtStart = dicBool[val.lower()]
                elif key == 'CSPAD_QUAD_NUMBER'        : self.cspadQuad         = int(val)
                elif key == 'CSPAD_PAIR_NUMBER'        : self.cspadPair         = int(val)
                elif key == 'CSPAD_IMAGE_AMIN'         : self.cspadImageAmin    = int(val)
                elif key == 'CSPAD_IMAGE_AMAX'         : self.cspadImageAmax    = int(val)
                elif key == 'CSPAD_SPECT_AMIN'         : self.cspadSpectrumAmin = int(val)
                elif key == 'CSPAD_SPECT_AMAX'         : self.cspadSpectrumAmax = int(val)


                else : print 'The record : %s %s \n is UNKNOWN in readParameters()' % (key, val) 
            f.close()
        else :
            print 'The file %s does not exist' % (fname)
            print 'WILL USE DEFAULT CONFIGURATION PARAMETERS'


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
        f.write('IMAGE_IMAGE_IS_ON'         + space + str(self.imageImageIsOn)          + '\n')
        f.write('IMAGE_IMAGE_SPEC_IS_ON'    + space + str(self.imageImageSpecIsOn)      + '\n')
        f.write('IMAGE_SPECT_IS_ON'         + space + str(self.imageSpectrumIsOn)       + '\n')
        f.write('WAVEF_WAVEF_IS_ON'         + space + str(self.waveformWaveformIsOn)    + '\n')
        f.write('READ_PARS_AT_START'        + space + str(self.readParsFromFileAtStart) + '\n')
        f.write('CSPAD_QUAD_NUMBER'         + space + str(self.cspadQuad)               + '\n')
        f.write('CSPAD_PAIR_NUMBER'         + space + str(self.cspadPair)               + '\n')
        f.write('CSPAD_IMAGE_AMIN'          + space + str(self.cspadImageAmin)          + '\n')
        f.write('CSPAD_IMAGE_AMAX'          + space + str(self.cspadImageAmax)          + '\n')
        f.write('CSPAD_SPECT_AMIN'          + space + str(self.cspadSpectrumAmin)       + '\n')
        f.write('CSPAD_SPECT_AMAX'          + space + str(self.cspadSpectrumAmax)       + '\n')

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
