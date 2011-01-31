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
        """Constructor.

        Contains initial values of main configuration parameters        
        """

        # define default initial values of the configuration parameters

        #self.dirName         = '/reg/neh/home/dubrovin/LCLS/test_h5py'
        #self.fileName        = 'test.h5'
        self.dirName            = '/reg/d/psdm/XPP/xppcom10/hdf5'
        self.fileName           = 'xppcom10-r0546.h5'
        #self.fileName           = 'xppcom10-r0900.h5'
        self.eventCurrent       = 1
        self.span               = 1
        self.list_of_checked_item_names=[]        

        # Status parameters which do not need to be saved
        self.confParsDirName    = '.'
        self.confParsFileName   = 'evtdispconfig'

        self.h5_file_is_open    = False
        self.wtdWindowIsOpen    = False
        self.wtdIMWindowIsOpen  = False
        self.wtdCSWindowIsOpen  = False
        self.wtdWFWindowIsOpen  = False

        self.treeWindowIsOpen   = False
        self.treeViewIsExpanded = False
        self.configGUIIsOpen    = False


        # Default parameters for CSpad plots
        self.cspadAmplitudeRange= 2000

        self.cspadImageIsOn     = True
        self.cspadImageAmin     = 0   
        self.cspadImageAmax     = 1000

        self.cspadSpectrumIsOn  = True
        self.cspadSpectrum08IsOn= False
        self.cspadSpectrumRange = None
        self.cspadSpectrumAmin  = 0   
        self.cspadSpectrumAmax  = 1000
        self.cspadSpectrumNbins = 50


        # Default parameters for Image plots
        self.imageAmplitudeRange= 500

        self.imageImageIsOn     = True
        self.imageImageAmin     = 0    #  15
        self.imageImageAmax     = 100  #  45

        self.imageSpectrumIsOn  = True
        self.imageSpectrumRange = None # (15,45)
        self.imageSpectrumAmin  = 0    #  15
        self.imageSpectrumAmax  = 100  #  45
        self.imageSpectrumNbins = 50   #  30


        # Default parameters for Waveform plots
        self.waveformImageIsOn    = True
        self.waveformSpectrumIsOn = True

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

        print 'CSPAD_IMAGE_IS_ON', self.cspadImageIsOn       
        print 'CSPAD_SPECT_IS_ON', self.cspadSpectrumIsOn    
        print 'CSPAD_SPE08_IS_ON', self.cspadSpectrum08IsOn    
        print 'IMAGE_IMAGE_IS_ON', self.imageImageIsOn       
        print 'IMAGE_SPECT_IS_ON', self.imageSpectrumIsOn    
        print 'WAVEF_IMAGE_IS_ON', self.waveformImageIsOn    
        print 'VAVEF_SPECT_IS_ON', self.waveformSpectrumIsOn 


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
                if   key == 'HDF5_FILE_NAME'    : self.dirName,self.fileName = os.path.split(val)
                elif key == 'N_CHECKED_ITEMS'   : number_of_items = int(val)
                elif key == 'ITEM_NAME'         : self.list_of_checked_item_names.append(val) 
                elif key == 'CURRENT_EVENT'     : self.eventCurrent = int(val)
                elif key == 'SPAN'              : self.span = int(val)
                elif key == 'CSPAD_IMAGE_IS_ON' : self.cspadImageIsOn       = dicBool[val.lower()]
                elif key == 'CSPAD_SPECT_IS_ON' : self.cspadSpectrumIsOn    = dicBool[val.lower()]
                elif key == 'CSPAD_SPE08_IS_ON' : self.cspadSpectrum08IsOn  = dicBool[val.lower()]
                elif key == 'IMAGE_IMAGE_IS_ON' : self.imageImageIsOn       = dicBool[val.lower()]
                elif key == 'IMAGE_SPECT_IS_ON' : self.imageSpectrumIsOn    = dicBool[val.lower()]
                elif key == 'WAVEF_IMAGE_IS_ON' : self.waveformImageIsOn    = dicBool[val.lower()]
                elif key == 'VAVEF_SPECT_IS_ON' : self.waveformSpectrumIsOn = dicBool[val.lower()]

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
        f.write('HDF5_FILE_NAME'    + space + self.dirName + '/' + self.fileName + '\n')
        f.write('N_CHECKED_ITEMS'   + space + str(len(self.list_of_checked_item_names)) + '\n')
        for name in self.list_of_checked_item_names :
            f.write('ITEM_NAME'     + space + str(name)                     + '\n')
        f.write('CURRENT_EVENT'     + space + str(self.eventCurrent)        + '\n')
        f.write('SPAN'              + space + str(self.span)                + '\n')
        f.write('CSPAD_IMAGE_IS_ON' + space + str(self.cspadImageIsOn)      + '\n')
        f.write('CSPAD_SPECT_IS_ON' + space + str(self.cspadSpectrumIsOn)   + '\n')
        f.write('CSPAD_SPE08_IS_ON' + space + str(self.cspadSpectrum08IsOn) + '\n')
        f.write('IMAGE_IMAGE_IS_ON' + space + str(self.imageImageIsOn)      + '\n')
        f.write('IMAGE_SPECT_IS_ON' + space + str(self.imageSpectrumIsOn)   + '\n')
        f.write('WAVEF_IMAGE_IS_ON' + space + str(self.waveformImageIsOn)   + '\n')
        f.write('VAVEF_SPECT_IS_ON' + space + str(self.waveformSpectrumIsOn)+ '\n')
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
