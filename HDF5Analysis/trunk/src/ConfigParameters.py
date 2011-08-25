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

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor"""
        self.setRunTimeParametersInit()
        self.setDefaultParameters()


    def setRunTimeParametersInit ( self ) :

        self.h5_file_is_open         = False
        self.dsetsGUIIsOpen          = False
        self.treeWindowIsOpen        = False
        self.treeViewIsExpanded      = False
        self.configGUIIsOpen         = False

        self.dsTreeIsExpanded        = False

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

        self.list_of_checked_item_names=[]
        self.list_of_checked_item_names.append('/Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/CxiSc1.0:Tm6740.1/image')       
        #self.list_of_checked_item_names.append('/Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/CxiSc1.0:Tm6740.2/image')
        #self.list_of_checked_item_names.append('/Configure:0000/Run:0000/CalibCycle:0000/Camera::FrameV1/CxiSc1.0:Tm6740.3/image')

        # Status parameters which do not need to be saved
        self.confParsDirName         = '.'
        self.confParsFileName        = 'confpars'

        self.posGUIMain              = (370,10)

        self.readParsFromFileAtStart = True
        self.current_item_name_for_title = 'Current extended item name for title'

 
        # Default parameters for Selection algorithms
        self.selectionNWindows      = 1
        self.selectionNWindowsMax   = 10 # Maximal number of windows for selection algorithms

        self.dsWindowParameters = []
        for win in range(self.selectionNWindowsMax) :
            self.dsWindowParameters.append([0, True, 0, 1000, 0, 1000, 'None', None])
                                                 #[Theshold, InBin, Xmin, Xmax, Ymin, Ymax, dslist]



    #-------------------
    #  Public methods --
    #-------------------

    def Print ( self ) :
        """Prints current values of configuration parameters
        """
        print '\nConfigParameters'
        print 'READ_PARS_AT_START',        self.readParsFromFileAtStart
        print 'HDF5 file : %s' % ( self.dirName + '/' + self.fileName )
        print 'Number of items to plot =', len(self.list_of_checked_item_names)
        for name in self.list_of_checked_item_names :
            print str(name)

        print 'STEP_01_IS_DONE',           self.step01IsDone
        print 'STEP_02_IS_DONE',           self.step02IsDone
        print 'STEP_03_IS_DONE',           self.step03IsDone
        print 'STEP_04_IS_DONE',           self.step04IsDone
        
        print 'DATASET_N_WINDOWS_MAX',       self.selectionNWindowsMax 
        print 'DATASET_N_WINDOWS',           self.selectionNWindows 

        for win in range(self.selectionNWindows) :

            print 'DATASET_WINDOW_NUMBER',   win 
            print 'DATASET_THRESHOLD',       self.dsWindowParameters[win][0] 
            print 'DATASET_IN_BIN',          self.dsWindowParameters[win][1] 
            print 'DATASET_XMIN',            self.dsWindowParameters[win][2] 
            print 'DATASET_XMAX',            self.dsWindowParameters[win][3] 
            print 'DATASET_YMIN',            self.dsWindowParameters[win][4] 
            print 'DATASET_YMAX',            self.dsWindowParameters[win][5] 
            print 'DATASET_NAME',            self.dsWindowParameters[win][6] 

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

                elif key == 'READ_PARS_AT_START'       : self.readParsFromFileAtStart = dicBool[val.lower()]

                elif key == 'STEP_01_IS_DONE'          : self.step01IsDone            = dicBool[val.lower()]
                elif key == 'STEP_02_IS_DONE'          : self.step02IsDone            = dicBool[val.lower()]
                elif key == 'STEP_03_IS_DONE'          : self.step03IsDone            = dicBool[val.lower()]
                elif key == 'STEP_04_IS_DONE'          : self.step04IsDone            = dicBool[val.lower()]

                elif key == 'DATASET_N_WINDOWS_MAX'      : self.selectionNWindowsMax     = int(val)
                elif key == 'DATASET_N_WINDOWS'          : self.selectionNWindows        = int(val)

                elif key == 'DATASET_WINDOW_NUMBER'      : win                           = int(val)
                elif key == 'DATASET_THRESHOLD'          : self.dsWindowParameters[win][0] = int(val)
                elif key == 'DATASET_IN_BIN'             : self.dsWindowParameters[win][1] = dicBool[val.lower()]
                elif key == 'DATASET_XMIN'               : self.dsWindowParameters[win][2] = int(val)
                elif key == 'DATASET_XMAX'               : self.dsWindowParameters[win][3] = int(val)
                elif key == 'DATASET_YMIN'               : self.dsWindowParameters[win][4] = int(val)
                elif key == 'DATASET_YMAX'               : self.dsWindowParameters[win][5] = int(val)
                elif key == 'DATASET_NAME'               : self.dsWindowParameters[win][6] = val

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

        f.write('READ_PARS_AT_START'        + space + str(self.readParsFromFileAtStart) + '\n')
        f.write('HDF5_FILE_NAME'            + space + self.dirName + '/' + self.fileName + '\n')
        f.write('N_CHECKED_ITEMS'           + space + str(len(self.list_of_checked_item_names)) + '\n')
        for name in self.list_of_checked_item_names :
            f.write('ITEM_NAME'             + space + str(name)                         + '\n')

        f.write('STEP_01_IS_DONE'           + space + str(self.step01IsDone)            + '\n')
        f.write('STEP_02_IS_DONE'           + space + str(self.step02IsDone)            + '\n')
        f.write('STEP_03_IS_DONE'           + space + str(self.step03IsDone)            + '\n')
        f.write('STEP_04_IS_DONE'           + space + str(self.step04IsDone)            + '\n')

        f.write('\n')
        f.write('DATASET_N_WINDOWS_MAX'       + space + str(self.selectionNWindowsMax)     + '\n')
        f.write('DATASET_N_WINDOWS'           + space + str(self.selectionNWindows)        + '\n')

        for win in range(self.selectionNWindows) :
            f.write('\n')
            f.write('DATASET_WINDOW_NUMBER'   + space + str(win)                                       + '\n')
            f.write('DATASET_THRESHOLD'       + space + str(self.dsWindowParameters[win][0] )   + '\n')
            f.write('DATASET_IN_BIN'          + space + str(self.dsWindowParameters[win][1] )   + '\n')
            f.write('DATASET_XMIN'            + space + str(self.dsWindowParameters[win][2] )   + '\n')
            f.write('DATASET_XMAX'            + space + str(self.dsWindowParameters[win][3] )   + '\n')
            f.write('DATASET_YMIN'            + space + str(self.dsWindowParameters[win][4] )   + '\n')
            f.write('DATASET_YMAX'            + space + str(self.dsWindowParameters[win][5] )   + '\n')
            f.write('DATASET_NAME'            + space + str(self.dsWindowParameters[win][6] )   + '\n')

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
