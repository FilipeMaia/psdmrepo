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

       #self.isSetWarningModel       = False
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

        self.dirName              = '/reg/d/psdm/CXI/cxitut13/hdf5'
        self.fileName             = 'cxitut13-r0135.h5'

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
        self.dsNWindows      = 1
        self.dsNWindowsMax   = 10 # Maximal number of windows for selection algorithms

        self.dsWindowParameters = []
        for win in range(self.dsNWindowsMax) :
            self.dsWindowParameters.append(['None', 0, None])
                                          #[DS_NAME_IN_HDF5, DS_N_CHECKED_DS_IN_WIN, DS_INDEX_LIST]

    #-------------------
    #  Public methods --
    #-------------------

    def Print ( self ) :
        """Prints current values of configuration parameters"""

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
        
        print 'DS_N_WINDOWS_MAX',          self.dsNWindowsMax 
        print 'DS_N_WINDOWS',              self.dsNWindows 

        for win in range(self.dsNWindows) :

            print 'DS_WINDOW_NUMBER',       win 
            print 'DS_NAME_IN_HDF5',        self.dsWindowParameters[win][0] 
            print 'DS_N_CHECKED_DS_IN_WIN', self.dsWindowParameters[win][1] 

            if self.dsWindowParameters[win][1] != 0 :
                list_of_indexes_2d =        self.dsWindowParameters[win][2]
                if list_of_indexes_2d == None : continue
                for list_of_indexes_of_checked_ds in list_of_indexes_2d :
                    print 'DS_N_INDEXES_IN_DS', len(list_of_indexes_of_checked_ds) 
                    for index in list_of_indexes_of_checked_ds :
                        print 'DS_INDEX_IN_LIST', index


        print 70*'='
        self.print_all_checked_dataset_indexes()
        print 70*'='


    def readParameters(self, fname=None) :
        """Read the configuration parameters form the file"""

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

                elif key == 'DS_N_WINDOWS_MAX'      : self.dsNWindowsMax     = int(val)
                elif key == 'DS_N_WINDOWS'          : self.dsNWindows        = int(val)

                elif key == 'DS_WINDOW_NUMBER'      :
                    win = int(val)
                    self.list_of_indexes_2d = []
                elif key == 'DS_NAME_IN_HDF5'       : self.dsWindowParameters[win][0] = val
                elif key == 'DS_N_CHECKED_DS_IN_WIN': self.dsWindowParameters[win][1] = int(val)
                elif key == 'DS_N_INDEXES_IN_DS'    :
                    self.num_of_indexes = int(val)
                    self.list_of_indexes_of_checked_ds = []

                elif key == 'DS_INDEX_IN_LIST' :
                    self.list_of_indexes_of_checked_ds.append(val)
                    if len(self.list_of_indexes_of_checked_ds) == self.num_of_indexes :
                        self.list_of_indexes_2d.append(self.list_of_indexes_of_checked_ds)
                        if len(self.list_of_indexes_2d) == self.dsWindowParameters[win][1] :
                            self.dsWindowParameters[win][2] = self.list_of_indexes_2d

                else :
                    print 'The record : %s %s \n is UNKNOWN in readParameters()' % (key, val) 
            f.close()
        else :
            print 'The file %s does not exist' % (fname)
            print 'WILL USE DEFAULT CONFIGURATION PARAMETERS'


    def getValIntOrNone(self,val) :
        if val == 'None' : return None
        else :             return int(val)


    def writeParameters(self, fname=None) :
        """Write the configuration parameters in the file"""

        self.__setConfigParsFileName(fname)        
        print 'Write parameters in file:', self._fname
        space = '    '
        f=open(self._fname,'w')

        f.write('READ_PARS_AT_START'        + space + str(self.readParsFromFileAtStart)         + '\n')
        f.write('HDF5_FILE_NAME'            + space + self.dirName + '/' + self.fileName        + '\n')
        f.write('N_CHECKED_ITEMS'           + space + str(len(self.list_of_checked_item_names)) + '\n')
        for name in self.list_of_checked_item_names :
            f.write('ITEM_NAME'             + space + str(name)                         + '\n')

        f.write('STEP_01_IS_DONE'           + space + str(self.step01IsDone)            + '\n')
        f.write('STEP_02_IS_DONE'           + space + str(self.step02IsDone)            + '\n')
        f.write('STEP_03_IS_DONE'           + space + str(self.step03IsDone)            + '\n')
        f.write('STEP_04_IS_DONE'           + space + str(self.step04IsDone)            + '\n')

        f.write('\n')
        f.write('DS_N_WINDOWS_MAX'       + space + str(self.dsNWindowsMax)     + '\n')
        f.write('DS_N_WINDOWS'           + space + str(self.dsNWindows)        + '\n')

        for win in range(self.dsNWindows) :
            f.write('\n')
            f.write('DS_WINDOW_NUMBER'       + space + str(win)                                + '\n')
            f.write('DS_NAME_IN_HDF5'        + space + str(self.dsWindowParameters[win][0] )   + '\n')
            f.write('DS_N_CHECKED_DS_IN_WIN' + space + str(self.dsWindowParameters[win][1] )   + '\n')

            if self.dsWindowParameters[win][1] != 0 :
                list_of_indexes_2d =        self.dsWindowParameters[win][2]
                if list_of_indexes_2d == None : continue
                for list_of_indexes_of_checked_ds in list_of_indexes_2d :
                    f.write('DS_N_INDEXES_IN_DS' + space + str(len(list_of_indexes_of_checked_ds)) + '\n') 
                    for index in list_of_indexes_of_checked_ds :
                        f.write('DS_INDEX_IN_LIST' + space + str(index ) + '\n')

        f.close()


    def __setConfigParsFileName(self, fname=None) :
        if fname == None :
            self._fname = self.confParsDirName + '/' + self.confParsFileName
        else :
            self._fname = fname


    def get_list_of_indexes_of_all_checked_datasets(self) :
        """Join lists of the checked ds indexes from different windows in a single list"""
        
        list_of_indexes_of_all_checked_ds = []
        for win in range(self.dsNWindows) :
            if self.dsWindowParameters[win][2] == None : continue
            for list_of_indexes_1d in self.dsWindowParameters[win][2] :
                list_of_idexes_of_one_ds = []
                # list_of_idexes_of_one_ds.append( self.dsWindowParameters[win][0] ) # is already in list...
                for index in list_of_indexes_1d : 
                    list_of_idexes_of_one_ds.append(index)
                list_of_indexes_of_all_checked_ds.append(list_of_idexes_of_one_ds)

        return list_of_indexes_of_all_checked_ds


    def print_all_checked_dataset_indexes(self) :
        print """Print all checked dataset indexes"""

        list_of_indexes_2d = self.get_list_of_indexes_of_all_checked_datasets()
        if list_of_indexes_2d == None :
            print 'There is no checked item in the list...'
            return
        for list_of_indexes_1d in list_of_indexes_2d :
            print list_of_indexes_1d

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
