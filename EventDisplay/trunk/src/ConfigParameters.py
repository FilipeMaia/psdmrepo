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
        self.treeViewIsExpanded = False
        self.configGUIIsOpen    = False


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
        print 70*'='


    def readParameters(self, fname=None) :
        self.__setConfigParsFileName(fname)        
        print 'Read parameters from file:', self._fname
        if os.path.exists(self._fname) :
            f=open(self._fname,'r')
            self.list_of_checked_item_names = []
            for line in f :
                key = line.split()[0]
                val = line.split()[1]
                if   key == 'HDF5_FILE_NAME'  : self.dirName,self.fileName = os.path.split(val)
                elif key == 'N_CHECKED_ITEMS' : number_of_items = int(val)
                elif key == 'ITEM_NAME'       : self.list_of_checked_item_names.append(val) 
                elif key == 'CURRENT_EVENT'   : self.eventCurrent = int(val)
                elif key == 'SPAN'            : self.span = int(val)
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
            f.write('ITEM_NAME'     + space + str(name)              + '\n')
        f.write('CURRENT_EVENT'     + space + str(self.eventCurrent) + '\n')
        f.write('SPAN'              + space + str(self.span)         + '\n')
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
