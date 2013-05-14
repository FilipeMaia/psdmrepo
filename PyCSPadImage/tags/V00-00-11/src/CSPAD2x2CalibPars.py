#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CSPAD2x2CalibPars...
#
#------------------------------------------------------------------------

"""
This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 2013-05-10$

@author Mikhail S. Dubrovin
"""

#---------------------
import sys
import os

import numpy as np
import CSPAD2x2CalibParsDefault as cpd
from CalibPars import findCalibFile

#---------------------
#  Class definition --
#---------------------

class CSPAD2x2CalibPars (object) :
    """This class provides access to the calibration parameters.
    """
    #enum_status = {0:'DEFAULT', 1:'FROM_FILE'}

    list_of_clib_types_total =[
         'center'
        ,'tilt'
        ,'beam_vector'
        ,'common_mode'
        ,'pedestals'
        ,'filter'
        ,'pixel_status'
        ]

#---------------------
    # Example of parameters:
    # official : path = '/reg/d/psdm/mec/mec73313/calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/'
    # local    : path = '/reg/neh/home1/dubrovin/LCLS/CSPad2x2Alignment/calib-cspad2x2-02-2013-02-13/'
    # run = 10 

    def __init__ (self, path=None, run=None, list_of_clib_types=None) :
        """Class constructor:
           The path and run need to be defined to instantiate the object and load correct set of parameters.
           If path or run is omitted, default parameters will be used. 
           list_of_clib_types - optional parameter for optimization of time; only types from the list are loaded.
           If the list_of_clib_types is omitted, all parameters will be loaded and used.
        """    
        self.cpars        = {}
        self.cpars_status = {} # 'DEFAULT', 'FROM_FILE'

        self.run = run
        self.path_to_calib_types = path

        self.setListOfCalibTypes(list_of_clib_types)
        
        self.setCalibParsDefault()

        if path!=None and run!=None :
            self.setCalibParsForPath(run, path)

#---------------------

    def setListOfCalibTypes(self, list_of_clib_types) :
        """Defines the internal list of calibration types which will be used.
        """    
        if list_of_clib_types == None :
            self.list_of_clib_types = self.list_of_clib_types_total
            return

        self.list_of_clib_types = []
        for type in list_of_clib_types :
            if type in self.list_of_clib_types_total :
                self.list_of_clib_types.append(type)
            else :
                msg = 'WARNING: TYPE ' + type + ' IS UNKNOWN FOR CSPAD2x2' + \
                      '\n KNOWN TYPES:' + str(self.list_of_clib_types_total)
                print msg

#---------------------

    def setCalibParsDefault (self) :
        """Loads default calibration parameters from singleton object.
        """    
        #print 'Set default calibration parameters'
        for type in self.list_of_clib_types :
            self.cpars[type] = self.getCalibParsDefault (type)
            self.cpars_status[type] = 'DEFAULT'

#---------------------

    def getCalibParsDefault (self, type) :
        """Returns the default calibration parameters for specified type.
        """    
        return cpd.cspad2x2calibparsdefault.getCalibParsDefault (type)

#---------------------

    def setRun ( self, run=None ) :
        """Resets the run number and loads calibration parameters, if calib files are available.
        """    
        if run!=None  :
            self.run = run
            #print 'Load the calibration parameters for run ', self.run
            self.setCalibParsForPath()

#---------------------

    def setCalibPars (self,
                      run      = None, 
                      calibdir = '/reg/d/psdm/mec/mec73313/calib',
                      group    = 'CsPad2x2::CalibV1',
                      source   = 'MecTargetChamber.0:Cspad2x2.1') :
        """Set calibration parameters for specified input pars.
        """
        self.setCalibParsForPath (run, calibdir + '/' + group + '/' + source)

#---------------------

    def setCalibParsForPath (self,
                             run  = None,
                             path = '/reg/d/psdm/mec/mec73313/calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1' ) :
        """ Set calibration parameters for specified input pars.
        """
        if run!=None  : self.run = run
        if path!=None : self.path_to_calib_types = path

        self.loadAllCalibPars ()

#---------------------

    def loadAllCalibPars (self) :
        """Loads all calibration parameters, if the files are available or set default.
        """

        self.cpars = {}

        for type in self.list_of_clib_types :
            fname = findCalibFile (self.path_to_calib_types, self.run, type)
            #print 'Load calibpars: ', fname

            cpars_for_type = self.loadCalibParsFromFileOrDefault (fname, type)

            print 'cpars_for_type.shape = ', cpars_for_type.shape
        
            # Special case of array shapes:
            if type == 'pedestals' \
            or type == 'pixel_status' : cpars_for_type.shape = (185,388,2)

            self.cpars[type] = cpars_for_type

#---------------------

    def loadCalibParsFromFileOrDefault (self, fname, type) :
        """Load parameters of specified type from file or set default.
        """

        if fname == None :
            self.cpars_status[type] = 'DEFAULT'
            #print 'WARNING: CALIBRATION FILE ', fname, '\nDOES NOT EXIST, WILL USE DEFAULT CONSTANTS.'
            return self.getCalibParsDefault (type)

        try :
            self.cpars_status[type] = 'FROM_FILE'
            return np.loadtxt (fname)

        except IOError :
            print 80*'!'
            print 'WARNING: CALIBRATION FILE\n', fname, '\nIS CORRUPTED, WILL USE DEFAULT CONSTANTS.'
            print 80*'!'
            self.cpars_status[type] = 'DEFAULT'
            return self.getCalibParsDefault (type)

#---------------------

    def printCalibFiles (self) :
        """Print the list of calibration files for this object.
        """
        for type in self.list_of_clib_types :
            fname = findCalibFile (self.path_to_calib_types, self.run, type)
            print 'Calib type: %15s has file: %s' % (type, fname)

#---------------------

    def printListOfCalibTypes (self) :
        """Print the list of calibration types for this object.
        """
        print 'list_of_clib_types:'
        for type in self.list_of_clib_types : print '   ' + type

#---------------------

    def printCalibPars (self) :
        """Print all calibration parameters.
        """
        for type in self.list_of_clib_types :
            print '\nCalibration constants type "' + type + '" with shape' + str(self.cpars[type].shape)
            print self.cpars[type]
            
#---------------------

    def printCalibParsStatus (self) :
        """Print status of calibration parameters for all specified files.
        """
        print 'Status of CSPAD2x2 calibration parameters:'
        for type, status in self.cpars_status.iteritems() :
            print 'Type: %s    Status: %s    Shape: %s' % (type.ljust(12), status.ljust(10), str(self.cpars[type].shape))

#---------------------

    def getCalibPars (self, type) :
        """Returns the numpy array of with calibration parameters for specified type.
        """
        if type in self.list_of_clib_types :
            return self.cpars[type]
        else :
            msg = 'WARNING: THE REQUESTED TYPE OF CALIBRATION PARS "' + type + \
                   '" IS NOT FOUND IN THE AVAILABLE LIST:\n' + str(self.list_of_clib_types)
            print msg
            return None

#---------------------
#-- Global methods ---
#---------------------

def data2x2ToTwo2x1(arr2x2) :
    """Converts array shaped as CSPAD2x2 data (185,388,2)
       to two 2x1 arrays with shape=(2,185,388).
    """
    return np.array([arr2x2[:,:,0], arr2x2[:,:,1]])

#---------------------

def two2x1ToData2x2(arrTwo2x1) :
    """Converts array shaped as two 2x1 arrays (2,185,388)
       to CSPAD2x2 data shape=(185,388,2).
    """
    arr2x2 = np.array(zip(arrTwo2x1[0].flatten(), arrTwo2x1[1].flatten()))
    arr2x2.shape = (185,388,2)
    return arr2x2

#---------------------
#---------------------
#---------------------
#------- TEST --------
#---------------------
#---------------------
#---------------------

def main_test() :

    path = '/reg/d/psdm/mec/mec73313/calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/'
    #path = '/reg/neh/home1/dubrovin/LCLS/CSPad2x2Alignment/calib-cspad2x2-02-2013-02-13/'
    run = 180

    #calib = CSPAD2x2CalibPars()                                     # Sets all default calibration parameters
    #calib = CSPAD2x2CalibPars(list_of_clib_types=['center', 'tilt', 'pedestals']) # Sets default calibration parameters
    #calib.setCalibParsForPath (run, path)                           
    #calib = CSPAD2x2CalibPars(path, run)
    calib = CSPAD2x2CalibPars(path, run) #, ['center', 'tilt', 'pedestals'])

    #print 'pedestals:\n', calib.getCalibPars('pedestals')
    print 'center:\n', calib.getCalibPars('center')
    print 'tilt:\n',   calib.getCalibPars('tilt')

    calib.printCalibParsStatus()
    #calib.printListOfCalibTypes()
    #calib.printCalibPars()

def test_reshaping_arrs_for_cspad2x2() :

    run  = 180
    path = '/reg/d/psdm/mec/mec73313/calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/'
    #path = '/reg/neh/home1/dubrovin/LCLS/CSPad2x2Alignment/calib-cspad2x2-01-2013-02-13/'
    calib = CSPAD2x2CalibPars(path, run, ['center', 'tilt', 'pedestals'])

    raw_arr = calib.getCalibPars('pedestals')
    ord_arr = data2x2ToTwo2x1(raw_arr)
    tst_arr = two2x1ToData2x2(ord_arr)

    print 'raw_arr:', raw_arr
    print 'ord_arr:', ord_arr
    print 'tst_arr:', tst_arr

    print 'raw_arr.shape:', raw_arr.shape
    print 'ord_arr.shape:', ord_arr.shape
    print 'tst_arr.shape:', tst_arr.shape

    if np.equal(tst_arr,raw_arr).all() : print 'Arrays are equal after two transformations'
    else                               : print 'Arrays are NOT equal after two transformations'

if __name__ == "__main__" :
    main_test()
    #test_reshaping_arrs_for_cspad2x2()
    sys.exit ( 'End of job' )

#----------------------------------------------
