#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CalibPars...
#
#------------------------------------------------------------------------

"""This module provides access to the calibration parameters

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 2008-09-22$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#----------
#  Imports 
#----------
import sys
import os
#from PyQt4 import QtGui, QtCore
import numpy as np
import CalibParsDefault   as cpd
import CalibParsEvaluated as cpe

#---------------------
#  Class definition --
#---------------------

class CalibPars (object) :
    """This class provides access to the calibration parameters
    """
    list_of_clib_types_total =[
         'center'
        ,'center_corr'  
        ,'marg_gap_shift' 
        ,'offset'
        ,'offset_corr'
        ,'rotation'
        ,'tilt'
        ,'quad_rotation'
        ,'quad_tilt'
        ,'center_global'  
        ,'tilt_global'  
        ,'beam_vector'
        ,'beam_intersect'
        ,'common_mode'
        #,'pedestals'
        #,'filter'
        #,'pixel_status'
        ]

#---------------------

    def __init__ (self, path=None, run=None, list_of_clib_types=None) :

        self.cpars        = {}
        self.cpars_status = {} # 'DEFAULT', 'FROM_FILE', 'EVALUATED'

        self.cpeval = None # Will be defined after input of all parameters

        self.run = run # 10
        self.path_to_calib_types = path 

        self.setListOfCalibTypes(list_of_clib_types)
        
        self.setCalibParsDefault()

        if path!=None and run!=None :
            self.setCalibParsForPath(run, path)

#---------------------

    def setListOfCalibTypes(self, list_of_clib_types) :

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

        #print 'Set default calibration parameters'
        for type in self.list_of_clib_types :
            self.cpars[type] = self.getCalibParsDefault (type)
            self.cpars_status[type] = 'DEFAULT'

#---------------------

    def getCalibParsDefault (self, type) :
        return cpd.calibparsdefault.getCalibParsDefault (type)

#---------------------

    def setRun ( self, run=None ) :
        if run!=None  :
            self.run = run
            #print 'Load the calibration parameters for run ', self.run
            self.setCalibParsForPath()

#---------------------

    def setCalibPars (self,
                      run      = None, # 1
                      calibdir = '/reg/d/psdm/CXI/cxi35711/calib',
                      group    = 'CsPad::CalibV1',
                      source   = 'CxiDs1.0:Cspad.0') :
        """ Set calibration parameters for specified input pars"""

        self.setCalibParsForPath (run, calibdir + '/' + group + '/' + source)

#---------------------

    def setCalibParsForPath (self,
                             run  = None, # 1
                             path = '/reg/d/psdm/CXI/cxi35711/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0' ) :
        """ Set calibration parameters for specified input pars"""

        if run!=None  : self.run = run
        if path!=None : self.path_to_calib_types = path

        self.loadAllCalibPars ()

#---------------------

    def loadAllCalibPars (self) :

        self.cpars = {}

        #print 'Load the calibration parameters for run ', self.run

        for type in self.list_of_clib_types :
            fname = findCalibFile (self.path_to_calib_types, type, self.run) # self.path_to_calib_types + type + '/0-end.data'
            #print 'Load calibpars: ', fname

            cpars_for_type = self.loadCalibParsFromFileOrDefault (fname, type)

            # Special case of 3-d arrays:
            if type == 'center' \
            or type == 'center_corr' \
            or type == 'center_global' : cpars_for_type.shape = (3,4,8)
            #print 'cpars_for_type.shape = ', cpars_for_type.shape
            self.cpars[type] = cpars_for_type

        #=================================
        self.cpeval = cpe.CalibParsEvaluated(self)
        self.cpeval.setCalibParsEvaluated()
        #self.cpeval.printCalibParsEvaluated ('center_global') 
        #=================================

#---------------------

    def loadCalibParsFromFileOrDefault (self, fname, type) :

        if fname == None :
            print 'WARNING: CALIBRATION FILE\n', fname, '\nDOES NOT EXIST, WILL USE DEFAULT CONSTANTS.'
            self.cpars_status[type] = 'DEFAULT'
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
        for type in self.list_of_clib_types :
            fname = findCalibFile (self.path_to_calib_types, type, self.run) # self.path_to_calib_types + type + '/0-end.data'
            print 'Calib type: %15s has file: %s' % (type, fname)

#---------------------

    def printListOfCalibTypes (self) :
        print 'list_of_clib_types:'
        for type in self.list_of_clib_types : print '   ' + type

#---------------------

    def printCalibPars (self) :
        for type in self.list_of_clib_types :
            print '\nCalibration constants type "' + type + '" with shape' + str(self.cpars[type].shape)
            print self.cpars[type]
            
#---------------------

    def getCalibPars (self, type) :
        if type in self.list_of_clib_types :
            return self.cpars[type]
        else :
            print  'getCalibPars() WARNING: THE REQUESTED TYPE OF CALIBRATION PARS "', type, \
                   '" IS NOT FOUND IN THE AVAILABLE LIST:\n', self.list_of_clib_types
            return None

#---------------------

def findCalibFile (path_to_clib_types, type=None, run=None) :
    """Use the run number, self.path_to_calib_types, and type of the calibration constants.
       From the directory self.path_to_calib_types + '/' + type select the file
       which run range is valid for requested run.
       None is returned if the file is not found.
    """

    err_msg_prefix = 'findCalibFile(): ERROR in findCalibFile(path, type, run): '

    if type==None :
        print  err_msg_prefix + 'type IS NOT SPECIFIED'
        return None

    if run==None :
        print  err_msg_prefix + 'run IS NOT SPECIFIED'
        return None

    if path_to_clib_types[-1] != '/' : path = path_to_clib_types + '/' + type
    else                             : path = path_to_clib_types + type

    run_max = 9999
    calibfname = None

    if not os.path.exists(path) :
        print  'WARNING in findCalibFile(): PATH %s DOES NOT EXIST.' % path
        return calibfname

    flist = os.listdir(path)
    if len(flist) > 1 : flist.sort()

    for fname in flist :

        if fname[-1] == '~' : continue # skip old files with ~(tilde) at the end

        basename = fname.split('.') # Assume: basename[0]='0-end', basename[1]='data' 
        basename_beg, basename_end = basename[0].split('-')

        run_beg = int(basename_beg)
        if basename_end == 'end' :
            run_end = run_max
        else :
            run_end = int(basename_end)

        # Find the last file in the list which run number is in the range
        if run_beg <= run and run <= run_end :
            calibfname = fname

        #print fname, basename[0], run, run_beg, run_end, calibfname

    if calibfname != None : calibfname = path + '/' + calibfname

    #print 'calibfname = ', calibfname
    return calibfname

#----------------------------------------------
# In case someone decides to run this module --
#----------------------------------------------

def main() :

    calibpars = CalibPars() # Sets default calibration parameters.

    calibpars.printListOfCalibTypes()
    calibpars.printCalibPars() # prints the default calib pars

    #calibpars.setCalibPars(10, '/reg/d/psdm/CXI/cxi35711/calib', 'CsPad::CalibV1', 'CxiDs1.0:Cspad.0')
    calibpars.setCalibParsForPath (run=10, path='/reg/d/psdm/CXI/cxi35711/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0')
    calibpars.printCalibPars()


    print '\n\nPRINT EVALUATED PARAMETERS:' 
    calibpars.cpeval.printCalibParsEvaluatedAll() 

    #calibpars.printCalibFiles()
    #print calibpars.getCalibPars('offset')
    #print calibpars.getCalibPars('XxX')
    #cpd.calibparsdefault.printCalibParsDefault()

    print 'End of test'

if __name__ == "__main__" :

    main()
    sys.exit ( 'End of job' )

#----------------------------------------------
