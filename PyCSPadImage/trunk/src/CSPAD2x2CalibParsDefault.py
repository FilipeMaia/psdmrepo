#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CSPAD2x2CalibParsDefault...
#
#------------------------------------------------------------------------

"""
This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 2013-05-10$

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
import numpy as np

#---------------------
#  Class definition --
#---------------------

class CSPAD2x2CalibParsDefault (object) :
    """This class provides access to the CSPAD2x2 calibration parameters
    """
    list_of_clib_types =[
         'center'
        ,'tilt'
        ,'beam_vector'
        ,'common_mode'
        ,'pedestals'
        ,'pixel_status'
        ,'filter'
        ]

#---------------------

    def __init__ (self) :

        self.loadCSPAD2x2CalibParsDefault()

#---------------------

    def loadCSPAD2x2CalibParsDefault (self) :

        self.defpars = {}

        self.defpars['center'] = np.array(  [[198., 198.],
                                             [ 95., 308.],
                                             [  0.,   0.]])

        self.defpars['tilt']          = np.zeros((2), dtype=np.float32)

        self.defpars['common_mode']   = np.array([1, 100, 30])

        self.defpars['pedestals']     = np.zeros((185, 388, 2), dtype=np.float32)

        self.defpars['pixel_status']  = np.zeros((185, 388, 2), dtype=np.uint16)

        self.defpars['beam_vector']   = np.zeros((3), dtype=np.float32)

        self.defpars['filter']        = np.array([1, 100, 10])

#---------------------

    def printCalibParsDefault (self, partype=None) :
        """Print the calibration prarameters of specified partype or all for dafault.
        """        
        if partype==None :
            for type in self.list_of_clib_types :
                print '\nprintCalibParsDefault(): Calibration constants type "' + type + '"' # + '" with shape', self.cpars[type].shape
                print self.defpars[type]
        else :
            if partype in self.list_of_clib_types :
                print '\nprintCalibParsDefault(): Calibration constants type "' + partype + '"' # + '" with shape', self.cpars[type].shape
                print self.defpars[partype]
            else :
                msg =  'WARNING: THE REQUESTED TYPE OF CALIBRATION PARS "' + partype + \
                       '" IS NOT FOUND IN THE AVAILABLE LIST:\n' + str(self.list_of_clib_types)
                print msg
            
#---------------------

    def printListOfCalibTypes (self) :
        print '\nprintListOfCalibTypes(): list_of_clib_types:' #, self.list_of_clib_types
        for type in self.list_of_clib_types : print '    ', type

#---------------------

    def getCalibParsDefault (self, type) :

        if type in self.list_of_clib_types :
            return self.defpars[type]
        else :
            msg = 'WARNING: THE REQUESTED TYPE OF CALIBRATION PARS "' + type + \
                  '" IS NOT FOUND IN THE AVAILABLE LIST:\n' + str(self.list_of_clib_types)
            print msg
            return None

#---------------------------------------

cspad2x2calibparsdefault = CSPAD2x2CalibParsDefault()

#----------------------------------------------
# In case someone decides to run this module --
#----------------------------------------------

def main_test() :
    cspad2x2calibparsdefault.printCalibParsDefault()
    cspad2x2calibparsdefault.printListOfCalibTypes()

if __name__ == "__main__" :
    main_test()
    sys.exit ( 'End of job' )

#----------------------------------------------
