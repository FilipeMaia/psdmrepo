#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CalibParsDefault...
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

#---------------------
#  Class definition --
#---------------------

class CalibParsDefault (object) :
    """This class provides access to the calibration parameters
    """

#---------------------

    def __init__ (self) :
        """Constructor"""

        self.list_of_clib_types =[  'center'
                                   ,'center_corr'  
                                   ,'marg_gap_shift' 
                                   ,'offset'
                                   ,'offset_corr'
                                   ,'rotation'
                                   ,'tilt'
                                   ,'quad_rotation'
                                   ,'quad_tilt'
                                   ,'common_mode'
                                   ,'pedestals'
                                   ,'filter'
                                   ,'pixel_status'
                                   ]

        self.loadCalibParsDefault()

#---------------------

    def loadCalibParsDefault (self) :

        self.defpars = {}

        self.defpars['center'] = np.array(
                    [[[198.,  198.,  310.,   98.,  627.,  628.,  711.,  498.],
                      [198.,  198.,  310.,   98.,  627.,  628.,  711.,  498.],
                      [198.,  198.,  310.,   98.,  627.,  628.,  711.,  498.],
                      [198.,  198.,  310.,   98.,  627.,  628.,  711.,  498.]],
        
                     [[307.,   95.,  625.,  625.,  515.,  727.,  198.,  199.],
                      [307.,   95.,  625.,  625.,  515.,  727.,  198.,  199.],
                      [307.,   95.,  625.,  625.,  515.,  727.,  198.,  199.],
                      [307.,   95.,  625.,  625.,  515.,  727.,  198.,  199.]],
       
                     [[  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]]])

        self.defpars['center_corr'] = np.array(
                    [[[  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]],
                     [[  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]],
                     [[  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                      [  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]]])

        self.defpars['marg_gap_shift'] = np.array(
                    [[ 15.,  40.,   0.,  38.],
                     [ 15.,  40.,   0.,  38.],
                     [  0.,   0.,   0.,   0.]])

        self.defpars['offset'] = np.array(
                    [[   0.,    0.,  834.,  834.],
                     [   0.,  834.,  834.,    0.],
                     [   0.,    0.,    0.,    0.]])

        self.defpars['offset_corr'] = np.array(
                    [[   0.,    0.,    0.,    0.],
                     [   0.,    0.,    0.,    0.],
                     [   0.,    0.,    0.,    0.]])

        self.defpars['rotation'] = np.array(
                    [[   0.,    0.,  270.,  270.,  180.,  180.,  270.,  270.],
                     [   0.,    0.,  270.,  270.,  180.,  180.,  270.,  270.],
                     [   0.,    0.,  270.,  270.,  180.,  180.,  270.,  270.],
                     [   0.,    0.,  270.,  270.,  180.,  180.,  270.,  270.]])

        self.defpars['tilt'] = np.array(
                    [[0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],  
                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],  
                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],  
                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

        self.defpars['quad_rotation'] = np.array([ 180.,   90.,    0.,  270.])

        self.defpars['quad_tilt']     = np.array([   0.,    0.,    0.,    0.])

        self.defpars['common_mode']   = np.array([   1, 100, 30])

        self.defpars['filter']        = np.array([   1, 100, 10])

        self.defpars['pedestals']     = np.zeros((5920, 388), dtype=np.float32) # SHAPE: (5920, 388)

        self.defpars['pixel_status']  = np.zeros((5920, 388), dtype=np.uint16) # SHAPE: (5920, 388)

#---------------------

    def printCalibParsDefault (self) :

        for type in self.list_of_clib_types :
            print '\nCalibration constants type "' + type + '"' # + '" with shape', self.cpars[type].shape
            print self.defpars[type]
            
#---------------------

    def printListOfCalibTypes (self) :
        print 'list_of_clib_types:', self.list_of_clib_types

#---------------------

    def getCalibParsDefault (self, type) :

        if type in self.list_of_clib_types :
            return self.defpars[type]
        else :
            print  'WARNING: THE REQUESTED TYPE OF CALIBRATION PARS "', type, \
                   '" IS NOT FOUND IN THE AVAILABLE LIST:\n', self.list_of_clib_types
            return None

#---------------------------------------

calibparsdefault = CalibParsDefault()

#----------------------------------------------
# In case someone decides to run this module --
#----------------------------------------------

def main() :

    calibparsdefault.printCalibParsDefault()
    calibparsdefault.printListOfCalibTypes()
    print 'End of test'

if __name__ == "__main__" :

    main()
    sys.exit ( 'End of job' )

#----------------------------------------------
