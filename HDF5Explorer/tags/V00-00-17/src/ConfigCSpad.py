#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigCSpad...
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
#import os
import time
import PyCSPadImage.CalibPars as calp

#---------------------
#  Class definition --
#---------------------

class ConfigCSpad ( object ) :
    """This class contains all configuration parameters"""

    def __init__ ( self ) :
        """Constructor"""

        self.calibpars = calp.CalibPars() # Sets default calibration parameters.
        self.setConfigParsDefault()

#==========================================

    def setConfigParsDefault( self ) :
        # Later we get this array dynamically from /Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV*/CxiDs1.0:Cspad.0/element
        self.quad_nums_in_event = [0, 1, 2, 3] # <- default values

        # Later we get this array dynamically from /Configure:0000/CsPad::ConfigV2/CxiDs1.0:Cspad.0/config
        self.indPairsInQuads = [[ 0,   1,   2,   3,   4,   5,   6,   7],
                                [ 8,   9,  10,  11,  12,  13,  14,  15],
                                [16,  17,  18,  19,  20,  21,  22,  23],
                                [24,  25,  26,  27,  28,  29,  30,  31]]

#==========================================

    def setCSpadParameters( self ) :

        print 'setCSpadParameters():Set parameters from calib dir',
        self.setCSpadParametersFromCalibDir() 

#==========================================

    def setCSpadParametersFromCalibDir ( self ) :
        """Set the calibration parameters from the official location
           The self.cspad_calib_dir, self.cspad_name, and self.run_num should be provided...
        """
        print 'setCSpadParameters(): calib dir :', self.cspad_calib_dir
        print 'setCSpadParameters(): CSPad name:', self.cspad_name
        print 'setCSpadParameters(): Run number:', self.run_num

        self.calibpars.setCalibParsForPath ( run = self.run_num, path = self.cspad_calib_dir )
        #self.calibpars.loadAllCalibPars () 


        #self.calibpars.printCalibPars()
        #self.calibpars.printCalibFiles ()
        #self.calibpars.printListOfCalibTypes()
        #print self.calibpars.getCalibPars('center',self.run_num) # depricated : self.calibpars.cpars['center']
        #print self.calibpars.getCalibPars('center_corr',self.run_num)
        #print self.calibpars.getCalibPars('offset',self.run_num)
        #print self.calibpars.getCalibPars('offset_corr',self.run_num)
        #print self.calibpars.getCalibPars('marg_gap_shift',self.run_num)
        #print self.calibpars.getCalibPars('tilt',self.run_num)

        self.isCSPad2x2 = False

        # Detector and quar array dimennsions
        self.detDimX = 1765
        self.detDimY = 1765

        self.quadDimX = 850
        self.quadDimY = 850

        self.marg_gap_shift = self.calibpars.getCalibPars('marg_gap_shift',self.run_num) #  depricated : .cpars['marg_gap_shift']
        self.offset         = self.calibpars.getCalibPars('offset',self.run_num)         #  depricated : .cpars['offset']
        self.offset_corr    = self.calibpars.getCalibPars('offset_corr',self.run_num)    #  depricated : .cpars['offset_corr']
        #self.offset_corr    = [[ 0.,  5.,  4., -1.],
        #                       [ 0., -2., -3., -1.],
        #                       [ 0.,  0.,  0.,  0.]]

        dq                  = self.offset + self.offset_corr 

        #print 'self.marg_gap_shift =\n', self.marg_gap_shift 
        #print 'self.offset =\n',      self.offset
        #print 'self.offset_corr =\n', self.offset_corr 
        #print 'dq =\n', dq 

        self.preventiveRotationOffset = self.marg_gap_shift[0,0] # 15 # (pixel) increase effective canva for rotation

        offX   = self.marg_gap_shift[0,1] # 40
        offY   = self.marg_gap_shift[1,1] # 40

        gapX   = self.marg_gap_shift[0,2] # 0
        gapY   = self.marg_gap_shift[1,2] # 0

        shiftX = self.marg_gap_shift[0,3] # 28
        shiftY = self.marg_gap_shift[1,3] # 28

        #d0     = 834
        #self.quadXOffset = [ offX+ 0-gapX+shiftX,  offX+ 0+0-gapX-shiftX,  offX+d0-4+gapX-shiftX,  offX+d0-4+gapX+shiftX]
        #self.quadYOffset = [ offY+ 3-gapY-shiftY,  offY+d0-1+gapY-shiftY,  offY+d0-0+gapY+shiftY,  offY+ 0+8-gapY+shiftY]

        self.quadXOffset = [ offX-gapX+shiftX,  offX-gapX-shiftX,  offX+gapX-shiftX,  offX+gapX+shiftX]
        self.quadYOffset = [ offY-gapY-shiftY,  offY+gapY-shiftY,  offY+gapY+shiftY,  offY-gapY+shiftY]

        self.quadXOffset += dq[0] 
        self.quadYOffset += dq[1] 

        # Quad rotation angles
        self.quadInDetOrient = [ 180,   90,    0,  270]
        self.quadInDetOriInd = [   2,    1,    0,    3]

        # 2x1 section rotation angles
        self.pairInQaudOrient = [ [ 270, 270, 180, 180,  90,  90, 180, 180],
                                  [ 270, 270, 180, 180,  90,  90, 180, 180],
                                  [ 270, 270, 180, 180,  90,  90, 180, 180],
                                  [ 270, 270, 180, 180,  90,  90, 180, 180] ]

        # 2x1 section rotation index
        self.pairInQaudOriInd = [ [   3,   3,   2,   2,   1,   1,   2,   2],
                                  [   3,   3,   2,   2,   1,   1,   2,   2],
                                  [   3,   3,   2,   2,   1,   1,   2,   2],
                                  [   3,   3,   2,   2,   1,   1,   2,   2] ]

        #self.dPhi = [ [-0.33819,  0.00132,  0.31452, -0.03487,  0.14738,  0.07896, -0.21778, -0.10396],  
        #              [-0.27238, -0.00526,  0.02545,  0.03066, -0.03619,  0.02434,  0.08027,  0.15067],  
        #              [-0.04803, -0.00592,  0.11318, -0.07896, -0.36125, -0.31846, -0.16527,  0.09200],  
        #              [ 0.12436,  0.00263,  0.44809,  0.25794, -0.18029, -0.00117,  0.32701,  0.32439] ]
        self.dPhi = self.calibpars.getCalibPars('tilt',self.run_num) # depricated : .cpars['tilt']

        #self.pairXInQaud = [[199.14,  198.05,  310.67,   98.22,  629.71,  629.68,  711.87,  499.32],
        #                    [198.52,  198.08,  311.50,   98.69,  627.27,  627.27,  712.35,  499.77],
        #                    [198.32,  198.04,  310.53,   97.43,  626.68,  628.45,  710.86,  498.01],
        #                    [198.26,  198.04,  308.70,   96.42,  627.66,  628.04,  711.12,  498.25]]
        
        #self.pairYInQaud = [[308.25,   95.11,  625.60,  625.70,  515.02,  727.37,  198.53,  199.30],
        #                    [307.18,   95.08,  622.98,  623.51,  514.99,  727.35,  199.27,  198.94],
        #                    [307.68,   95.09,  623.95,  625.29,  512.32,  724.63,  198.04,  200.35],
        #                    [307.39,   95.12,  627.57,  626.65,  518.03,  730.95,  200.02,  199.70]]
        
        #self.pairZInQaud = [[  0.31,    0.12,    0.05,    0.12,    0.28,    0.24,    0.40,    0.27],
        #                    [  0.45,    0.36,    0.62,    0.33,    1.02,    0.92,    1.30,    1.07],
        #                    [  0.23,    0.22,    0.11,    0.15,    0.24,    0.20,    0.60,    0.42],
        #                    [  0.25,    0.21,    0.12,    0.10,    0.35,    0.28,    0.66,    0.40]]

        self.pairXInQaud = self.calibpars.getCalibPars('center',self.run_num)[0] # depricated : .cpars['center'][0]
        self.pairYInQaud = self.calibpars.getCalibPars('center',self.run_num)[1] # depricated : .cpars['center'][1]
        self.pairZInQaud = self.calibpars.getCalibPars('center',self.run_num)[1] # depricated : .cpars['center'][2]

                             #   0    1    2    3    4    5    6    7
        #self.dXInQaud    = [[   0,   0,   0,   1,   1,   0,   1,   0],
        #                    [   0,   0,   0,   0,   0,   0,  -1,   0],
        #                    [   0,   0,   0,   0,   0,   0,   0,   0],
        #                    [   0,   0,   0,  -1,   0,   0,   0,   1]]
                                                                   
        #self.dYInQaud    = [[   0,   0,   0,   0,  -1,   0,  -1,   0],
        #                    [   0,   0,   0,   0,   0,   1,   0,   0],
        #                    [   0,   0,   0,   0,   0,   0,  -1,  -2],
        #                    [   0,   0,   0,   0,   0,   0,   0,   0]]
        self.dXInQaud    = self.calibpars.getCalibPars('center_corr',self.run_num)[0] # depricated : .cpars['center_corr'][0]
        self.dYInQaud    = self.calibpars.getCalibPars('center_corr',self.run_num)[1] # depricated : .cpars['center_corr'][1]


#==========================================
#==========================================

    def Print( self ) :
        """Print CSpad configuration parameters"""

        print 'pairInQaudOrient =\n',  self.pairInQaudOrient
        print 'dPhi =\n',              self.dPhi
        print 'pairXInQaud =\n',       self.pairXInQaud       
        print 'pairYInQaud =\n',       self.pairYInQaud       
        print 'firstPairInQuad =\n',   self.firstPairInQuad       
        print 'lastPairInQuad =\n',    self.lastPairInQuad       

#---------------------------------------
# Makes a single object of this class --
#---------------------------------------

confcspad = ConfigCSpad()

#----------------------------------------------
# In case someone decides to run this module --
#----------------------------------------------
if __name__ == "__main__" :

    sys.exit ( "Module is not supposed to be run as main module" )

#----------------------------------------------
