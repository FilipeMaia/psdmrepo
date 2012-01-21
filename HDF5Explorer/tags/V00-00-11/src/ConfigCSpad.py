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

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor"""

        self.setTimeOfRunStart()

        #self.setCSpadParametersV0001() #DS1
        #self.setCSpadParametersV0002() #DS1
        #self.setCSpadParametersV0003() #CSPad for xpp36211
        self.setCSpadParametersV0004() #DS1
        #self.setCSpadParametersV0005() #DSD
        #self.Print()
        self.run_start_seconds = 0


    def setTimeOfRunStart( self ) :
        self.t_sec_r0003 = int( time.mktime((2010, 11, 19, 16, 25, 00, 0, 0, 0)) ) # converts date-time to seconds
        self.t_sec_r0004 = int( time.mktime((2011,  6, 23,  8, 00, 00, 0, 0, 0)) ) # converts date-time to seconds
        self.t_sec_r0005 = int( time.mktime((2011,  9,  1,  0, 00, 00, 0, 0, 0)) ) # converts date-time to seconds
        self.t_sec_r0006 = int( time.mktime((2012,  1, 14,  0, 00, 00, 0, 0, 0)) ) # runs after winter break
        self.t_sec_Infty = int( time.mktime((2100,  1,  1,  0, 00, 00, 0, 0, 0)) ) # converts date-time to seconds

        print 'Start time for runs for CSPad configuration'
        print 'self.t_sec_r0003 =', self.t_sec_r0003, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(self.t_sec_r0003))
        print 'self.t_sec_r0004 =', self.t_sec_r0004, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(self.t_sec_r0004))
        print 'self.t_sec_r0005 =', self.t_sec_r0005, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(self.t_sec_r0005))
        print 'self.t_sec_r0006 =', self.t_sec_r0006, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(self.t_sec_r0006))
        print 'self.t_sec_Infty =', self.t_sec_Infty, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(self.t_sec_Infty))

       #tloc = time.localtime(start_seconds) # converts sec to the tuple struct_time in local
       #print 'Local start time :', time.strftime('%Y-%m-%d %H:%M:%S',tloc)

#==========================================
#==========================================

    def setCSpadParameters( self ) :

        print 'setCSpadParameters():',

        if self.run_start_seconds < self.t_sec_r0004 :
            self.setCSpadParametersV0001()
            print 'set parameters for V0001'

        elif self.t_sec_r0004 < self.run_start_seconds and self.run_start_seconds < self.t_sec_r0005:
            self.setCSpadParametersV0002()        
            print 'set parameters for V0002'

        elif self.t_sec_r0005 < self.run_start_seconds and self.run_start_seconds < self.t_sec_r0006:
            self.setCSpadParametersV0003()        
            print 'set parameters for V0003'

        elif self.t_sec_r0006 < self.run_start_seconds and self.run_start_seconds < self.t_sec_Infty:

            #if   self.cspad_name == 'CxiDs1.0:Cspad.0' :
            #    self.setCSpadParametersV0004() # For DS1       
            #    print 'set parameters for V0004'

            if   self.cspad_name == 'CxiDsd.0:Cspad.0' :
                 self.setCSpadParametersV0005() # For DSD
                 print 'set parameters for V0005'

            else : 
                 self.setCSpadParametersFromCalibDir() 
                 print 'set parameters for from calib dir'

#==========================================
#==========================================

    def setCSpadParametersFromCalibDir ( self ) :
        """Set the calibration parameters from the official location
           The name of self.cspad_calib_dir should be provided...
           Use V0004 as a default for fallback...
        """
        print 'setCSpadParameters(): calib dir :', self.cspad_calib_dir
        print 'setCSpadParameters(): CSPad name:', self.cspad_name
        print 'setCSpadParameters(): Run number:', self.run_num

        calp.calibpars.setCalibParsForPath ( run = self.run_num, path = self.cspad_calib_dir )
        #calp.calibpars.printCalibPars()
        #calp.calibpars.printCalibFiles ()
        #calp.calibpars.printListOfCalibTypes()
        #print calp.calibpars.cpars['center']
        #print calp.calibpars.cpars['center_corr']
        #print calp.calibpars.cpars['offset']
        #print calp.calibpars.cpars['offset_corr']
        #print calp.calibpars.cpars['marg_gap_shift']
        #print calp.calibpars.cpars['tilt']

        self.isCSPad2x2 = False

        # Detector and quar array dimennsions
        self.detDimX = 1765
        self.detDimY = 1765

        self.quadDimX = 850
        self.quadDimY = 850

        self.marg_gap_shift = calp.calibpars.cpars['marg_gap_shift']
        self.offset         = calp.calibpars.cpars['offset']
        self.offset_corr    = calp.calibpars.cpars['offset_corr']
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

        # We get this array dynamically from /Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV*/CxiDs1.0:Cspad.0/element
        self.quad_nums_in_event = [0, 1, 2, 3] # <- default values

        # Quad rotation angles
        self.quadInDetOrient = [ 180,   90,    0,  270]
        self.quadInDetOriInd = [   2,    1,    0,    3]

        # We get this array dynamically from /Configure:0000/CsPad::ConfigV2/CxiDs1.0:Cspad.0/config
        self.indPairsInQuads = [[ 0,   1,   2,   3,   4,   5,   6,   7],
                                [ 8,   9,  10,  11,  12,  13,  14,  15],
                                [16,  17,  18,  19,  20,  21,  22,  23],
                                [24,  25,  26,  27,  28,  29,  30,  31]]

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
        self.dPhi = calp.calibpars.cpars['tilt']

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

        self.pairXInQaud = calp.calibpars.cpars['center'][0]
        self.pairYInQaud = calp.calibpars.cpars['center'][1]
        self.pairZInQaud = calp.calibpars.cpars['center'][2]

                             #   0    1    2    3    4    5    6    7
        #self.dXInQaud    = [[   0,   0,   0,   1,   1,   0,   1,   0],
        #                    [   0,   0,   0,   0,   0,   0,  -1,   0],
        #                    [   0,   0,   0,   0,   0,   0,   0,   0],
        #                    [   0,   0,   0,  -1,   0,   0,   0,   1]]
                                                                   
        #self.dYInQaud    = [[   0,   0,   0,   0,  -1,   0,  -1,   0],
        #                    [   0,   0,   0,   0,   0,   1,   0,   0],
        #                    [   0,   0,   0,   0,   0,   0,  -1,  -2],
        #                    [   0,   0,   0,   0,   0,   0,   0,   0]]
        self.dXInQaud    = calp.calibpars.cpars['center_corr'][0]
        self.dYInQaud    = calp.calibpars.cpars['center_corr'][1]


#==========================================
#==========================================

    def setCSpadParametersV0005 ( self ) :
        """DSD Configuration parameters based on 2012-01-12-Metrology optical measurement"""

        self.isCSPad2x2 = False

        # Detector and quar array dimennsions
        self.detDimX = 1765
        self.detDimY = 1765

        self.quadDimX = 850
        self.quadDimY = 850

        # Quad rotation angles
        self.quadInDetOrient = [ 180,   90,    0,  270]
        self.quadInDetOriInd = [   2,    1,    0,    3]

        self.preventiveRotationOffset = 15 # (pixel) increase effective canva for rotation
        off = 40

        gapX = 0
        gapY = 0

        shiftX = 24
        shiftY = 24

        d0     = 834

        self.quadXOffset = [ off+ 1-gapX+shiftX,  off+ 0-3-gapX-shiftX,  off+d0+0+gapX-shiftX,  off+d0+1+gapX+shiftX]
        self.quadYOffset = [ off+ 1-gapY-shiftY,  off+d0+1+gapY-shiftY,  off+d0-1+gapY+shiftY,  off+ 0-1-gapY+shiftY]


        # We get this array dynamically from /Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV*/CxiDs1.0:Cspad.0/element
        self.quad_nums_in_event = [0, 1, 2, 3] # <- default values

        # We get this array dynamically from /Configure:0000/CsPad::ConfigV2/CxiDs1.0:Cspad.0/config
        self.indPairsInQuads = [[ 0,   1,   2,   3,   4,   5,   6,   7],
                                [ 8,   9,  10,  11,  12,  13,  14,  15],
                                [16,  17,  18,  19,  20,  21,  22,  23],
                                [24,  25,  26,  27,  28,  29,  30,  31]]

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

        # 2x1 section tilt angles
        self.dPhi = [ [-0.02500,  0.00329, -0.16977, -0.13423, -0.32451, -0.25566,  0.02698,  0.31916],  
                      [-0.02369,  0.00461,  0.24123,  0.06686, -0.22574, -0.26358, -0.16712,  0.16513],  
                      [-0.03159,  0.01447,  0.00526,  0.33226, -0.17809,  0.10962, -0.26712, -0.19214],  
                      [-0.12963,  0.00526, -0.23951, -0.22634,  0.23030,  0.08285,  0.05132,  0.06119] ]

        # 2x1 section center coordinates
        self.pairXInQaud = [[ 198.46,  198.05,  311.84,   99.32,  628.17,  629.59,  711.73,  499.58 ],
                            [ 198.24,  198.03,  311.37,   98.35,  625.63,  626.78,  711.25,  498.41 ],
                            [ 198.16,  198.04,  309.53,   96.76,  626.36,  626.53,  711.73,  499.20 ],
                            [ 198.36,  198.02,  309.53,   97.25,  627.67,  627.51,  711.80,  499.15 ]]
                                                                                                   
        self.pairYInQaud = [[ 307.49,   95.11,  626.06,  627.17,  513.15,  725.70,  198.75,  198.39 ],
                            [ 307.48,   95.10,  625.67,  625.63,  510.49,  723.25,  196.42,  196.40 ],
                            [ 307.77,   95.14,  625.26,  624.72,  513.51,  725.92,  199.69,  200.90 ],
                            [ 307.76,   95.09,  627.28,  628.36,  512.75,  725.08,  199.95,  199.96 ]]
                                                                                                   
        self.pairZInQaud = [[   0.19,    0.22,    0.42,    0.47,    0.25,    0.48,    0.00,    0.17 ],
                            [   0.42,    0.27,    0.57,    0.65,    0.55,    0.82,    0.32,    0.38 ],
                            [   0.05,   -0.17,    0.70,    0.50,    0.63,    1.11,    0.20,   -0.06 ],
                            [  -0.01,    0.00,   -0.12,   -0.15,   -0.08,   -0.09,    0.06,    0.03 ]]

        # 2x1 section center coordinate corrections
                            #   0    1    2    3    4    5    6    7
        self.dXInQaud    = [[   0,   0,   0,   0,   0,   0,   0,   0], 
                            [   0,   0,   0,   0,   0,   0,   0,   0], 
                            [   0,   0,   0,   0,   0,   0,   0,   0], 
                            [   0,   0,   0,   0,   0,   0,   0,   0]] 
                                                                   
        self.dYInQaud    = [[   0,   0,   0,   0,   0,   0,   0,   0], 
                            [   0,   0,   0,   0,   0,   0,   0,   0], 
                            [   0,   0,   0,   0,   0,   0,   0,   0], 
                            [   0,   0,   0,   0,   0,   0,   0,   0]] 

#==========================================
#==========================================

    def setCSpadParametersV0004 ( self ) :
        """DS1 Configuration parameters based on 2011-06-20-Metrology optical measurement

           Comparing to V0002 the shiftX, shiftY are changed
        """

        self.isCSPad2x2 = False

        # Detector and quar array dimennsions
        self.detDimX = 1765
        self.detDimY = 1765

        self.quadDimX = 850
        self.quadDimY = 850

        # Quad rotation angles
        self.quadInDetOrient = [ 180,   90,    0,  270]
        self.quadInDetOriInd = [   2,    1,    0,    3]

        self.preventiveRotationOffset = 15 # (pixel) increase effective canva for rotation
        off = 40

        gapX = 0
        gapY = 0

        shiftX = 28
        shiftY = 28

        d0     = 834

        self.quadXOffset = [ off+ 0-gapX+shiftX,  off+ 0+0-gapX-shiftX,  off+d0-4+gapX-shiftX,  off+d0-4+gapX+shiftX]
        self.quadYOffset = [ off+ 3-gapY-shiftY,  off+d0-1+gapY-shiftY,  off+d0-0+gapY+shiftY,  off+ 0+8-gapY+shiftY]

        # We get this array dynamically from /Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV*/CxiDs1.0:Cspad.0/element
        self.quad_nums_in_event = [0, 1, 2, 3] # <- default values

        # We get this array dynamically from /Configure:0000/CsPad::ConfigV2/CxiDs1.0:Cspad.0/config
        self.indPairsInQuads = [[ 0,   1,   2,   3,   4,   5,   6,   7],
                                [ 8,   9,  10,  11,  12,  13,  14,  15],
                                [16,  17,  18,  19,  20,  21,  22,  23],
                                [24,  25,  26,  27,  28,  29,  30,  31]]

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

        self.dPhi = [ [-0.33819,  0.00132,  0.31452, -0.03487,  0.14738,  0.07896, -0.21778, -0.10396],  
                      [-0.27238, -0.00526,  0.02545,  0.03066, -0.03619,  0.02434,  0.08027,  0.15067],  
                      [-0.04803, -0.00592,  0.11318, -0.07896, -0.36125, -0.31846, -0.16527,  0.09200],  
                      [ 0.12436,  0.00263,  0.44809,  0.25794, -0.18029, -0.00117,  0.32701,  0.32439] ]

        self.pairXInQaud = [[199.14,  198.05,  310.67,   98.22,  629.71,  629.68,  711.87,  499.32],
                            [198.52,  198.08,  311.50,   98.69,  627.27,  627.27,  712.35,  499.77],
                            [198.32,  198.04,  310.53,   97.43,  626.68,  628.45,  710.86,  498.01],
                            [198.26,  198.04,  308.70,   96.42,  627.66,  628.04,  711.12,  498.25]]
        
        self.pairYInQaud = [[308.25,   95.11,  625.60,  625.70,  515.02,  727.37,  198.53,  199.30],
                            [307.18,   95.08,  622.98,  623.51,  514.99,  727.35,  199.27,  198.94],
                            [307.68,   95.09,  623.95,  625.29,  512.32,  724.63,  198.04,  200.35],
                            [307.39,   95.12,  627.57,  626.65,  518.03,  730.95,  200.02,  199.70]]
        
        self.pairZInQaud = [[  0.31,    0.12,    0.05,    0.12,    0.28,    0.24,    0.40,    0.27],
                            [  0.45,    0.36,    0.62,    0.33,    1.02,    0.92,    1.30,    1.07],
                            [  0.23,    0.22,    0.11,    0.15,    0.24,    0.20,    0.60,    0.42],
                            [  0.25,    0.21,    0.12,    0.10,    0.35,    0.28,    0.66,    0.40]]

                            #   0    1    2    3    4    5    6    7
        self.dXInQaud    = [[   0,   0,   0,   1,   1,   0,   1,   0], #DONE
                            [   0,   0,   0,   0,   0,   0,  -1,   0], #DONE
                            [   0,   0,   0,   0,   0,   0,   0,   0], #DONE
                            [   0,   0,   0,  -1,   0,   0,   0,   1]] #DONE
                                                                   
        self.dYInQaud    = [[   0,   0,   0,   0,  -1,   0,  -1,   0], #DONE
                            [   0,   0,   0,   0,   0,   1,   0,   0], #DONE
                            [   0,   0,   0,   0,   0,   0,  -1,  -2], #DONE
                            [   0,   0,   0,   0,   0,   0,   0,   0]] #DONE
        
#==========================================
#==========================================

    def setCSpadParametersV0003 ( self ) :
        """CSPad for XPP Configuration parameters based on 2011-08-10-Metrology optical measurement"""

        #print 'setCSpadParametersV0003'

        self.isCSPad2x2 = False

        # Detector and quar array dimennsions
        self.detDimX = 1765
        self.detDimY = 1765

        self.quadDimX = 850
        self.quadDimY = 850

        # Quad rotation angles
        self.quadInDetOrient = [ 180,   90,    0,  270]
        self.quadInDetOriInd = [   2,    1,    0,    3]

        self.preventiveRotationOffset = 15 # (pixel) increase effective canva for rotation
        off = 40

        gapX = 0
        gapY = 0

        shiftX = 18
        shiftY = 18

        d0     = 834

        #self.quadXOffset = [ off+0-gapX+shiftX,  off+  0+1-gapX-shiftX,  off+834+0+gapX-shiftX,  off+834+0+gapX+shiftX]
        #self.quadYOffset = [ off+0-gapY-shiftY,  off+834-3+gapY-shiftY,  off+834-0+gapY+shiftY,  off+  0+2-gapY+shiftY]

        #self.quadXOffset = [ off+0-gapX+shiftX,  off+  0+0-gapX-shiftX,  off+834-2+gapX-shiftX,  off+834+0+gapX+shiftX]
        #self.quadYOffset = [ off+3-gapY-shiftY,  off+834-1+gapY-shiftY,  off+834-5+gapY+shiftY,  off+  0+2-gapY+shiftY]

        self.quadXOffset = [ off+ 0-gapX+shiftX,  off+ 0+0-gapX-shiftX,  off+d0-0+gapX-shiftX,  off+d0+0+gapX+shiftX]
        self.quadYOffset = [ off+ 0-gapY-shiftY,  off+d0-0+gapY-shiftY,  off+d0-0+gapY+shiftY,  off+ 0+0-gapY+shiftY]


        # We get this array dynamically from /Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV*/CxiDs1.0:Cspad.0/element
        self.quad_nums_in_event = [0, 1, 2, 3] # <- default values

        # We get this array dynamically from /Configure:0000/CsPad::ConfigV2/CxiDs1.0:Cspad.0/config
        self.indPairsInQuads = [[ 0,   1,   2,   3,   4,   5,   6,   7],
                                [ 8,   9,  10,  11,  12,  13,  14,  15],
                                [16,  17,  18,  19,  20,  21,  22,  23],
                                [24,  25,  26,  27,  28,  29,  30,  31]]

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

        # 2x1 section tilt angles
        self.dPhi = [ [-0.06186, -0.00526,  0.17107,  0.16384, -0.02763,  0.38428,  0.04672,  0.11581],  
                      [-0.06251, -0.00066, -0.06843,  0.01382, -0.15199, -0.09540,  0.05330, -0.07172],  
                      [-0.26781, -0.00263, -0.28227, -0.12172, -0.24212,  0.00000, -0.11382, -0.13160],  
                      [-0.21897,  0.00132,  0.00526, -0.08423, -0.31617,  0.00395, -0.06251,  0.02106] ]

        # 2x1 section center coordinates
        self.pairXInQaud = [[ 198.48,  198.05,  307.91,   95.69,  625.60,  624.69,  709.79,  497.97],
                            [ 198.36,  198.05,  310.89,   98.49,  627.36,  627.76,  712.15,  498.90],
                            [ 198.78,  198.04,  310.98,   97.86,  627.09,  627.61,  713.33,  500.94],
                            [ 198.90,  198.05,  309.48,   96.66,  626.23,  626.67,  712.47,  499.68]]
        
        self.pairYInQaud = [[ 306.92,   95.08,  625.56,  625.52,  516.16,  729.09,  200.58,  201.70],
                            [ 307.40,   95.09,  624.41,  624.85,  519.48,  731.87,  204.71,  205.30],
                            [ 307.61,   95.10,  625.99,  626.70,  514.76,  727.56,  200.39,  201.28],
                            [ 308.23,   95.08,  624.73,  625.09,  513.34,  726.52,  196.72,  196.82]]
        
        self.pairZInQaud = [[   0.33,    0.28,    0.21,    0.08,    0.43,    0.43,    0.54,    0.48],
                            [  -0.68,   -0.42,   -1.15,   -0.87,   -1.63,   -1.86,   -1.53,   -1.07],
                            [  -0.37,   -0.01,   -0.50,   -0.33,   -0.89,   -0.95,   -1.01,   -0.77],
                            [  -0.46,   -0.37,   -0.68,   -0.41,   -1.16,   -1.25,   -1.11,   -1.79]]

        # 2x1 section center coordinate corrections
                            #   0    1    2    3    4    5    6    7
        self.dXInQaud    = [[   0,   0,   0,   0,   0,   0,   0,   0], 
                            [   0,   0,   0,   0,   0,   0,   0,   0], 
                            [   0,   0,   0,   0,   0,   0,   0,   0], 
                            [   0,   0,   0,   0,   0,   0,   0,   0]] 
                                                                   
        self.dYInQaud    = [[   0,   0,   0,   0,   0,   0,   0,   0], 
                            [   0,   0,   0,   0,  -7,  -7, -10, -10], 
                            [   0,   0,   0,   0,   0,   0,   0,   0], 
                            [   0,   0,   0,   0,   0,   0,   0,   2]] 

#==========================================
#==========================================

    def setCSpadParametersV0002 ( self ) :
        """DS1 Configuration parameters based on 2011-06-20 before-run4 optical measurement"""

        #print 'setCSpadParametersV0002'

        self.isCSPad2x2 = False

        # Detector and quar array dimennsions
        self.detDimX = 1765
        self.detDimY = 1765

        self.quadDimX = 850
        self.quadDimY = 850

        # Quad orientation
        self.quadInDetOrient = [ 180,   90,    0,  270]
        self.quadInDetOriInd = [   2,    1,    0,    3]

        self.preventiveRotationOffset = 15 # (pixel) increase effective canva for rotation
        off = 40

        gapX = 0
        gapY = 0

        shiftX = 38
        shiftY = 38

        #self.quadXOffset = [ off+0-gapX+shiftX,  off+  0+1-gapX-shiftX,  off+834+0+gapX-shiftX,  off+834+0+gapX+shiftX]
        #self.quadYOffset = [ off+0-gapY-shiftY,  off+834-3+gapY-shiftY,  off+834-0+gapY+shiftY,  off+  0+2-gapY+shiftY]

        #self.quadXOffset = [ off+0-gapX+shiftX,  off+  0+0-gapX-shiftX,  off+834-2+gapX-shiftX,  off+834+0+gapX+shiftX]
        #self.quadYOffset = [ off+3-gapY-shiftY,  off+834-1+gapY-shiftY,  off+834-5+gapY+shiftY,  off+  0+2-gapY+shiftY]

        self.quadXOffset = [ off+0-gapX+shiftX,  off+  0+0-gapX-shiftX,  off+834-6+gapX-shiftX,  off+834-4+gapX+shiftX]
        self.quadYOffset = [ off+3-gapY-shiftY,  off+834-1+gapY-shiftY,  off+834+0+gapY+shiftY,  off+  0+8-gapY+shiftY]


        # We get this array dynamically from /Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV*/CxiDs1.0:Cspad.0/element
        self.quad_nums_in_event = [0, 1, 2, 3] # <- default values

        # We get this array dynamically from /Configure:0000/CsPad::ConfigV2/CxiDs1.0:Cspad.0/config
        self.indPairsInQuads = [[ 0,   1,   2,   3,   4,   5,   6,   7],
                                [ 8,   9,  10,  11,  12,  13,  14,  15],
                                [16,  17,  18,  19,  20,  21,  22,  23],
                                [24,  25,  26,  27,  28,  29,  30,  31]]

        # Section orientation
        self.pairInQaudOrient = [ [ 270, 270, 180, 180,  90,  90, 180, 180],
                                  [ 270, 270, 180, 180,  90,  90, 180, 180],
                                  [ 270, 270, 180, 180,  90,  90, 180, 180],
                                  [ 270, 270, 180, 180,  90,  90, 180, 180] ]

        self.pairInQaudOriInd = [ [   3,   3,   2,   2,   1,   1,   2,   2],
                                  [   3,   3,   2,   2,   1,   1,   2,   2],
                                  [   3,   3,   2,   2,   1,   1,   2,   2],
                                  [   3,   3,   2,   2,   1,   1,   2,   2] ]

        self.dPhi = [ [-0.33819,  0.00132,  0.31452, -0.03487,  0.14738,  0.07896, -0.21778, -0.10396],  
                      [-0.27238, -0.00526,  0.02545,  0.03066, -0.03619,  0.02434,  0.08027,  0.15067],  
                      [-0.04803, -0.00592,  0.11318, -0.07896, -0.36125, -0.31846, -0.16527,  0.09200],  
                      [ 0.12436,  0.00263,  0.44809,  0.25794, -0.18029, -0.00117,  0.32701,  0.32439] ]

        self.pairXInQaud = [[199.14,  198.05,  310.67,   98.22,  629.71,  629.68,  711.87,  499.32],
                            [198.52,  198.08,  311.50,   98.69,  627.27,  627.27,  712.35,  499.77],
                            [198.32,  198.04,  310.53,   97.43,  626.68,  628.45,  710.86,  498.01],
                            [198.26,  198.04,  308.70,   96.42,  627.66,  628.04,  711.12,  498.25]]
        
        self.pairYInQaud = [[308.25,   95.11,  625.60,  625.70,  515.02,  727.37,  198.53,  199.30],
                            [307.18,   95.08,  622.98,  623.51,  514.99,  727.35,  199.27,  198.94],
                            [307.68,   95.09,  623.95,  625.29,  512.32,  724.63,  198.04,  200.35],
                            [307.39,   95.12,  627.57,  626.65,  518.03,  730.95,  200.02,  199.70]]
        
        self.pairZInQaud = [[  0.31,    0.12,    0.05,    0.12,    0.28,    0.24,    0.40,    0.27],
                            [  0.45,    0.36,    0.62,    0.33,    1.02,    0.92,    1.30,    1.07],
                            [  0.23,    0.22,    0.11,    0.15,    0.24,    0.20,    0.60,    0.42],
                            [  0.25,    0.21,    0.12,    0.10,    0.35,    0.28,    0.66,    0.40]]

                            #   0    1    2    3    4    5    6    7
        self.dXInQaud    = [[   0,   0,   0,   1,   1,   0,   1,   0], #DONE
                            [   0,   0,   0,   0,   0,   0,  -1,   0], #DONE
                            [   0,   0,   0,   0,   0,   0,   0,   0], #DONE
                            [   0,   0,   0,  -1,   0,   0,   0,   1]] #DONE
                                                                   
        self.dYInQaud    = [[   0,   0,   0,   0,  -1,   0,  -1,   0], #DONE
                            [   0,   0,   0,   0,   0,   1,   0,   0], #DONE
                            [   0,   0,   0,   0,   0,   0,  -1,  -2], #DONE
                            [   0,   0,   0,   0,   0,   0,   0,   0]] #DONE

#==========================================
#==========================================

    def setCSpadParametersV0001 ( self ) :
        """DS1 Configuration parameters based on 2011-03-29 post-run3 optical measurement"""

        #print 'setCSpadParametersV0001'

        self.isCSPad2x2 = False

        # Detector and quar array dimennsions
        self.detDimX = 1750
        self.detDimY = 1750

        self.quadDimX = 850
        self.quadDimY = 850


        # Quad orientation
        self.quadInDetOrient = [ 180,   90,    0,  270]
        self.quadInDetOriInd = [   2,    1,    0,    3]

        self.preventiveRotationOffset = 15 # (pixel) increase effective canva for rotation
        off = 30

        gapX = 0
        gapY = 0

        shiftX = 18
        shiftY = 18

        self.quadXOffset = [ off+0-gapX+shiftX,  off+  0+1-gapX-shiftX,  off+834+0+gapX-shiftX,  off+834+0+gapX+shiftX]
        self.quadYOffset = [ off+0-gapY-shiftY,  off+834-3+gapY-shiftY,  off+834-0+gapY+shiftY,  off+  0+2-gapY+shiftY]


        # We get this array dynamically from /Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV*/CxiDs1.0:Cspad.0/element
        self.quad_nums_in_event = [0, 1, 2, 3] # <- default values

        # We get this array dynamically from /Configure:0000/CsPad::ConfigV2/CxiDs1.0:Cspad.0/config
        self.indPairsInQuads = [[ 0,   1,   2,   3,   4,   5,   6,   7],
                                [ 8,   9,  10,  11,  12,  13,  14,  15],
                                [16,  17,  18,  19,  20,  21,  22,  23],
                                [24,  25,  26,  27,  28,  29,  30,  31]]

        # Section orientation
        self.pairInQaudOrient = [ [ 270, 270, 180, 180,  90,  90, 180, 180],
                                  [ 270, 270, 180, 180,  90,  90, 180, 180],
                                  [ 270, 270, 180, 180,  90,  90, 180, 180],
                                  [ 270, 270, 180, 180,  90,  90, 180, 180] ]

        self.pairInQaudOriInd = [ [   3,   3,   2,   2,   1,   1,   2,   2],
                                  [   3,   3,   2,   2,   1,   1,   2,   2],
                                  [   3,   3,   2,   2,   1,   1,   2,   2],
                                  [   3,   3,   2,   2,   1,   1,   2,   2] ]

        # 2011-03-29 post run3: Signs of angles for a half of sensors are corrected on 2011-05-23
        self.dPhi = [ [-0.27305, 0.01711,-0.34736,-0.08158,-0.15462,-0.12369, 0.09212, 0.39342],
                      [ 0.14215, 0.00395, 0.13488, 0.12106, 0.11221, 0.11025,-0.00921, 0.06316],
                      [-0.33008, 0.00196,-0.16524,-0.56502,-0.44282,-0.39677,-0.18349,-0.22444],
                      [-0.35074, 0.00131,-0.01513, 0.03882, 0.00000, 0.34609, 0.00000, 0.08816] ]

        # "0" version of coordinates:

        self.pairXInQaud = [ [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400] ]

        self.pairYInQaud = [ [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400] ]

        self.pairXInQaud = [[ 198.59,  198.04,  310.42,   98.22,  629.25,  630.01,  712.11,  499.91],  
                            [ 198.40,  198.13,  310.55,   97.68,  626.40,  626.59,  710.49,  498.15],  
                            [ 200.58,  199.79,  314.91,  103.43,  631.36,  633.34,  714.25,  501.44],  
                            [ 198.89,  198.18,  310.75,   98.13,  630.00,  629.41,  710.00,  499.99]] # 4,6 (630,710) were not measured  
                                                                                                    
        self.pairYInQaud = [[ 308.00,   95.24,  626.85,  627.63,  517.84,  730.54,  200.79,  200.22],  
                            [ 308.35,   95.09,  626.15,  626.58,  513.07,  725.86,  200.67,  200.57],  
                            [ 309.62,   97.24,  622.35,  625.76,  513.46,  725.96,  199.78,  199.76],  
                            [ 307.80,   95.08,  628.38,  628.43,  515.00,  730.70,  200.00,  202.96]] # 4,6 (515,200) were not measured  

        self.pairZInQaud = [[   0.37,    0.15,    0.52,    0.62,    0.39,    0.55,    0.24,    0.18],  
                            [   2.16,    1.08,    4.14,    3.51,    4.61,    5.68,    3.35,    2.63],  
                            [   0.28,    0.20,    0.50,    0.65,    0.39,    0.56,    0.30,    0.27],  
                            [   0.37,    0.30,    0.47,    0.25,    0.00,    1.06,    0.00,    0.85]]

                            #   0    1    2    3    4    5    6    7
        self.dXInQaud    = [[   0,   0,   0,   0,   0,   0,   0,   0], #*** DO NOT TOCH !
                            [   0,   0,   0,   0,  -1,   0,   0,   0], #*** DO NOT TOCH !
                            [   0,   0,  -1,  -1,   0,   0,   0,   0], #*** DO NOT TOCH !
                            [   0,   0,   0,   0,   0,   0,   0,   0]] #*** DO NOT TOCH ! 4,6 (630,710) were not measured
                                                                   
        self.dYInQaud    = [[   0,   0,   0,   0,   0,   0,-0.5,   0], #*** DO NOT TOCH !
                            [   0,   0,   0, 0.5,  -1,   0,   0,   1], #*** DO NOT TOCH !
                            [   1,   0,   1,  -2,   0,   0,  -5,  -4], #*** DO NOT TOCH !
                            [   0,   0,   0,   0,   0,  -1,   0,  -2]] #*** DO NOT TOCH ! 4,6 (515,200) were not measured

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
