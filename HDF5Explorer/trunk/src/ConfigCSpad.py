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

        #self.setCSpadParametersV0001()
        self.setCSpadParametersV0002()
        #self.Print()
        self.run_start_seconds = 0


    def setTimeOfRunStart( self ) :
        self.t_sec_r0003 = int( time.mktime((2010, 11, 19, 16, 25, 00, 0, 0, 0)) ) # converts date-time to seconds
        self.t_sec_r0004 = int( time.mktime((2011,  6, 23,  8, 00, 00, 0, 0, 0)) ) # converts date-time to seconds
        self.t_sec_Infty = int( time.mktime((2100,  0,  0,  0, 00, 00, 0, 0, 0)) ) # converts date-time to seconds

        print 'Start time for runs for CSPad configuration'
        print 'self.t_sec_r0003 =', self.t_sec_r0003, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(self.t_sec_r0003))
        print 'self.t_sec_r0004 =', self.t_sec_r0004, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(self.t_sec_r0004))
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
        elif self.t_sec_r0004 < self.run_start_seconds and self.run_start_seconds < self.t_sec_Infty:
            self.setCSpadParametersV0002()        
            print 'set parameters for V0002'


#==========================================
#==========================================

    def setCSpadParametersV0002 ( self ) :
        """Configuration parameters based on 2011-06-20 before-run4 optical measurement"""

        #print 'setCSpadParametersV0002'

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

        self.quadXOffset = [ off+0-gapX+shiftX,  off+  0+0-gapX-shiftX,  off+834-2+gapX-shiftX,  off+834+0+gapX+shiftX]
        self.quadYOffset = [ off+3-gapY-shiftY,  off+834-1+gapY-shiftY,  off+834-5+gapY+shiftY,  off+  0+2-gapY+shiftY]


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

        # "0" version of coordinates:

        self.pairXInQaud = [ [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400] ]

        self.pairYInQaud = [ [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400] ]

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
        """Configuration parameters based on 2011-03-29 post-run3 optical measurement"""

        #print 'setCSpadParametersV0001'

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
