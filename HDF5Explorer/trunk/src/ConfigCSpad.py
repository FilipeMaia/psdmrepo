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
        self.setCSpadParametersV0001()
        #self.Print()

    def setCSpadParametersV0001 ( self ) :
        """Set default configuration parameters hardwired in this module"""

        print 'setCSpadParameters'

        self.preventiveRotationOffset = 5 # (pixel) increase effective canva for rotation
        off2 = 2*self.preventiveRotationOffset

        self.quadInDetOrient = [   90,    0,   270,  180]
        self.quadInDetOriInd = [    1,    0,     3,    2]

        gapX = 30
        gapY = 40

        #self.quadXOffset     = [   3,    0,      800+gapX, 806+gapX]
        #self.quadYOffset     = [   0,  792+gapY, 803+gapY,       10]

        self.quadXOffset     = [   3+off2,    0,      800+gapX, 806+gapX+off2]
        self.quadYOffset     = [   0,  792+gapY, 803+gapY+off2,       10+off2]

        self.firstPairInQuad = [0, 0,  8, 16]
        self.lastPairInQuad  = [0, 8, 16, 20]

        # We get this array dynamically from /Configure:0000/CsPad::ConfigV2/CxiDs1.0:Cspad.0/config
        self.indPairsInQuads = [[-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
                                [ 0,   1,   2,   3,   4,   5,   6,   7],
                                [ 8,   9,  10,  11,  12,  13,  14,  15],
                                [16,  17,  -1,  -1,  -1,  -1,  18,  19]]

        self.pairInQaudOrient = [ [   0,   0, 270, 270, 180, 180, 270, 270],
                                  [   0,   0, 270, 270, 180, 180, 270, 270],
                                  [   0,   0, 270, 270, 180, 180, 270, 270],
                                  [   0,   0, 270, 270, 180, 180, 270, 270] ]
        #                         [ 180, 180, 270, 270,   0,   0, 270, 270] ]

        self.pairInQaudOriInd = [ [   0,   0,   3,   3,   2,   2,   3,   3],
                                  [   0,   0,   3,   3,   2,   2,   3,   3],
                                  [   0,   0,   3,   3,   2,   2,   3,   3],
                                  [   0,   0,   3,   3,   2,   2,   3,   3] ]
        #                         [   2,   2,   1,   1,   0,   3,   3,   2] ]


        # 2011-02-10 before run3:
        #self.dPhi = [ [0,0,0,0,0,0,0,0],
        #              [0.144, 0.466, -0.049, -0.186, -0.291, -0.338, 0.119, 0.135],
        #              [0,0,0,0,0,0,0,0],
        #              [0,0,0,0,0,0,0,0] ]

        # 2011-03-29 post run3: 
        self.dPhi = [ [-0.27305, 0.01711, 0.34736, 0.08158,-0.15462,-0.12369,-0.09212,-0.39342],
                      [ 0.14215, 0.00395,-0.13488,-0.12106, 0.11221, 0.11025, 0.00921,-0.06316],
                      [-0.33008, 0.00196, 0.16524, 0.56502,-0.44282,-0.39677, 0.18349, 0.22444],
                      [-0.35074, 0.00131, 0.01513,-0.03882, 0.00000, 0.34609, 0.00000,-0.08816] ]

        # "0" version of coordinates:

        self.pairXInQaud = [ [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400] ]

        self.pairYInQaud = [ [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400] ]


        # Optical alignment 2011-02-10 before run3:

        #self.pairXInQaud = [[ 419,  631,    0,    0,  208,    1,  423,  424],  # 2:5 were not measured
        #                    [ 421,  634,    0,    0,  213,    1,  424,  425],
        #                    [ 417,  630,    0,    1,  212,    0,  425,  426],
        #                    [ 421,  635,    0,    0,  213,    1,  425,  426]] # 2:5 were not measured 
                                                                      
        #self.pairYInQaud = [[   0,    0,  214,    1,  430,  430,  615,  402],  # 2:5 were not measured
        #                    [   0,    0,  214,    1,  425,  425,  615,  402],
        #                    [   0,    0,  215,    3,  431,  431,  616,  403],
        #                    [   0,    0,  214,    1,  425,  425,  615,  403]] # 2:5 were not measured

        # Optical alignment 2011-03-29 post run3:

        self.pairXInQaud = [[422,  635,    0,    0,  212,    0,  427,  428],
                            [421,  634,    0,    0,  214,    1,  425,  425],
                            [418,  631,    2,    0,  216,    3,  431,  430],
                            [422,  636,    0,    0,  215,    1,  430,  425]] # 4,6 (215,430) were not measured
                                                                     
        self.pairYInQaud = [[  1,    0,  216,    3,  431,  432,  616,  403],
                            [  0,    0,  214,    2,  426,  426,  615,  402],
                            [  1,    0,  219,    8,  433,  435,  617,  404],
                            [  1,    0,  215,    2,  430,  430,  615,  404]] # 4,6 (430,615) were not measured


        #for ix in range(8) : self.pairXInQaud.append(random.randint(0,600))
        #for iy in range(8) : self.pairYInQaud.append(random.randint(0,600))

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
