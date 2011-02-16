#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigCSpad...
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


        self.quadInDetOrient = [   90,    0,   270,  180]
        self.quadInDetOriInd = [    1,    0,     3,    2]

        gapX = 30
        gapY = 40

        self.quadXOffset     = [   3,    0,      800+gapX, 804+gapX]
        self.quadYOffset     = [   8,  800+gapY, 810+gapY,       18]

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


        self.dPhi = [ [0,0,0,0,0,0,0,0],
                      [0.144, 0.466, -0.049, -0.186, -0.291, -0.338, 0.119, 0.135],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0] ]

        # "0" version of coordinates:

        self.pairXInQaud = [ [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400] ]

        self.pairYInQaud = [ [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400] ]


        # Optical alignment:

        self.pairXInQaud = [[ 414,  626,    0,    0,  213,    1,  418,  419],  # 2:5 were not measured
                            [ 421,  634,    0,    0,  213,    1,  424,  425],
                            [ 417,  630,    0,    1,  212,    0,  425,  426],
                            [ 416,  630,    0,    0,  213,    1,  420,  421]] # 2:5 were not measured 
                                                                      
        self.pairYInQaud = [[   0,    0,  214,    1,  425,  425,  615,  402],  # 2:5 were not measured
                            [   0,    0,  214,    1,  425,  425,  615,  402],
                            [   0,    0,  215,    3,  431,  431,  616,  403],
                            [   0,    0,  214,    1,  425,  425,  615,  403]] # 2:5 were not measured


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
