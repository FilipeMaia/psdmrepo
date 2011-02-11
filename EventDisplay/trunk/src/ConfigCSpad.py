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
        self.Print()

    def setCSpadParametersV0001 ( self ) :
        """Set default configuration parameters hardwired in this module"""

        print 'setCSpadParameters'


        self.quadInDetOrient = [   0,    0,   270,   0]
        self.quadInDetOriInd = [   0,    0,     3,   0]

        self.quadYOffset     = [   0,  800,   800,   0]
        self.quadXOffset     = [   0,    0,   800, 800]

        self.firstPairInQuad = [0, 0,  8, 16]
        self.lastPairInQuad  = [0, 8, 16, 20]


        self.pairInQaudOrient = [ [   0,   0, 270, 270, 180, 180, 270, 270],
                                  [   0,   0, 270, 270, 180, 180, 270, 270],
                                  [   0,   0, 270, 270, 180, 180, 270, 270],
                                  [ 180, 180, 270, 270,   0,   0, 270, 270] ]

        self.pairInQaudOriInd = [ [   0,   0,   3,   3,   2,   2,   3,   3],
                                  [   0,   0,   3,   3,   2,   2,   3,   3],
                                  [   0,   0,   3,   3,   2,   2,   3,   3],
                                  [   2,   2,   1,   1,   0,   3,   3,   2] ]


        self.dPhi = [ [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0.144, 0.466, -0.049, -0.186, -0.291, -0.338, 0.119, 0.135],
                      [0,0,0,0,0,0,0,0] ]

        self.pairXInQaud = [ [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400],
                             [400,600,  0,  0,200,  0,400,400],
                             [200,  0,  0,  0,400,600,  0,  0] ]

        self.pairYInQaud = [ [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400],
                             [  0,  0,200,  0,400,400,600,400],
                             [400,400,  0,200,  0,  0,200,  0] ]

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
