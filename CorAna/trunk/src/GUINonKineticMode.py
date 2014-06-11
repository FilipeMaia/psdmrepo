#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUINonKineticMode...
#
#------------------------------------------------------------------------

"""GUI sets the kinetic mode parameters"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------
from Logger import logger
from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------
class GUINonKineticMode ( QtGui.QWidget ) :
    """GUI sets the non-kinetic mode parameters"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Non-Kinetics Mode:')
        self.setFrame()

        self.tit_nonkinetic = QtGui.QLabel('Non-kinetic mode')
        self.tit_empty      = QtGui.QLabel('does not require extra parameters here...')

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.tit_nonkinetic)
        self.vbox.addWidget(self.tit_empty)
        self.vbox.addStretch(1)     
        self.setLayout(self.vbox)
 
        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        self.tit_nonkinetic.setToolTip('Empty gui for Non-kinetic mode')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):
        self.setFixedHeight(150)

        self.                    setStyleSheet (cp.styleBkgd)
        self.tit_nonkinetic     .setStyleSheet (cp.styleTitle)
        self.tit_empty          .setStyleSheet (cp.styleLabel)
        self.tit_empty          .setAlignment(QtCore.Qt.AlignCenter)

    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        try: # try to delete self object in the cp
            del cp.guinonkineticmode # GUINonKineticMode
        except AttributeError:
            pass # silently ignore

    def processClose(self):
        logger.debug('processClose', __name__)
        self.close()

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__)
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__)
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUINonKineticMode ()
    widget.show()
    app.exec_()

#-----------------------------
