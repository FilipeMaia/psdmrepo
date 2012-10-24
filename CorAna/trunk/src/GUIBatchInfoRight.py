#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIBatchInfoRight...
#
#------------------------------------------------------------------------

"""GUI sets pars"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
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

from ConfigParametersCorAna import confpars as cp
from GUIKineticMode         import *
from GUIBatchPars           import *
from Logger                 import logger

#---------------------
#  Class definition --
#---------------------
class GUIBatchInfoRight ( QtGui.QWidget ) :
    """GUI Batch Info Right Panel"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Batch Info Right Panel')
        self.setFrame()
        self.setMinimumWidth(400) 

        cp.guikineticmode = GUIKineticMode()
        cp.guibatchpars   = GUIBatchPars()

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(cp.guikineticmode)
        self.vbox.addWidget(cp.guibatchpars)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        #msg_edi = 'WARNING: whatever you edit may be incorrect...\nIt is recommended to use the '
        #self.butInstr  .setToolTip('Select the instrument name from the pop-up menu.')
        pass

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setStyleSheet(cp.styleBkgd)

    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        logger.debug('moveEvent', __name__) 
#        cp.posGUIMain = (self.pos().x(),self.pos().y())

    def closeEvent(self, event):
        logger.info('closeEvent', __name__)
        try    : del cp.guibatchinforight # GUIBatchInfoRight
        except : pass # silently ignore

        try    : cp.guikineticmode.close()
        except : pass

        try    : cp.guibatchpars  .close()
        except : pass

    def onClose(self):
        logger.info('onClose', __name__)
        self.close()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIBatchInfoRight ()
    widget.show()
    app.exec_()

#-----------------------------
