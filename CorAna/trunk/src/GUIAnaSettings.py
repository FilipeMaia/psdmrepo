#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIAnaSettings...
#
#------------------------------------------------------------------------

"""GUI sets parameters for analysis"""

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
from Logger import logger
from ConfigParametersCorAna import confpars as cp

from GUIAnaSettingsLeft  import *
from GUIAnaSettingsRight import *

#---------------------
#  Class definition --
#---------------------
class GUIAnaSettings ( QtGui.QWidget ) :
    """GUI Analysis Settings"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Analysis Settings')
        self.setFrame()
 
        self.titTitle  = QtGui.QLabel('Analysis Settings')
        self.titStatus = QtGui.QLabel('Status: Loading')
        self.butClose  = QtGui.QPushButton('Close') 
        self.butApply  = QtGui.QPushButton('Apply') 
        self.butShow   = QtGui.QPushButton('Show Mask & Partitions') 
        cp.guianasettingsleft  = GUIAnaSettingsLeft()
        cp.guianasettingsright = GUIAnaSettingsRight()

        self.hboxM = QtGui.QHBoxLayout()
        self.hboxM.addWidget(cp.guianasettingsleft)
        self.hboxM.addWidget(cp.guianasettingsright)

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.titStatus)
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.butClose)
        self.hboxB.addWidget(self.butApply)
        self.hboxB.addWidget(self.butShow )

        self.vbox  = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.titTitle)
        self.vbox.addLayout(self.hboxM)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)
        
        self.connect( self.butClose, QtCore.SIGNAL('clicked()'), self.onClose )
        self.connect( self.butApply, QtCore.SIGNAL('clicked()'), self.onApply )
        self.connect( self.butShow , QtCore.SIGNAL('clicked()'), self.onShow  )

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
        #self.          setStyleSheet (cp.styleYellow)
        self.titTitle .setStyleSheet (cp.styleTitle + 'font-size: 18pt; font-family: Courier; font-weight: bold')
        self.titStatus.setStyleSheet (cp.styleTitle)
        self.butClose .setStyleSheet (cp.styleButton)
        self.butApply .setStyleSheet (cp.styleButton) 
        self.butShow  .setStyleSheet (cp.styleButton) 

        self.titTitle .setAlignment(QtCore.Qt.AlignCenter)
        #self.titTitle .setBold()

    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        logger.debug('closeEvent')
        try: # try to delete self object in the cp
            del cp.guianasettings # GUIAnaSettings
        except AttributeError:
            pass # silently ignore

        try    : cp.guianasettingsleft.close()
        except : pass

        try    : cp.guianasettingsright.close()
        except : pass

    def onClose(self):
        logger.info('onClose')
        self.close()

    def resizeEvent(self, e):
        logger.debug('resizeEvent') 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        logger.debug('moveEvent') 
        pass
#        cp.posGUIMain = (self.pos().x(),self.pos().y())


    def onApply(self):
        logger.info('onApply - is not implemented yet')

    def onShow(self):
        logger.info('onShow - is not implemented yet')

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIAnaSettings ()
    widget.show()
    app.exec_()

#-----------------------------
