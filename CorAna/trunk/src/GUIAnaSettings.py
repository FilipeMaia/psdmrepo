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
from ConfigParametersCorAna import confpars as cp
from Logger                 import logger
from GUIAnaSettingsLeft     import *
from GUIAnaSettingsRight    import *
from GUIListOfTau           import *

#---------------------
#  Class definition --
#---------------------
class GUIAnaSettings ( QtGui.QWidget ) :
    """GUI Analysis Settings"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(20, 40, 800, 630)
        self.setWindowTitle('Analysis Settings')
        self.setFrame()
 
        self.tit_title  = QtGui.QLabel('Analysis Settings')
        self.tit_status = QtGui.QLabel('Status:')
        self.but_close  = QtGui.QPushButton('Close') 
        self.but_apply  = QtGui.QPushButton('Save') 
        self.but_show   = QtGui.QPushButton('Show Mask && Partitions') 
        cp.guianasettingsleft  = GUIAnaSettingsLeft()
        cp.guianasettingsright = GUIAnaSettingsRight()
        cp.guilistoftau        = GUIListOfTau()

        self.vboxR = QtGui.QVBoxLayout()
        self.vboxR.addWidget(cp.guianasettingsright)
        self.vboxR.addWidget(cp.guilistoftau)

        self.hboxM = QtGui.QHBoxLayout()
        self.hboxM.addWidget(cp.guianasettingsleft)
        #self.hboxM.addWidget(cp.guianasettingsright)
        self.hboxM.addLayout(self.vboxR)

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.tit_status)
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.but_close)
        self.hboxB.addWidget(self.but_apply)
        self.hboxB.addWidget(self.but_show )

        self.vbox  = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.tit_title)
        self.vbox.addLayout(self.hboxM)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)
        
        self.connect( self.but_close, QtCore.SIGNAL('clicked()'), self.onClose )
        self.connect( self.but_apply, QtCore.SIGNAL('clicked()'), self.onSave )
        self.connect( self.but_show , QtCore.SIGNAL('clicked()'), self.onShow  )

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
        self.setMinimumHeight(630)
        self.setMinimumWidth(800)

        self.           setStyleSheet (cp.styleBkgd)
        self.tit_title .setStyleSheet (cp.styleTitleBold)
        self.tit_status.setStyleSheet (cp.styleTitle)
        self.but_close .setStyleSheet (cp.styleButton)
        self.but_apply .setStyleSheet (cp.styleButton) 
        self.but_show  .setStyleSheet (cp.styleButton) 

        self.tit_title .setAlignment(QtCore.Qt.AlignCenter)
        #self.tit_title .setBold()

    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__ ) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__ ) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__ )
        try    : cp.guianasettingsleft.close()
        except : pass

        try    : cp.guianasettingsright.close()
        except : pass

        try    : del cp.guianasettings # GUIAnaSettings
        except : pass

    def onClose(self):
        logger.debug('onClose', __name__ )
        self.close()

    def onShow(self):
        logger.debug('onShow - is not implemented yet', __name__ )

    def onApply(self):
        logger.debug('onApply - is already applied...', __name__ )

    def onSave(self):
        fname = cp.fname_cp.value()
        logger.debug('onSave:', __name__)
        cp.saveParametersInFile( fname )



#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIAnaSettings ()
    widget.show()
    app.exec_()

#-----------------------------
