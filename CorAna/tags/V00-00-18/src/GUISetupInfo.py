#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISetupInfo...
#
#------------------------------------------------------------------------

"""GUI Setup Info"""

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

from GUISetupInfoLeft   import *
from GUISetupInfoRight  import *
from Logger             import logger

#---------------------
#  Class definition --
#---------------------
class GUISetupInfo ( QtGui.QWidget ) :
    """GUI Setup Info"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 500, 630)
        self.setWindowTitle('Setup Info')
        self.setFrame()
 
        self.tit_title  = QtGui.QLabel('Setup Info')
        self.tit_status = QtGui.QLabel('Status: ')
        self.but_close  = QtGui.QPushButton('Close') 
        self.but_apply  = QtGui.QPushButton('Save') 
        self.but_show   = QtGui.QPushButton('Show Image')
        cp.guisetupinfoleft  = GUISetupInfoLeft()
        cp.guisetupinforight = GUISetupInfoRight()

        self.hboxM = QtGui.QHBoxLayout()
        self.hboxM.addWidget(cp.guisetupinfoleft)
        self.hboxM.addWidget(cp.guisetupinforight)

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
        self.connect( self.but_apply, QtCore.SIGNAL('clicked()'), self.onSave  )
        self.connect( self.but_show , QtCore.SIGNAL('clicked()'), self.onShow  )

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        self.but_close .setToolTip('Close this window.')
        self.but_apply .setToolTip('Apply changes to configuration parameters.')
        self.but_show  .setToolTip('Show ...')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setMinimumHeight(630)

        self.           setStyleSheet (cp.styleBkgd)
        self.tit_title .setStyleSheet (cp.styleTitleBold)
        self.tit_status.setStyleSheet (cp.styleTitle)
        self.but_close .setStyleSheet (cp.styleButton)
        self.but_apply .setStyleSheet (cp.styleButton) 
        self.but_show  .setStyleSheet (cp.styleButton) 

        self.tit_title .setAlignment(QtCore.Qt.AlignCenter)
        #self.titTitle .setBold()

    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)

        try    : cp.guisetupinfoleft.close()
        except : pass

        try    : cp.guisetupinforight.close()
        except : pass

        try    : del cp.guisetupinfo # GUISetupInfo
        except : pass

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def onSave(self):
        fname = cp.fname_cp.value()
        logger.debug('onSave:', __name__)# - save all configuration parameters in file: ' + fname, __name__)
        cp.saveParametersInFile( fname )


    def onShow(self):
        logger.debug('onShow - is not implemented yet...', __name__)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUISetupInfo ()
    widget.show()
    app.exec_()

#-----------------------------
