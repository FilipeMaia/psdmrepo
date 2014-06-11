#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIHelp ...
#
#------------------------------------------------------------------------

"""GUI for File Browser"""

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
#import GlobalUtils          as     gu

#---------------------
#  Class definition --
#---------------------
class GUIHelp ( QtGui.QWidget ) :
    """GUI Help"""

    def __init__ ( self, parent=None, msg='No message in GUIHelp...' ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(100, 100, 730, 200)
        self.setWindowTitle('GUI Help')
        try : self.setWindowIcon(cp.icon_help)
        except : pass
        self.setFrame()

        self.box_txt    = QtGui.QTextEdit()
        self.tit_status = QtGui.QLabel('Status:')
        self.but_close  = QtGui.QPushButton('Close') 

        self.hboxM = QtGui.QHBoxLayout()
        self.hboxM.addWidget( self.box_txt )

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.tit_status)
        self.hboxB.addStretch(4)     
        self.hboxB.addWidget(self.but_close)

        self.vbox  = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.hboxM)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)
        
        self.connect( self.but_close, QtCore.SIGNAL('clicked()'), self.onClose )
 
        self.setHelpMessage(msg)

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        #self           .setToolTip('This GUI is intended for run control and monitoring.')
        self.but_close .setToolTip('Close this window.')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def setStyle(self):
        self.setMinimumHeight(300)
        self.           setStyleSheet (cp.styleBkgd)
        self.tit_status.setStyleSheet (cp.styleTitle)
        self.but_close .setStyleSheet (cp.styleButton)
        self.box_txt   .setReadOnly   (True)
        self.box_txt   .setStyleSheet (cp.styleWhiteFixed) 


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
        #self.saveLogTotalInFile() # It will be saved at closing of GUIMain

        #try    : cp.guimain.butLogger.setStyleSheet(cp.styleButtonBad)
        #except : pass

        self.box_txt.close()

        try    : del cp.guihelp # GUIHelp
        except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def setHelpMessage(self, msg) :
        logger.debug('Set help message',__name__)
        self.box_txt.setText(msg)
        self.setStatus(0, 'Status: show help info...')


    def setStatus(self, status_index=0, msg=''):
        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0 : self.tit_status.setStyleSheet(cp.styleStatusGood)
        if status_index == 1 : self.tit_status.setStyleSheet(cp.styleStatusWarning)
        if status_index == 2 : self.tit_status.setStyleSheet(cp.styleStatusAlarm)

        #self.tit_status.setText('Status: ' + list_of_states[status_index] + msg)
        self.tit_status.setText(msg)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    w = GUIHelp()
    w.setHelpMessage('This is a test message to test methods of GUIHelp...')
    w.show()
    app.exec_()

#-----------------------------
