#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUILogger ...
#
#------------------------------------------------------------------------

"""GUI for Logger."""

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
from Logger                 import logger

#from GUILoggerLeft   import *
#from GUILoggerRight  import *

#---------------------
#  Class definition --
#---------------------
class GUILogger ( QtGui.QWidget ) :
    """GUI for Logger"""

    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 500, 600)
        self.setWindowTitle('GUI Logger')
        self.setFrame()

        self.box_txt    = QtGui.QTextEdit()
 
        #self.tit_title  = QtGui.QLabel('Logger')
        self.tit_status = QtGui.QLabel('Status:')
        self.but_close  = QtGui.QPushButton('Close') 
        self.but_save   = QtGui.QPushButton('Save Log-file') 

        self.hboxM = QtGui.QHBoxLayout()
        self.hboxM.addWidget( self.box_txt )

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.tit_status)
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.but_close)
        self.hboxB.addWidget(self.but_save)

        self.vbox  = QtGui.QVBoxLayout()
        #self.vbox.addWidget(self.tit_title)
        self.vbox.addLayout(self.hboxM)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)
        
        self.connect( self.but_close, QtCore.SIGNAL('clicked()'), self.onClose )
        self.connect( self.but_save,  QtCore.SIGNAL('clicked()'), self.onSave  )

        self.startLog()

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        #self           .setToolTip('This GUI is intended for run control and monitoring.')
        self.but_close .setToolTip('Close this window.')
        self.but_save  .setToolTip('Save current content of the GUI Logger\nin file: '+cp.fname_guilog)
        #self.but_show  .setToolTip('Show ...')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.           setStyleSheet (cp.styleBkgd)
        #self.tit_title .setStyleSheet (cp.styleTitle + 'font-size: 18pt; font-family: Courier; font-weight: bold')
        self.tit_status.setStyleSheet (cp.styleTitle)
        self.but_close .setStyleSheet (cp.styleButton)
        self.but_save  .setStyleSheet (cp.styleButton) 
        #self.but_show  .setStyleSheet (cp.styleButton) 

        #self.tit_title .setAlignment(QtCore.Qt.AlignCenter)
        #self.titTitle .setBold()

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

        try    : cp.butLogger.setStyleSheet(cp.styleButtonBad)
        except : pass

        try    : del cp.guilogger # GUILogger
        except : pass


    def onClose(self):
        logger.info('onClose', __name__)

        clicked = self.getConfirmation()
        if clicked == QtGui.QMessageBox.Save :
            logger.info('Saving is requested', __name__)
            self.saveGUILogInFile(cp.path_guilog)

        elif clicked == QtGui.QMessageBox.Discard :
            logger.info('Discard is requested', __name__)
        else :
            logger.info('Cancel is requested', __name__)
            return

        self.close()


    def onSave(self):
        fname = cp.fname_cp.value()
        logger.info('onSave:', __name__)
        self.saveGUILogInFile(cp.path_guilog)

    def saveGUILogInFile(self, fname):
        logger.info('saveGUILogInFile: '+fname, __name__)
        doc = self.box_txt.document() # returns QTextDocument
        txt = doc.toPlainText()
        #print txt
        #print '\n'
        f=open(fname,'w')
        f.write(txt)
        f.close() 


    def getConfirmation(self):
        msg = QtGui.QMessageBox(self, windowTitle='Confirm closing!',
            text='You are about to close GUI Logger...\nIf the log-file is not saved it will be lost.',
            standardButtons=QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
        msg.setDefaultButton(msg.Save)
        return msg.exec_()


    def onShow(self):
        logger.info('onShow - is not implemented yet...', __name__)


    def startLog(self) :
        self.str_start_time = logger.time_stamp( fmt='%Y-%m-%d-%H%M%S' )
        cp.fname_guilog = self.str_start_time + '-guilog.txt'
        cp.path_guilog  = cp.dir_work.value() + '/' + cp.fname_guilog
        self.setStatus(0, 'Log-file: ' + cp.fname_guilog)

        logger.setGUILogger(self)
        self.box_txt.setText( self.str_start_time + ' ==== Start Logger' )

    def appendLog(self, msg='...'):
        self.box_txt.append(msg)

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
    widget = GUILogger ()
    widget.show()
    app.exec_()

#-----------------------------
