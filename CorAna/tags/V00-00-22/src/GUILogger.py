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
from FileNameManager        import fnm

#from GUILoggerLeft   import *
#from GUILoggerRight  import *

#---------------------
#  Class definition --
#---------------------
class GUILogger ( QtGui.QWidget ) :
    """GUI for Logger"""

    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 900, 500)
        self.setWindowTitle('GUI Logger')
        self.setWindowIcon(cp.icon_logger)

        self.setFrame()

        self.box_txt    = QtGui.QTextEdit()
 
        #self.tit_title  = QtGui.QLabel('Logger')
        self.tit_status = QtGui.QLabel('Status:')
        self.tit_level  = QtGui.QLabel('Verbosity level:')
        self.but_close  = QtGui.QPushButton('&Close') 
        self.but_save   = QtGui.QPushButton('&Save log-file') 

        self.list_of_levels = logger.getListOfLevels()
        self.box_level      = QtGui.QComboBox( self ) 
        self.box_level.addItems(self.list_of_levels)
        self.box_level.setCurrentIndex( self.list_of_levels.index(cp.log_level.value()) )
        
        self.hboxM = QtGui.QHBoxLayout()
        self.hboxM.addWidget( self.box_txt )

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.tit_status)
        self.hboxB.addStretch(4)     
        self.hboxB.addWidget(self.tit_level)
        self.hboxB.addWidget(self.box_level)
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.but_save)
        self.hboxB.addWidget(self.but_close)

        self.vbox  = QtGui.QVBoxLayout()
        #self.vbox.addWidget(self.tit_title)
        self.vbox.addLayout(self.hboxM)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)
        
        self.connect( self.but_close, QtCore.SIGNAL('clicked()'), self.onClose )
        self.connect( self.but_save,  QtCore.SIGNAL('clicked()'), self.onSave  )
        self.connect( self.box_level, QtCore.SIGNAL('currentIndexChanged(int)'), self.onBox  )
 
        self.startGUILog()

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        #self           .setToolTip('This GUI is for browsing log messages')
        self.box_txt    .setToolTip('Window for log messages')
        self.but_close  .setToolTip('Close this window')
        self.but_save   .setToolTip('Save current content of the GUI Logger\nin work directory file: '+os.path.basename(self.fname_log))
        self.tit_status .setToolTip('The file name, where this log \nwill be saved at the end of session')
        self.box_level  .setToolTip('Click on this button and \nselect the level of messages \nwhich will be displayed')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def setStyle(self):
        self.           setStyleSheet (cp.styleBkgd)
        #self.tit_title.setStyleSheet (cp.styleTitleBold)
        self.tit_status.setStyleSheet (cp.styleTitle)
        self.tit_level .setStyleSheet (cp.styleTitle)
        self.but_close .setStyleSheet (cp.styleButton)
        self.but_save  .setStyleSheet (cp.styleButton) 
        self.box_level .setStyleSheet (cp.styleButton) 
        self.box_txt   .setReadOnly(True)
        self.box_txt   .setStyleSheet (cp.styleWhiteFixed) 
        #self.box_txt   .ensureCursorVisible()
        #self.tit_title.setAlignment(QtCore.Qt.AlignCenter)
        #self.titTitle.setBold()


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

        try    : cp.guimain.butLogger.setStyleSheet(cp.styleButtonBad)
        except : pass

        try    : del cp.guilogger # GUILogger
        except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def onSave(self):
        logger.debug('onSave:', __name__)
        self.saveLogInFile()


    def onBox(self):
        level_selected = self.box_level.currentText()
        cp.log_level.setValue( level_selected ) 
        logger.info('onBox - selected ' + self.tit_level.text() + ' ' + cp.log_level.value(), __name__)
        logger.setLevel(cp.log_level.value())
        self.box_txt.setText( logger.getLogContent() )


    def saveLogInFile(self):
        logger.info('saveLogInFile' + self.fname_log, __name__)        
        logger.saveLogInFile(self.fname_log)


    def saveLogTotalInFile(self):
        logger.info('saveLogTotalInFile' + self.fname_log_total, __name__)
        if cp.res_save_log : 
            logger.saveLogTotalInFile(self.fname_log_total)


    def getConfirmation(self):
        """Pop-up box for confirmation"""
        msg = QtGui.QMessageBox(self, windowTitle='Confirm closing!',
            text='You are about to close GUI Logger...\nIf the log-file is not saved it will be lost.',
            standardButtons=QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
        msg.setDefaultButton(msg.Save)

        clicked = msg.exec_()

        if   clicked == QtGui.QMessageBox.Save :
            logger.info('Saving is requested', __name__)
        elif clicked == QtGui.QMessageBox.Discard :
            logger.info('Discard is requested', __name__)
        else :
            logger.info('Cancel is requested', __name__)
        return clicked


    def onShow(self):
        logger.info('onShow - is not implemented yet...', __name__)


    def startGUILog(self) :
        self.fname_log       = fnm.log_file()
        self.fname_log_total = fnm.log_file_total()
        self.setStatus(0, 'Log-file: ' + os.path.basename(self.fname_log))

        logger.setLevel(cp.log_level.value())
        self.box_txt.setText(logger.getLogContent())
        
        logger.setGUILogger(self)
        logger.debug('GUILogger is open', __name__)
        self.box_txt.moveCursor(QtGui.QTextCursor.End)


    def appendGUILog(self, msg='...'):
        self.box_txt.append(msg)
        self.scrollDown()


    def scrollDown(self):
        #print 'scrollDown'
        #scrol_bar_v = self.box_txt.verticalScrollBar() # QScrollBar
        #scrol_bar_v.setValue(scrol_bar_v.maximum()) 
        self.box_txt.moveCursor(QtGui.QTextCursor.End)
        self.box_txt.repaint()
        #self.raise_()
        #self.box_txt.update()

        
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
