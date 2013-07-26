#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIStatus ...
#
#------------------------------------------------------------------------

"""GUI for File Browser"""

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

from ConfigParametersForApp import cp
from Logger                 import logger
#import GlobalUtils          as     gu
from GUIStatusTable         import *

#---------------------
#  Class definition --
#---------------------
class GUIStatus ( QtGui.QWidget ) :
    """GUI Help"""

    def __init__ ( self, parent=None, msg='No message in GUIStatus...' ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(100, 100, 730, 500)
        self.setWindowTitle('GUI Status')
        #try : self.setWindowIcon(cp.icon_help)
        #except : pass
        self.setFrame()

        self.box_txt        = QtGui.QTextEdit()
        self.tit_status     = QtGui.QLabel('Status:')
        self.guistatustable = GUIStatusTable(parent=self, list_of_files=[])

        self.hboxM = QtGui.QHBoxLayout()
        self.hboxM.addWidget( self.box_txt )

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.tit_status)
        #self.hboxB.addStretch(4)     

        self.vbox  = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.hboxB)
        self.vbox.addLayout(self.hboxM)
        self.vbox.addWidget(self.guistatustable)
        self.setLayout(self.vbox)
        
        #self.connect( self.but_close, QtCore.SIGNAL('clicked()'), self.onClose )
 
        self.setStatusMessage(msg)

        self.showToolTips()
        self.setStyle()

        cp.guistatus = self

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        #self           .setToolTip('This GUI is intended for run control and monitoring.')
        #self.but_close .setToolTip('Close this window.')
        pass

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setStyle(self):
        self.setMinimumHeight(300)
        self.           setStyleSheet (cp.styleBkgd)
        self.tit_status.setStyleSheet (cp.styleTitle)
        self.box_txt   .setReadOnly   (True)
        self.box_txt   .setStyleSheet (cp.styleWhiteFixed) 

        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))


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

        try    : del cp.guistatus # GUIStatus
        except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def setStatusMessage(self, msg) :
        logger.debug('Set help message',__name__)
        self.box_txt.setText(msg)
        #self.setStatus(0, 'Status: unknown')


    def setStatus(self, status_index=0, msg=''):
        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0 : self.tit_status.setStyleSheet(cp.styleStatusGood)
        if status_index == 1 : self.tit_status.setStyleSheet(cp.styleStatusWarning)
        if status_index == 2 : self.tit_status.setStyleSheet(cp.styleStatusAlarm)

        #self.tit_status.setText('Status: ' + list_of_states[status_index] + msg)
        self.tit_status.setText(msg)


    def statusOfDir(self, dir) :
        msg = '%s' % dir
        if not os.path.exists(dir) :
            msg += ' DOES NOT EXIST'
            self.setStatus(2,msg)
            self.setStatusMessage('DOES NOT EXIST !!!')
            self.guistatustable.clearTable()
        else :
            msg += ' IS AVAILABLE'
            self.setStatus(0,msg)

            list = sorted(os.listdir(dir))
            list_of_files = []
            for name in list :
                path = os.path.join(dir,name)
                print path
                list_of_files.append(path)
                
            #print list_dir

            msgw = '# of entries %i:\n' % len(list)
            #for name in list :
            #    msgw += '\n   %s' % name

            self.setStatusMessage(msgw)
            self.guistatustable.makeTable(list_of_files)
            #self.guistatustable.setTableItems()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    w = GUIStatus()
    w.setStatusMessage('Test of GUIStatus...')
    w.statusOfDir('./')
    w.show()
    app.exec_()

#-----------------------------
