#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIRunAuto...
#
#------------------------------------------------------------------------

"""GUI controls the automatic procedure"""

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
from FileNameManager        import fnm
from Logger                 import logger
import GlobalUtils          as     gu
from GUIFileBrowser         import *
from BatchJobCorAna         import bjcora

#---------------------
#  Class definition --
#---------------------
class GUIRunAuto ( QtGui.QWidget ) :
    """GUI controls the automatic procedure"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(50, 100, 700, 500)
        self.setWindowTitle('Auto-run')
        self.setFrame()

        self.dict_status = {True  : 'Yes',
                            False : 'No' }

        self.nparts = cp.bat_img_nparts.value()
        #print 'self.nparts = ', self.nparts

        self.lab_status_split = QtGui.QLabel('')
        self.lab_status_proc  = QtGui.QLabel('')
        self.lab_status_merge = QtGui.QLabel('')

        self.vboxS = QtGui.QVBoxLayout()
        self.vboxS.addWidget(self.lab_status_split)
        self.vboxS.addWidget(self.lab_status_proc )
        self.vboxS.addWidget(self.lab_status_merge)

        self.makeButtons()
 
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.hboxB)
        self.vbox.addLayout(self.vboxS)
        self.vbox.addStretch(1)     
 
        self.setLayout(self.vbox)

        self.showToolTips()
        self.setStyle()
        
        self.onStatus()
        self.connectToThread1()

        
    #-------------------
    #  Public methods --
    #-------------------


    def connectToThread1(self):
        try : self.connect   ( cp.thread1, QtCore.SIGNAL('update(QString)'), self.updateStatus )
        except : logger.warning('connectToThread1 IS FAILED !!!', __name__)


    def disconnectFromThread1(self):
        try : self.disconnect( cp.thread1, QtCore.SIGNAL('update(QString)'), self.updateStatus )
        except : logger.warning('connectToThread1 IS FAILED !!!', __name__)


    def updateStatus(self, text):
        #print 'GUIRunAuto: Signal is recieved ' + str(text)
        self.onStatus()


    def showToolTips(self):
        msg = 'GUI sets system parameters.'
        #self.tit_sys_ram_size.setToolTip(msg)
        self.but_run   .setToolTip('Start auto-processing of the data file')
        self.but_status.setToolTip('Update status info.\nStatus is self-updated by timer.')
        self.but_stop  .setToolTip('Stop auto-processing')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def makeButtons(self):
        """Makes the horizontal box with buttons"""
        self.but_run    = QtGui.QPushButton('Run') 
        self.but_status = QtGui.QPushButton('Status') 
        self.but_stop   = QtGui.QPushButton('Stop') 

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.but_run)
        self.hboxB.addWidget(self.but_status)
        self.hboxB.addWidget(self.but_stop)
        self.hboxB.addStretch(1)     

        self.connect( self.but_run,    QtCore.SIGNAL('clicked()'), self.onRun    )
        self.connect( self.but_status, QtCore.SIGNAL('clicked()'), self.onStatus )
        self.connect( self.but_stop,   QtCore.SIGNAL('clicked()'), self.onStop   )


    def setStyle(self):
        self.setMinimumSize(700,500)
        self.setStyleSheet(cp.styleBkgd)

        self.but_run   .setStyleSheet (cp.styleButton) 
        self.but_status.setStyleSheet (cp.styleButton)
        self.but_stop  .setStyleSheet (cp.styleButton)

        self.but_status.setFixedWidth(100)


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())


    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)

        self.disconnectFromThread1()

        try    : del cp.guirunauto # GUIRunAuto
        except : pass

        #try    : cp.guiccdsettings.close()
        #except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

#-----------------------------

    def onRun(self):
        logger.debug('onRun', __name__)
        bjcora.start_auto_processing()

    def onStop(self):
        logger.debug('onStop', __name__)
        bjcora.stop_auto_processing()

#-----------------------------

    def onStatus(self):
        self.onStatusSplit()
        self.onStatusProc()
        self.onStatusMerge()


    def onStatusSplit(self):
        logger.debug('onStatusSplit', __name__)
        bstatus, bstatus_str = bjcora.status_batch_job_for_cora_split()
        fstatus, fstatus_str = bjcora.status_for_cora_split_files(comment='')
        status_str = '1. SPLIT: ' + bstatus_str + '   ' + fstatus_str
        self.setStatusMessage(fstatus, self.lab_status_split, status_str)


    def onStatusProc(self):
        logger.debug('onStatusProc', __name__)
        bstatus, bstatus_str = bjcora.status_batch_job_for_cora_proc_all()
        fstatus, fstatus_str = bjcora.status_for_cora_proc_files(comment='')
        status_str = '2. PROC: ' + bstatus_str + '   ' + fstatus_str
        self.setStatusMessage(fstatus, self.lab_status_proc, status_str)


    def onStatusMerge(self):
        logger.debug('onStatusMerge', __name__)
        bstatus, bstatus_str = bjcora.status_batch_job_for_cora_merge()
        fstatus, fstatus_str = bjcora.status_for_cora_merge_files(comment='')
        status_str = '3. MERGE: ' + bstatus_str + '   ' + fstatus_str
        self.setStatusMessage(fstatus, self.lab_status_merge, status_str)
        self.setStatusButton(fstatus)


    def setStatusButton(self, status):
        if status :
            self.but_status.setStyleSheet(cp.styleButtonGood)
        else :
            self.but_status.setStyleSheet(cp.styleButtonBad)

    
    def setStatusMessage(self, status, field, msg):
        if status :
            self.setStatus(field, 0, msg)
        else :
            self.setStatus(field, 2, msg)


    def setStatus(self, field, status_index=0, msg=''):
        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0 : field.setStyleSheet(cp.styleStatusGood)
        if status_index == 1 : field.setStyleSheet(cp.styleStatusWarning)
        if status_index == 2 : field.setStyleSheet(cp.styleStatusAlarm)

        field.setText(msg)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIRunAuto ()
    widget.show()
    app.exec_()

#-----------------------------
