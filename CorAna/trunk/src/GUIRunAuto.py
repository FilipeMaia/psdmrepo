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
        self.onStatus()
 
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.hboxB)
        self.vbox.addLayout(self.vboxS)
        self.vbox.addStretch(1)     
 
        self.setLayout(self.vbox)

        self.showToolTips()
        self.setStyle()
        
        self.connectToThread1()

        
    #-------------------
    #  Public methods --
    #-------------------


    def connectToThread1(self):
        try : self.connect   ( cp.thread1, QtCore.SIGNAL('update(QString)'), self.updateStatus )
        except : logger.warning('connectToThread1 is failed', __name__)


    def disconnectFromThread1(self):
        try : self.disconnect( cp.thread1, QtCore.SIGNAL('update(QString)'), self.updateStatus )
        except : pass


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

    def onStop(self):
        cp.autoRunStatus = 0            
        self.killAllBatchJobs()
        logger.info('Auto-processing IS STOPPED', __name__)


    def killAllBatchJobs(self):
        logger.debug('stopBatchJobs', __name__)
        bjcora.kill_batch_job_for_cora_split()
        bjcora.kill_batch_job_for_cora_merge()
        for i in range(self.nparts) :
            bjcora.kill_batch_job_for_cora_proc(i)

#-----------------------------

    def onRun(self):
        logger.debug('onRun', __name__)

        if cp.autoRunStatus != 0 :            
            logger.warning('Auto run procedure is already active in stage '+str(cp.autoRunStatus), __name__)

        else :
            self.removeWorkFilesBeforeAutoRun()
            self.onRunSplit()

#-----------------------------

    def updateRunState(self):
        logger.info('Auto run stage '+str(cp.autoRunStatus), __name__)

        if   cp.autoRunStatus == 1 and self.status_split :            
            logger.info('updateRunState: Split is completed, begin processing', __name__)
            self.onRunProc()

        elif cp.autoRunStatus == 2 and self.status_proc : 
            logger.info('updateRunState: Processing is completed, begin merging', __name__)
            self.onRunMerge()

        elif cp.autoRunStatus == 3 and self.status_merge : 
            logger.info('updateRunState: Merging is completed, stop auto-run', __name__)
            cp.autoRunStatus = 0            

#-----------------------------

    def removeWorkFilesBeforeAutoRun(self):
        logger.debug('removeFilesBeforeAutoRun', __name__)
        bjcora.remove_files_cora_split()
        bjcora.remove_files_cora_merge()
        for i in range(self.nparts) :
            bjcora.remove_files_cora_proc(i)

#-----------------------------

    def onRunSplit(self):
        logger.debug('onRunSplit', __name__)
        if self.isReadyToStartRunSplit() :
            bjcora.submit_batch_for_cora_split()
            cp.autoRunStatus = 1


    def isReadyToStartRunSplit(self):
        msg1 = 'JOB IS NOT SUBMITTED !!!\nFirst, set the number of events for data.'
        if  (cp.bat_data_end.value() == cp.bat_data_end.value_def()) :
            logger.warning(msg1, __name__)
            return False

        elif(cp.bat_data_start.value() >= cp.bat_data_end.value()) :
            logger.warning(msg1, __name__)
            return False

        else :
            return True

#-----------------------------

    def onRunProc(self):
        logger.debug('onRunProc', __name__)

        for i in range(self.nparts) :
            if self.isReadyToStartRunProc(i) :
                bjcora.submit_batch_for_cora_proc(i)
                cp.autoRunStatus = 2


    def isReadyToStartRunProc(self, ind):

        fname = fnm.get_list_of_files_cora_split_work()[ind]
        if not os.path.exists(fname) :
            msg1 = 'JOB IS NOT SUBMITTED !!!\nThe file ' + str(fname) + ' does not exist'
            logger.warning(msg1, __name__)
            return False

        fsize = os.path.getsize(fname)
        if fsize < 1 :
            msg2 = 'JOB IS NOT SUBMITTED !!!\nThe file ' + str(fname) + ' has wrong size(Byte): ' + str(fsize) 
            logger.warning(msg2, __name__)
            return False

        msg3 = 'The file ' + str(fname) + ' exists and its size(Byte): ' + str(fsize) 
        logger.info(msg3, __name__)
        return True

#-----------------------------

    def onRunMerge(self):
        logger.debug('onRunMerge', __name__)
        if self.isReadyToStartRunMerge() :
            bjcora.submit_batch_for_cora_merge()
            cp.autoRunStatus = 3


    def isReadyToStartRunMerge(self):
        fstatus, fstatus_str = bjcora.status_for_cora_proc_files()
        if fstatus : 
            logger.info(fstatus_str, __name__)
            return True
        else :
            msg = 'JOB IS NOT SUBMITTED !!!' + fstatus_str
            logger.warning(msg, __name__)
            return False

#-----------------------------

    def onStatus(self):
        self.onStatusSplit()
        self.onStatusProc()
        self.onStatusMerge()

        if cp.autoRunStatus : self.updateRunState()


    def onStatusSplit(self):
        logger.debug('onStatusSplit', __name__)
        bstatus, bstatus_str = bjcora.status_batch_job_for_cora_split()
        fstatus, fstatus_str = bjcora.status_for_cora_split_files(comment='')
        status_str = '1. SPLIT: ' + bstatus_str + '   ' + fstatus_str
        self.setStatusMessage(fstatus, self.lab_status_split, status_str)
        self.status_split = fstatus


    def onStatusProc(self):
        logger.debug('onStatusProc', __name__)
        bstatus, bstatus_str = bjcora.status_batch_job_for_cora_proc_all()
        fstatus, fstatus_str = bjcora.status_for_cora_proc_files(comment='')
        status_str = '2. PROC: ' + bstatus_str + '   ' + fstatus_str
        self.setStatusMessage(fstatus, self.lab_status_proc, status_str)
        self.status_proc = fstatus


    def onStatusMerge(self):
        logger.debug('onStatusMerge', __name__)
        bstatus, bstatus_str = bjcora.status_batch_job_for_cora_merge()
        fstatus, fstatus_str = bjcora.status_for_cora_merge_files(comment='')
        status_str = '3. MERGE: ' + bstatus_str + '   ' + fstatus_str
        self.setStatusMessage(fstatus, self.lab_status_merge, status_str)
        self.setStatusButton(fstatus)
        self.status_merge = fstatus


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
