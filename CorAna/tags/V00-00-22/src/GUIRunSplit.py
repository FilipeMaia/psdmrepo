#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIRunSplit...
#
#------------------------------------------------------------------------

"""GUI controls the splitting procedure"""

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
class GUIRunSplit ( QtGui.QWidget ) :
    """GUI controls the splitting procedure"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(50, 100, 700, 500)
        self.setWindowTitle('Run splitting')
        self.setFrame()

        self.dict_status = {True  : 'Yes',
                            False : 'No' }

        self.nparts = cp.bat_img_nparts.value()
        #print 'self.nparts = ', self.nparts

        self.lab_status = QtGui.QLabel('Batch job status: ')
        self.hboxS = QtGui.QHBoxLayout()
        self.hboxS.addWidget(self.lab_status)

        self.makeButtons()
        self.makeTable()
 
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.hboxB)
        self.vbox.addLayout(self.hboxS)
        self.vbox.addWidget(self.table)
        #self.vbox.addStretch(1)     
 
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
        except : logger.warning('connectToThread1 is failed', __name__)


    def disconnectFromThread1(self):
        try : self.disconnect( cp.thread1, QtCore.SIGNAL('update(QString)'), self.updateStatus )
        except : pass


    def updateStatus(self, text):
        #print 'GUIRunSplit: Signal is recieved ' + str(text)
        self.onStatus()


    def showToolTips(self):
        msg = 'GUI sets system parameters.'
        #self.tit_sys_ram_size.setToolTip(msg)
        self.but_run   .setToolTip('Submit batch job')  
        self.but_status.setToolTip('Update status info.\nStatus is self-updated by timer')
        self.but_brow  .setToolTip('Open/close file viewer')
        self.but_remove.setToolTip('Remove output files')


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
        self.but_brow   = QtGui.QPushButton('View') 
        self.but_remove = QtGui.QPushButton('Remove files') 

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.but_run)
        self.hboxB.addWidget(self.but_status)
        self.hboxB.addWidget(self.but_brow)
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.but_remove)

        self.connect( self.but_run,    QtCore.SIGNAL('clicked()'), self.onRun    )
        self.connect( self.but_status, QtCore.SIGNAL('clicked()'), self.onStatus )
        self.connect( self.but_brow,   QtCore.SIGNAL('clicked()'), self.onBrow   )
        self.connect( self.but_remove, QtCore.SIGNAL('clicked()'), self.onRemove )


    def makeTable(self):
        """Makes the table for the list of output and log files"""
        self.table = QtGui.QTableWidget(self.nparts+8, 4, self)
        self.table.setHorizontalHeaderLabels(['File', 'Exists?', 'Creation time', 'Size(Byte)'])
        #self.table.setVerticalHeaderLabels([''])

        self.table.verticalHeader().hide()

        self.table.horizontalHeader().setDefaultSectionSize(60)
        self.table.horizontalHeader().resizeSection(0,300)
        self.table.horizontalHeader().resizeSection(1,60)
        self.table.horizontalHeader().resizeSection(2,150)
        self.table.horizontalHeader().resizeSection(3,120)

        self.row = -1
        self.list_of_items = []
        self.list_of_files = fnm.get_list_of_files_cora_split_all() \
                           + fnm.get_list_of_files_cora_split_intensity()

        for i, fname in enumerate(self.list_of_files) :
            #print fname

            file_exists = os.path.exists(fname)
            item_fname  = QtGui.QTableWidgetItem( os.path.basename(fname) )
            item_exists = QtGui.QTableWidgetItem( self.dict_status[file_exists] )
            item_ctime  = QtGui.QTableWidgetItem( 'N/A' )
            item_size   = QtGui.QTableWidgetItem( 'N/A' )

            item_exists.setTextAlignment(QtCore.Qt.AlignCenter)
            item_ctime .setTextAlignment(QtCore.Qt.AlignCenter)
            item_size  .setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

            self.row += 1
            self.table.setItem(self.row, 0, item_fname)
            self.table.setItem(self.row, 1, item_exists)
            self.table.setItem(self.row, 2, item_ctime)
            self.table.setItem(self.row, 3, item_size)
            
            row_of_items = [i, fname, item_fname, item_exists, item_ctime, item_size]
            self.list_of_items.append(row_of_items)

            #self.table.setSpan(self.row, 0, 1, 5)            
            #self.table.setItem(self.row, 0, self.title_split)

        self.table.setFixedWidth(self.table.horizontalHeader().length() + 20)


    def setTableItems(self) :     

        #self.fname_item_flags = QtCore.Qt.ItemFlags(QtCore.Qt.NoItemFlags|QtCore.Qt.ItemIsUserCheckable )

        for row_of_items in self.list_of_items :
            i, fname, item_fname, item_exists, item_ctime, item_size = row_of_items

            #item_fname.setCheckState(0)

            file_exists = os.path.exists(fname)
            item_exists.setText( self.dict_status[file_exists] )

            if not file_exists : 
                item_ctime.setText( 'N/A' )
                item_size .setText( 'N/A' )
                continue
            
            ctime_sec  = os.path.getctime(fname)
            ctime_str  = gu.get_local_time_str(ctime_sec, fmt='%Y-%m-%d %H:%M:%S')
            size_byte  = os.path.getsize(fname)
            item_ctime.setText( ctime_str )
            item_size .setText( str(size_byte) )


    def setStyle(self):
        self.setMinimumSize(700,500)
        self.setStyleSheet(cp.styleBkgd)

        self.but_run   .setStyleSheet (cp.styleButton) 
        self.but_status.setStyleSheet (cp.styleButton)
        #self.but_files .setStyleSheet (cp.styleButton)
        self.but_brow  .setStyleSheet (cp.styleButton)
        self.but_remove.setStyleSheet (cp.styleButtonBad)

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

        try    : del cp.guirunsplit # GUIRunSplit
        except : pass

        #try    : cp.guiccdsettings.close()
        #except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()



    def onRun(self):
        logger.debug('onRun', __name__)
        if self.isReadyToStartRun() :
            self.onRemove()
            bjcora.submit_batch_for_cora_split()
            job_id_str = str(bjcora.get_batch_job_id_cora_split())
            time_str   = str(bjcora.get_batch_job_cora_split_time_string())
            self.setStatus(0,'Batch job '+ job_id_str + ' is submitted at ' + time_str)


    def isReadyToStartRun(self):

        if cp.autoRunStatus != 0 :            
            msg = 'JOB IS NOT SUBMITTED !!! because Auto-processing procedure is active in stage '+str(cp.autoRunStatus)
            logger.warning(msg, __name__)
            return False 

        msg1 = 'JOB IS NOT SUBMITTED !!!\nFirst, set the number of events for data.'

        if(cp.bat_data_end.value() == cp.bat_data_end.value_def()) :
            #self.edi_bat_end.setStyleSheet(cp.styleEditBad)
            logger.warning(msg1, __name__)
            return False

        elif(cp.bat_data_start.value() >= cp.bat_data_end.value()) :
            #self.edi_bat_end.setStyleSheet(cp.styleEditBad)
            #self.edi_bat_start.setStyleSheet(cp.styleEditBad)
            logger.warning(msg1, __name__)
            return False

        else :
            #self.edi_bat_end.setStyleSheet  (cp.styleEditInfo)
            #self.edi_bat_start.setStyleSheet(cp.styleEditInfo)
            return True


    def onStatus(self):
        logger.debug('onStatus', __name__)

        #bjcora.check_batch_job_for_cora_split() # for record in Logger
        bstatus, bstatus_str = bjcora.status_batch_job_for_cora_split()
        fstatus, fstatus_str = bjcora.status_for_cora_split_files(comment='')
        status_str = bstatus_str + '   ' + fstatus_str

        if fstatus :
            self.but_status.setStyleSheet(cp.styleButtonGood)
            self.setStatus(0, status_str)
        else :
            self.but_status.setStyleSheet(cp.styleButtonBad)
            self.setStatus(2, status_str)

        self.setTableItems()


    def onBrow(self):
        logger.debug('onBrowser', __name__)
        try    :
            cp.guifilebrowser.close()
            self.but_brow.setStyleSheet(cp.styleButtonBad)
        except :
            self.but_brow.setStyleSheet(cp.styleButtonGood)
            cp.guifilebrowser = GUIFileBrowser(None, fnm.get_list_of_files_cora_split() +\
                                               fnm.get_list_of_files_cora_split_intensity(), \
                                               fnm.path_cora_split_psana_cfg())
            cp.guifilebrowser.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.guifilebrowser.show()


    def onRemove(self):
        logger.debug('onRemove', __name__)
        bjcora.remove_files_cora_split()
        self.onStatus()


    def setStatus(self, status_index=0, msg=''):
        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0 : self.lab_status.setStyleSheet(cp.styleStatusGood)
        if status_index == 1 : self.lab_status.setStyleSheet(cp.styleStatusWarning)
        if status_index == 2 : self.lab_status.setStyleSheet(cp.styleStatusAlarm)

        #self.lab_status.setText('Status: ' + list_of_states[status_index] + msg)
        self.lab_status.setText(msg)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIRunSplit ()
    widget.show()
    app.exec_()

#-----------------------------
