#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIRunProc...
#
#------------------------------------------------------------------------

"""GUI controls the time correlation processing"""

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
from GUICCDSettings         import *
import GlobalUtils          as     gu
from GUIFileBrowser         import *
from BatchJobCorAna         import bjcora

#---------------------
#  Class definition --
#---------------------
class GUIRunProc ( QtGui.QWidget ) :
    """GUI controls the time correlation processing"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(50, 100, 700, 500)
        self.setWindowTitle('Run processing')
        self.setFrame()

        self.dict_status = { True  : 'Yes',
                             False : 'No' }

        self.nparts = cp.bat_img_nparts.value()

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

        self.onStatus()        # calls self.setTableItems()
        self.connectToThread1()
        
    #-------------------
    #  Public methods --
    #-------------------

    def connectToThread1(self):
        try : self.connect( cp.thread1, QtCore.SIGNAL('update(QString)'), self.updateStatus )
        except : logger.warning('connectToThread1 is failed', __name__)


    def disconnectFromThread1(self):
        try : self.disconnect( cp.thread1, QtCore.SIGNAL('update(QString)'), self.updateStatus )
        except : pass


    def updateStatus(self, text):
        #print 'GUIRunProc: Signal is recieved ' + str(text)
        self.onStatus()


    def showToolTips(self):
        msg = 'GUI sets system parameters.'
        #self.tit_sys_ram_size.setToolTip(msg)
        self.but_run   .setToolTip('Submit batch job')  
        self.but_status.setToolTip('Update status info.\nStatus is self-updated by timer')
        self.but_brow  .setToolTip('Open/close file browser')
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

        self.setStyle()


    def setStyle(self):
        self.setStyleSheet(cp.styleBkgd)

        self.but_run   .setStyleSheet (cp.styleButton) 
        self.but_status.setStyleSheet (cp.styleButton)
        self.but_brow  .setStyleSheet (cp.styleButton)
        self.but_remove.setStyleSheet (cp.styleButtonBad)

        self.but_status.setFixedWidth(100)


    def makeTable(self):
        """Makes the table for the list of output and log files"""
        self.table = QtGui.QTableWidget(self.nparts+2, 8, self)
        self.table.setHorizontalHeaderLabels(['File', 'Exists?', 'Create time', 'Size(Byte)', 'Sub.time', 'Job Id', 'Stat.', 'Log?'])
        #self.table.setVerticalHeaderLabels(['' for i in range(self.nparts+2)])
        #self.table.setRowCount(self.nparts+1)

        self.table.verticalHeader().hide()

        self.table.horizontalHeader().setDefaultSectionSize(60)
        self.table.horizontalHeader().resizeSection(0,200)
        self.table.horizontalHeader().resizeSection(1,50)
        self.table.horizontalHeader().resizeSection(2,150)
        self.table.horizontalHeader().resizeSection(3,80)
        self.table.horizontalHeader().resizeSection(4,70)
        self.table.horizontalHeader().resizeSection(5,60)
        self.table.horizontalHeader().resizeSection(6,50)
        self.table.horizontalHeader().resizeSection(7,40)

        self.list_of_files_work     = fnm.get_list_of_files_cora_proc_work()
        self.list_of_files_work_log = fnm.get_list_of_files_cora_proc_work_log()
        self.list_of_files_main     = fnm.get_list_of_files_cora_proc_main()
        self.list_of_files          = fnm.get_list_of_files_cora_proc_all()

        self.row = -1
        self.makeTableAdd()

        self.list_of_items = []
        self.list_of_work_files = zip(fnm.get_list_of_files_cora_proc_work(), fnm.get_list_of_files_cora_proc_work_log())

        for i, (fname,lname) in enumerate(self.list_of_work_files) :

            file_exists = os.path.exists(fname)
            logf_exists = os.path.exists(lname)
            item_fname  = QtGui.QTableWidgetItem( str(os.path.basename(fname)) )
            item_exists = QtGui.QTableWidgetItem( self.dict_status[file_exists] )
            item_ctime  = QtGui.QTableWidgetItem( 'N/A' )
            item_size   = QtGui.QTableWidgetItem( 'N/A' )
            item_stime  = QtGui.QTableWidgetItem( 'N/A' )
            item_jobid  = QtGui.QTableWidgetItem( 'N/A' )
            item_jobst  = QtGui.QTableWidgetItem( 'N/A' )
            item_lname  = QtGui.QTableWidgetItem( 'N/A' ) # os.path.basename(lname) )

            item_fname.setCheckState(QtCore.Qt.Checked) # Unchecked, PartiallyChecked, Checked

            item_fname .setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter ) #| QtCore.Qt.AlignAbsolute)
            item_exists.setTextAlignment(QtCore.Qt.AlignCenter)
            item_ctime .setTextAlignment(QtCore.Qt.AlignCenter)
            item_size  .setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter )
            item_stime .setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter | QtCore.Qt.AlignAbsolute)
            item_jobid .setTextAlignment(QtCore.Qt.AlignCenter)
            item_jobst .setTextAlignment(QtCore.Qt.AlignCenter)
            item_lname .setTextAlignment(QtCore.Qt.AlignCenter)

            self.row += 1
            self.table.setItem(self.row, 0, item_fname)
            self.table.setItem(self.row, 1, item_exists)
            self.table.setItem(self.row, 2, item_ctime)
            self.table.setItem(self.row, 3, item_size)
            self.table.setItem(self.row, 4, item_stime)
            self.table.setItem(self.row, 5, item_jobid)
            self.table.setItem(self.row, 6, item_jobst)
            self.table.setItem(self.row, 7, item_lname)
            
            row_of_items = [i, fname, item_fname, item_exists, item_ctime, item_size, item_stime, item_jobid, item_jobst, lname, item_lname]
            self.list_of_items.append(row_of_items)

            #self.table.setSpan(self.row, 0, 1, 5)            
            #self.table.setItem(self.row, 0, self.title_split)

        self.table.setFixedWidth(self.table.horizontalHeader().length() + 20)

        #self.item_fname_header = self.table.horizontalHeaderItem(0)
        #print 'self.item_fname_header.text(): ', self.item_fname_header.text() 
        #flags = QtCore.Qt.ItemFlags(QtCore.Qt.NoItemFlags|QtCore.Qt.ItemIsUserCheckable )
        #self.item_fname_header.setFlags(flags)
        #self.item_fname_header.setCheckState(QtCore.Qt.PartiallyChecked) # Unchecked, PartiallyChecked, Checked

        self.cbx = QtGui.QCheckBox(self.table.horizontalHeader())
        self.cbx.setCheckState(QtCore.Qt.Checked)
        self.cbx.setGeometry(QtCore.QRect(3, 4, 16, 17)) # (self.table.columnWidth(0)/2)

        self.connect(self.cbx,  QtCore.SIGNAL('stateChanged(int)'), self.onCBox )

        

    def makeTableAdd(self):

        flags = QtCore.Qt.ItemFlags(QtCore.Qt.NoItemFlags) # |QtCore.Qt.ItemIsUserCheckable )
        
        self.list_of_files_add = fnm.get_list_of_files_cora_proc()
        self.list_of_items_add = []

        for i, fname in enumerate(self.list_of_files_add) :
            file_exists = os.path.exists(fname)
            item_fname  = QtGui.QTableWidgetItem( str(os.path.basename(fname)) )
            item_exists = QtGui.QTableWidgetItem( self.dict_status[file_exists] )
            item_ctime  = QtGui.QTableWidgetItem( 'N/A' )
            item_size   = QtGui.QTableWidgetItem( 'N/A' )

            if fname == fnm.path_cora_proc_tau_in() :
                item_fname.setFlags(flags)
                #item_fname.setCheckState(QtCore.Qt.Unchecked) # Unchecked, PartiallyChecked, Checked
            else :
                item_fname.setCheckState(QtCore.Qt.Checked)

            item_fname .setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter ) #| QtCore.Qt.AlignAbsolute)
            item_exists.setTextAlignment(QtCore.Qt.AlignCenter)
            item_ctime .setTextAlignment(QtCore.Qt.AlignCenter)
            item_size  .setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter )

            self.row += 1
            self.table.setItem(self.row, 0, item_fname)
            self.table.setItem(self.row, 1, item_exists)
            self.table.setItem(self.row, 2, item_ctime)
            self.table.setItem(self.row, 3, item_size)

            row_of_items_add = [i, fname, item_fname, item_exists, item_ctime, item_size]
            self.list_of_items_add.append(row_of_items_add)



    def setTableItems(self) :     

        self.fname_item_flags = QtCore.Qt.ItemFlags(QtCore.Qt.NoItemFlags|QtCore.Qt.ItemIsUserCheckable )

        for row_of_items in self.list_of_items :
            i, fname, item_fname, item_exists, item_ctime, item_size, item_stime, item_jobid, item_jobst, lname, item_lname = row_of_items
            #print 'set item i, fname, lname : ', i, fname, lname

            file_exists = os.path.exists(fname)
            item_exists.setText( self.dict_status[file_exists] )

            if file_exists : 
                ctime_sec  = os.path.getctime(fname)
                ctime_str  = gu.get_local_time_str(ctime_sec, fmt='%Y-%m-%d %H:%M:%S')
                size_byte  = os.path.getsize(fname)
                item_ctime.setText( ctime_str )
                item_size .setText( str(size_byte) )
            else :
                item_ctime.setText( 'N/A' )
                item_size .setText( 'N/A' )

            job_id_str = str(bjcora.get_batch_job_id_cora_proc(i))
            time_str   = str(bjcora.get_batch_job_cora_proc_time_string(i))
            time_str_fields = time_str.split(' ')

            item_stime.setText(time_str_fields[1])
            item_jobid.setText(job_id_str)

            job_status = str(bjcora.status_batch_job_for_cora_proc(i))
            item_jobst.setText(job_status)

            if os.path.exists(lname) : 
                item_lname.setText( 'Yes' )
            else :
                item_lname.setText( 'N/A' )

            self.setTableItemsAdd()


    def setTableItemsAdd(self) :     
        for row_of_items_add in self.list_of_items_add :
            i, fname, item_fname, item_exists, item_ctime, item_size = row_of_items_add

            file_exists = os.path.exists(fname)
            item_exists.setText( self.dict_status[file_exists] )

            if file_exists : 
                ctime_sec  = os.path.getctime(fname)
                ctime_str  = gu.get_local_time_str(ctime_sec, fmt='%Y-%m-%d %H:%M:%S')
                size_byte  = os.path.getsize(fname)
                item_ctime.setText( ctime_str )
                item_size .setText( str(size_byte) )
            else :
                item_ctime.setText( 'N/A' )
                item_size .setText( 'N/A' )


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

        try    : del cp.guirunproc # GUIRunProc
        except : pass

        #try    : cp.guiccdsettings.close()
        #except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def onRun(self):
        logger.debug('onRun', __name__)

        for row_of_items in self.list_of_items :
            i, fname, item_fname, item_exists, item_ctime, item_size, item_stime, item_jobid, item_jobst, lname, item_lname = row_of_items

            status = item_fname.checkState()
            if status == QtCore.Qt.Checked :
                #print 'i, fname, checkState:   ', i, fname, str(status)
                if self.isReadyToStartRun(i) :
                    self.removeFilesForIndex(i)

                    bjcora.submit_batch_for_cora_proc(i)

                    job_id_str = str(bjcora.get_batch_job_id_cora_proc(i))
                    time_str   = str(bjcora.get_batch_job_cora_proc_time_string(i))
                    time_str_fields = time_str.split(' ')

                    item_stime.setText(time_str_fields[1])
                    item_jobid.setText(job_id_str)

                    job_status = str(bjcora.status_batch_job_for_cora_proc(i))
                    item_jobst.setText(job_status)


    def isReadyToStartRun(self, ind):

        if cp.autoRunStatus != 0 :            
            msg = 'JOB IS NOT SUBMITTED !!! because Auto-processing procedure is active in stage '+str(cp.autoRunStatus)
            logger.warning(msg, __name__)
            return False 

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


    def onCBox(self):
        cbx_check_state = self.cbx.checkState()
        logger.info('onStatus: ' + str(cbx_check_state), __name__)
        for row_of_items in self.list_of_items :
            item_fname = row_of_items[2]
            item_fname.setCheckState(cbx_check_state) #(QtCore.Qt.Checked) # Unchecked, PartiallyChecked
        self.onCBoxAdd()


    def onCBoxAdd(self):
        cbx_check_state = self.cbx.checkState()
        for row_of_items in self.list_of_items_add :
            #fname = row_of_items[1]
            i, fname, item_fname, item_exists, item_ctime, item_size = row_of_items
            if fname != fnm.path_cora_proc_tau_in() :
                item_fname.setCheckState(cbx_check_state) #(QtCore.Qt.Checked) # Unchecked, PartiallyChecked


    def onStatus(self):
        logger.debug('onStatus', __name__)
        bstatus_str = ''
        #bjcora.check_batch_job_for_cora_proc() # for record in Logger
        #bstatus, bstatus_str = bjcora.status_batch_job_for_cora_proc()
        fstatus, fstatus_str = bjcora.status_for_cora_proc_files(comment='')
        #status_str = bstatus_str + '   ' + fstatus_str
        status_str = fstatus_str

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

            cp.guifilebrowser = GUIFileBrowser(None, fnm.get_list_of_files_cora_proc_browser(), \
                                               fnm.path_cora_proc_tau_out())
            cp.guifilebrowser.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.guifilebrowser.show()


    def onRemove(self):
        logger.debug('onRemove', __name__)

        for row_of_items in self.list_of_items :
            if row_of_items[2].checkState() == QtCore.Qt.Checked :
                self.removeFilesForIndex(ind=row_of_items[0])


        list_of_files_to_remove = []
        for row_of_items in self.list_of_items_add :
            if row_of_items[2].checkState() == QtCore.Qt.Checked :
                list_of_files_to_remove.append(row_of_items[1])

        #print 'list_of_files_to_remove:', list_of_files_to_remove
        bjcora.remove_files_for_list(list_of_files_to_remove,'of correlation proc:')
        self.onStatus()


    def removeFilesForIndex(self, ind=None):
        bjcora.remove_files_cora_proc(ind)


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
    widget = GUIRunProc ()
    widget.show()
    app.exec_()

#-----------------------------
