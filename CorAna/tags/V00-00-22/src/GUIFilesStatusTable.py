#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIFilesStatusTable...
#
#------------------------------------------------------------------------

"""GUI controls the merging procedure"""

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

#---------------------
#  Class definition --
#---------------------
class GUIFilesStatusTable ( QtGui.QWidget ) :
    """GUI controls the merging procedure"""

    def __init__ ( self, parent=None, list_of_files=[], title='') :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(50, 100, 660, 150)
        self.setWindowTitle('Run merging')
        self.setFrame()

        self.title = title
        self.list_of_files = list_of_files

        self.dict_status = {True  : 'Yes',
                            False : 'No' }

        #self.lab_title = QtGui.QLabel(self.title)
        #self.hboxT = QtGui.QHBoxLayout()
        #self.hboxT.addWidget(self.lab_title)

        self.lab_status = QtGui.QLabel('Batch job status: ')
        self.hboxS = QtGui.QHBoxLayout()
        self.hboxS.addWidget(self.lab_status)

        self.makeTable()
 
        self.vbox = QtGui.QVBoxLayout()
        #self.vbox.addLayout(self.hboxT)
        self.vbox.addLayout(self.hboxS)
        self.vbox.addWidget(self.table)
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
        except : logger.warning('connectToThread1 is failed', __name__)


    def disconnectFromThread1(self):
        try : self.disconnect( cp.thread1, QtCore.SIGNAL('update(QString)'), self.updateStatus )
        except : pass


    def updateStatus(self, text):
        #print 'GUIFilesStatusTable: Signal is recieved ' + str(text)
        self.onStatus()


    def showToolTips(self):
        msg = 'GUI sets system parameters.'
        #self.tit_sys_ram_size.setToolTip(msg)

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def makeTable(self):
        """Makes the table for the list of output and log files"""

        self.rows = len(self.list_of_files)
        self.table = QtGui.QTableWidget(self.rows, 4, self)
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
 
        for i, fname in enumerate(self.list_of_files) :

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

        if self.rows<5 :
            self.table.setFixedWidth(self.table.horizontalHeader().length() + 4)
        else           :
            self.table.setFixedWidth(self.table.horizontalHeader().length() + 24)
        #self.table.setFixedHeight(self.table.verticalHeader().length() + 29)
        self.table.setMinimumHeight(150)


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
        #self.setMinimumSize(650,200)
        self.setStyleSheet(cp.styleBkgd)
        #self.but_run   .setStyleSheet (cp.styleButton) 
        #self.but_remove.setStyleSheet (cp.styleButtonBad)
        #self.but_status.setFixedWidth(100)


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

        #try    : del cp.guirunmerge # GUIFilesStatusTable
        #except : pass

        #try    : cp.guiccdsettings.close()
        #except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def onStatus(self):
        logger.debug('onStatus', __name__)

        #bstatus, bstatus_str = bjcora.status_batch_job_for_cora_merge()
        #fstatus, fstatus_str = bjcora.status_for_cora_merge_files(comment='')
        #status_str = bstatus_str + '   ' + fstatus_str

        #if fstatus :
        #    self.but_status.setStyleSheet(cp.styleButtonGood)
        #    self.setStatus(0, status_str)
        #else :
        #    self.but_status.setStyleSheet(cp.styleButtonBad)
        #    self.setStatus(2, status_str)

        self.setTableItems()


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
    widget = GUIFilesStatusTable (parent=None, list_of_files=fnm.get_list_of_files_peds_scan())
    #widget = GUIFilesStatusTable (parent=None, list_of_files=fnm.get_list_of_files_peds_aver())
    #widget = GUIFilesStatusTable (parent=None, list_of_files=fnm.get_list_of_files_data_aver())

    widget.setStatus(0, 'Very, very good status...')
    widget.show()
    app.exec_()

#-----------------------------
