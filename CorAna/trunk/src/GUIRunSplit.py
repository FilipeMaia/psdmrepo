#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIRunSplit...
#
#------------------------------------------------------------------------

"""GUI sets system parameters"""

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
from GUICCDSettings         import *
import GlobalUtils          as     gu
from GUIFileBrowser         import *
from BatchJobCorAna         import bjcora

#---------------------
#  Class definition --
#---------------------
class GUIRunSplit ( QtGui.QWidget ) :
    """GUI sets system parameters"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(50, 100, 850, 500)
        self.setWindowTitle('System Settings Left')
        self.setFrame()

        self.dict_status = {True  : 'Yes',
                            False : 'No' }

        self.nparts = cp.bat_img_nparts.value()

        #print 'self.nparts = ', self.nparts


        self.but_run    = QtGui.QPushButton('Run') 
        self.but_status = QtGui.QPushButton('Check status') 
        #self.but_files  = QtGui.QPushButton('Check files') 
        self.but_brow   = QtGui.QPushButton('Browse') 
        self.but_remove = QtGui.QPushButton('Remove') 

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.but_run)
        self.hboxB.addWidget(self.but_status)
        #self.hboxB.addWidget(self.but_files)
        self.hboxB.addWidget(self.but_brow)
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.but_remove)

        self.connect( self.but_run,    QtCore.SIGNAL('clicked()'), self.onRun    )
        self.connect( self.but_status, QtCore.SIGNAL('clicked()'), self.onStatus )
        #self.connect( self.but_files,  QtCore.SIGNAL('clicked()'), self.onFiles  )
        self.connect( self.but_brow,   QtCore.SIGNAL('clicked()'), self.onBrow   )
        self.connect( self.but_remove, QtCore.SIGNAL('clicked()'), self.onRemove )

        self.table = QtGui.QTableWidget(self.nparts+5, 4, self)
        self.table.setHorizontalHeaderLabels(['File', 'Exists?', 'Creation time', 'Size(Byte)'])
        #self.table.setVerticalHeaderLabels([''])

        self.table.horizontalHeader().setDefaultSectionSize(60)
        self.table.horizontalHeader().resizeSection(0,300)
        self.table.horizontalHeader().resizeSection(1,60)
        self.table.horizontalHeader().resizeSection(2,150)
        self.table.horizontalHeader().resizeSection(3,120)

        self.fname_item_flags = QtCore.Qt.ItemFlags(QtCore.Qt.NoItemFlags|QtCore.Qt.ItemIsUserCheckable )


        self.row = -1
        #self.title_split = QtGui.QTableWidgetItem( 'Split files:' )
        #self.table.setSpan(self.row, 0, 1, 5)            
        #self.table.setItem(self.row, 0, self.title_split)

        #self.list_of_files_cora_split_work  = fnm.get_list_of_files_cora_split_work()
        #self.list_of_files_cora_result_work = fnm.get_list_of_files_cora_result_work()

        self.list_of_items = []

        self.list_of_files = fnm.get_list_of_files_cora_split_all()


        for i, fname in enumerate(self.list_of_files) :

            file_exists = os.path.exists(fname)
            item_fname = QtGui.QTableWidgetItem( os.path.basename(fname) )
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


        self.setTableItems()

 
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.hboxB)
        self.vbox.addWidget(self.table)
        #self.vbox.addStretch(1)     
 
        self.setLayout(self.vbox)

        self.showToolTips()
        self.setStyle()
        #self.setFields()
        #self.setStatus()

    #-------------------
    #  Public methods --
    #-------------------

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


    def setTableItems(self) :     

        for row_of_items in self.list_of_items :
            i, fname, item_fname, item_exists, item_ctime, item_size = row_of_items

            item_fname.setCheckState(0)

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


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())


    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        try    : del cp.guirunsplit # GUIRunSplit
        except : pass

        #try    : cp.guiccdsettings.close()
        #except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()



    def onRun(self):
        logger.debug('onRun', __name__)
        if self.isReadyToStartRun() : bjcora.submit_batch_for_cora_split()
        else : pass


    def isReadyToStartRun(self):

        msg1 = 'JOB IS NOT SUBMITTED !!!\nFirst, set the number of events for data.'

        if  (cp.bat_data_end.value() == cp.bat_data_end.value_def()) :
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

        if bjcora.status_for_cora_file() : self.but_status.setStyleSheet(cp.styleButtonGood)
        else                             : self.but_status.setStyleSheet(cp.styleButtonBad)
        bjcora.check_batch_job_for_cora_split()
        #blp.parse_batch_log_data_scan()
        #self.set_fields()

        self.onFiles()
        self.setTableItems()


    def onFiles(self):
        logger.debug('onFiles', __name__)
        bjcora.check_work_files_cora()


    def onBrow(self):
        logger.debug('onBrowser', __name__)
        try    :
            cp.guifilebrowser.close()
            self.but_brow.setStyleSheet(cp.styleButtonBad)
        except :
            self.but_brow.setStyleSheet(cp.styleButtonGood)
            cp.guifilebrowser = GUIFileBrowser(None, fnm.get_list_of_files_cora_split(), fnm.path_cora_split_psana_cfg())
            cp.guifilebrowser.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.guifilebrowser.show()


    def onRemove(self):
        logger.debug('onRemove', __name__)
        bjcora.remove_files_cora()
        #self.on_but_status()


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIRunSplit ()
    widget.show()
    app.exec_()

#-----------------------------
