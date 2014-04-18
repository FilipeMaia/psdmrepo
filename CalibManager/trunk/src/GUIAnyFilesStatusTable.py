#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIAnyFilesStatusTable...
#
#------------------------------------------------------------------------

"""GUIAnyFilesStatusTable GUI shows the status table for the list of files"""

#------------------------------
#  Module's version from SVN --
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
from ConfigParametersForApp import cp
from FileNameManager        import fnm
from Logger                 import logger
import GlobalUtils          as     gu

#---------------------
#  Class definition --
#---------------------
class GUIAnyFilesStatusTable ( QtGui.QWidget ) :
    """Status table of files from the list"""

    dict_status = {True  : 'Yes',
                   False : 'No' }

    def __init__ ( self, parent=None, list_of_files=[], title='') :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(50, 100, 600, 500)
        self.setWindowTitle('Files status table')
        self.setFrame()

        self.title = title

        self.table = QtGui.QTableWidget(self)
        self.makeTable(list_of_files)
 
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.table)
        self.vbox.addStretch(1)     
 
        self.setLayout(self.vbox)

        self.showToolTips()
        
        self.onStatus()
        #self.connectToThread1()

        
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
        #print 'GUIAnyFilesStatusTable: Signal is recieved ' + str(text)
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


    def clearTable(self):
        self.table.clearContents()
        self.table.setRowCount(0)


    def makeTable(self, list_of_files):
        """Makes the table for the list of output and log files"""

        self.list_of_files = list_of_files

        self.rows = len(list_of_files)
        #self.table = QtGui.QTableWidget(self.rows, 6, self)
        self.table.clear()
        self.table.setRowCount(self.rows)
        self.table.setColumnCount(5)

        self.table.setHorizontalHeaderLabels(['File', 'Creation time', 'Size(Byte)', 'Owner', 'Mode'])
        #self.table.setVerticalHeaderLabels([''])

        self.table.verticalHeader().hide()

        self.table.horizontalHeader().setDefaultSectionSize(60)
        self.table.horizontalHeader().resizeSection(0,200)
        self.table.horizontalHeader().resizeSection(1,150)
        self.table.horizontalHeader().resizeSection(2,100)
        self.table.horizontalHeader().resizeSection(3,80)
        self.table.horizontalHeader().resizeSection(4,70)

        self.row = -1
        self.list_of_items = []
 
        for i, fname in enumerate(list_of_files) :

            file_struct = os.path.exists(fname)
            item_fname  = QtGui.QTableWidgetItem( os.path.basename(fname) )
            #item_struct = QtGui.QTableWidgetItem( self.dict_status[file_struct] )
            #item_struct = QtGui.QTableWidgetItem( self.checkNameStructure(fname, self.list_expected) )
            #item_struct = QtGui.QTableWidgetItem( 'N/A' )
            item_ctime  = QtGui.QTableWidgetItem( 'N/A' )
            item_size   = QtGui.QTableWidgetItem( 'N/A' )
            item_owner  = QtGui.QTableWidgetItem( 'N/A' )
            item_mode   = QtGui.QTableWidgetItem( 'N/A' )

            item_fname.setBackgroundColor (cp.colorTabItem)
            item_ctime.setBackgroundColor (cp.colorTabItem)
            item_size .setBackgroundColor (cp.colorTabItem)
            item_owner.setBackgroundColor (cp.colorTabItem)
            item_mode .setBackgroundColor (cp.colorTabItem)

            #item_struct.setTextAlignment(QtCore.Qt.AlignCenter)
            item_ctime .setTextAlignment(QtCore.Qt.AlignCenter)
            item_size  .setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

            self.row += 1
            self.table.setItem(self.row, 0, item_fname)
            self.table.setItem(self.row, 1, item_ctime)
            self.table.setItem(self.row, 2, item_size)
            self.table.setItem(self.row, 3, item_owner)
            self.table.setItem(self.row, 4, item_mode)
            
            row_of_items = [i, fname, item_fname, item_ctime, item_size, item_owner, item_mode]
            self.list_of_items.append(row_of_items)

            #self.table.setSpan(self.row, 0, 1, 5)            
            #self.table.setItem(self.row, 0, self.title_split)

        if self.rows<5 :
            self.table.setFixedWidth(self.table.horizontalHeader().length() + 4)
        else           :
            self.table.setFixedWidth(self.table.horizontalHeader().length() + 24)


        if self.rows<5 :
            self.table.setFixedHeight(self.table.verticalHeader().length() + 29)
        else           :
            self.table.setFixedHeight(200)

        #self.table.setFixedHeight(200)

        self.setTableItems()


    def setTableItems(self) :     

        #self.fname_item_flags = QtCore.Qt.ItemFlags(QtCore.Qt.NoItemFlags|QtCore.Qt.ItemIsUserCheckable )

        for row_of_items in self.list_of_items :
            i, fname, item_fname, item_ctime, item_size, item_owner, item_mode = row_of_items

            #item_fname.setCheckState(0)

            #file_struct = os.path.exists(fname)
            #item_struct.setText( self.dict_status[file_struct] )

            if not os.path.exists(fname) : 
                item_ctime.setText( 'N/A' )
                item_size .setText( 'N/A' )
                item_owner.setText( 'N/A' )
                item_mode .setText( 'N/A' )
                continue
            
            ctime_sec  = os.path.getctime(fname)
            ctime_str  = gu.get_local_time_str(ctime_sec, fmt='%Y-%m-%d %H:%M:%S')
            size_byte  = os.path.getsize(fname)
            file_owner = gu.get_path_owner(fname)
            file_mode  = gu.get_path_mode(fname)

            item_ctime.setText( ctime_str )
            item_size .setText( str(size_byte) )
            item_owner.setText( str(file_owner) )
            item_mode .setText( str(oct(file_mode)) )

        self.setStyle()


    def checkNameStructure(self, fname, list_expected) :     
        """Checks if the base name of the fname path is in the list_expected and return the status string"""

        #print 'fname =', fname
        #print 'list_expected =', list_expected

        base_name = os.path.basename(fname) 

        if list_expected == [] : return self.checkCalibFileNameStructure(base_name)
        if base_name in list_expected : return 'OK'        
        else : return 'WRONG!'


    def checkCalibFileNameStructure(self, fname) :
        """Check that the calibration file name structure is consistent with something like 5-10.data or 5-end.data and return string status"""
        name, ext = os.path.splitext(fname)
        if ext == '.data' : return 'Ext OK'
        else : 'WRONG ext!'


    def setStyle(self):
        #self.setMinimumSize(650,300)
        #self.setStyleSheet(cp.styleBkgd)
        self.setStyleSheet(cp.styleGreenish)
        #self.setStyleSheet(cp.styleBluish)

        #self.but_run   .setStyleSheet (cp.styleButton) 
        #self.but_remove.setStyleSheet (cp.styleButtonBad)
        #self.setFixedWidth(100)
        #self.table.setFixedHeight(500)
        #self.table.setMinimumHeight(200)
        #self.table.setMinimumWidth(600)
        #self.table.adjustSize()
        self.setMinimumWidth(self.table.width())
        self.setFixedHeight(self.table.height())
        #self.setMinimumSize(self.table.size())
        self.setContentsMargins(-9,-9,-9,-9)


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())


    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)

        #self.disconnectFromThread1()

        #try    : del cp.guistatustable # GUIAnyFilesStatusTable
        #except : pass

        #try    : cp.guiccdsettings.close()
        #except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def onStatus(self):
        logger.debug('onStatus', __name__)
        self.setTableItems()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)

    dir = './'
    list_dir = sorted(os.listdir(dir))

    print list_dir

    #widget = GUIAnyFilesStatusTable (parent=None, list_of_files=list_dir)

    widget = GUIAnyFilesStatusTable(parent=None)
    widget.makeTable(list_dir)

    widget.show()
    app.exec_()

#-----------------------------
