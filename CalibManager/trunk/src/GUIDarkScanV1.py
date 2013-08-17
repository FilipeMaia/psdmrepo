#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDarkScan...
#
#------------------------------------------------------------------------

"""GUI for dark run scan"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#-------------------
#  Import modules --
#-------------------
import sys
import os

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

from ConfigParametersForApp import cp
from Logger                 import logger

from GUIDarkScan            import *
from BatchJobPedestals      import bjpeds
from GUIFileBrowser         import *
from GUIFilesStatusTable    import *

#-----------------------------

class GUIDarkScan ( QtGui.QWidget ) :
    """GUI works with dark run"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 650, 30)
        self.setWindowTitle('GUI for dark run scan')
        self.setFrame()

        #self.lab_src     = QtGui.QLabel('1. Set data source info:')
        self.but_scan  = QtGui.QPushButton('Scan')
        self.but_brow  = QtGui.QPushButton('File browser')

        self.table_scan = GUIFilesStatusTable (parent=None, list_of_files=fnm.get_list_of_files_peds_scan())
        #self.table_aver = GUIFilesStatusTable (parent=None, list_of_files=fnm.get_list_of_files_peds_aver())
 
        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addWidget(self.but_scan)
        self.hbox.addWidget(self.but_brow)
        self.hbox.addStretch(1)

        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.table_scan)
        self.setLayout(self.vbox)

        self.connect( self.but_scan,     QtCore.SIGNAL('clicked()'),          self.on_but_scan )
        self.connect( self.but_brow,     QtCore.SIGNAL('clicked()'),          self.on_but_brow )

        self.showToolTips()
        self.setStyle()
 
    def showToolTips(self):
        pass
        #self           .setToolTip('Use this GUI to work with xtc file.')
        #self.edi_path   .setToolTip('The path to the xtc file for processing in this GUI')
        #self.but_path   .setToolTip('Push this button and select \nthe xtc file for dark run')

    def setStyle(self):
        pass
        #self.lab_src.setStyleSheet (cp.styleTitle)

        #width = 60
        #self.setMinimumWidth(700)
        #self.setStyleSheet(cp.styleBkgd)
        #tit0   .setStyleSheet (cp.styleTitle)

        #self.cbx_all_chunks.setStyleSheet (cp.styleLabel)
        #self.lab_status    .setStyleSheet (cp.styleLabel)
        #self.lab_batch     .setStyleSheet (cp.styleLabel)

        self.check_status()

 
    def on_but_scan(self):
        logger.info('on_but_scan', __name__) 
        bjpeds.submit_batch_for_peds_scan()
        self.connectToThread1()

    def on_but_brow(self):
        logger.info('on_but_brow', __name__) 
        try    :
            cp.guifilebrowser.close()
            #self.but_browse.setStyleSheet(cp.styleButtonBad)
        except :
            #self.but_browse.setStyleSheet(cp.styleButtonGood)
            cp.guifilebrowser = GUIFileBrowser(None, fnm.get_list_of_files_peds_scan(), selected_file=fnm.path_peds_scan_psana_cfg())
            cp.guifilebrowser.move(cp.guimain.pos().__add__(QtCore.QPoint(740,140))) # open window with offset w.r.t. parent
            cp.guifilebrowser.show()


    def connectToThread1(self):
        try : self.connect   ( cp.thread1, QtCore.SIGNAL('update(QString)'), self.check_status )
        except : logger.warning('connectToThread1 is failed', __name__)

    def disconnectFromThread1(self):
        try : self.disconnect( cp.thread1, QtCore.SIGNAL('update(QString)'), self.check_status )
        except : pass


    def check_status(self):
        self.check_status_scan()
        #self.check_status_aver()

        print 'Check batch status here...'

        if cp.procDarkStatus == 0 : self.disconnectFromThread1()


    def check_status_scan(self):
        bstatus, bstatus_str = bjpeds.status_batch_job_for_peds_scan()
        fstatus, fstatus_str = bjpeds.status_for_peds_scan_files()
        msg = 'Scan: ' + bstatus_str + '   ' + fstatus_str
        self.set_for_status(fstatus, msg, self.table_scan)
        if fstatus :
            print 'Dark-scan files are available'
            #blp.parse_batch_log_peds_scan()

            print 'batch log parser should be called here....' 

            #self.set_fields()
        if cp.procDarkStatus & 1 : logger.info(msg, __name__) 


    def set_for_status(self, status, msg, table):
        """Sets the status string above the table and color of the submit button"""
        if status : table.setStatus(0, msg)
        else :      table.setStatus(2, msg)



#-----------------------------
  
    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)

        #if cp.res_save_log : 
        #    logger.saveLogInFile     ( fnm.log_file() )
        #    logger.saveLogTotalInFile( fnm.log_file_total() )

        #try    : self.gui_win.close()
        #except : pass

        #try    : del cp.guimain
        #except : pass

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIDarkScan ()
    widget.show()
    app.exec_()

#-----------------------------
#-----------------------------
