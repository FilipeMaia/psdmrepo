#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDark...
#
#------------------------------------------------------------------------

"""GUI works with dark run"""

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
from PlotImgSpe             import *
from BatchLogParser         import blp
from GUIFileBrowser         import *
from BatchJobPedestals      import bjpeds
from GUIFilesStatusTable    import *
#import GlobalGraphics       as gg

#---------------------
#  Class definition --
#---------------------
class GUIDark ( QtGui.QWidget ) :
    """GUI works with dark run"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 530, 30)
        self.setWindowTitle('Dark run processing')
        self.setFrame()

        self.cbx_dark = QtGui.QCheckBox('Use dark correction', self)
        self.cbx_dark.setChecked( cp.bat_dark_is_used.value() )

        self.cbx_all_chunks = QtGui.QCheckBox('Use all xtc chunks', self)
        self.cbx_all_chunks.setChecked( cp.use_dark_xtc_all.value() )

        self.edi_path    = QtGui.QLineEdit( fnm.path_dark_xtc_cond() )        
        self.edi_path.setReadOnly( True )  

        self.lab_status  = QtGui.QLabel('Status')
        self.lab_batch   = QtGui.QLabel('Batch')
        self.lab_start   = QtGui.QLabel('Start')
        self.lab_end     = QtGui.QLabel('End')
        self.lab_total   = QtGui.QLabel('Total')
        self.lab_time    = QtGui.QLabel(u'\u0394t(sec)')

        self.edi_bat_start  = QtGui.QLineEdit ( str( cp.bat_dark_start.value() ) )        
        self.edi_bat_end    = QtGui.QLineEdit ( str( cp.bat_dark_end  .value() ) )        
        self.edi_bat_total  = QtGui.QLineEdit ( str( cp.bat_dark_total.value() ) )        
        self.edi_bat_time   = QtGui.QLineEdit ( str( cp.bat_dark_time .value() ) )        
 
        self.but_path    = QtGui.QPushButton('File:')
        self.but_status  = QtGui.QPushButton('Check status')
        self.but_aver    = QtGui.QPushButton('Pedestal')
        self.but_scan    = QtGui.QPushButton('Scanner')
        self.but_browse  = QtGui.QPushButton('View')
        self.but_plot    = QtGui.QPushButton('Plot')
        self.but_remove  = QtGui.QPushButton('Remove')

        self.table_scan = GUIFilesStatusTable (parent=None, list_of_files=fnm.get_list_of_files_peds_scan())
        self.table_aver = GUIFilesStatusTable (parent=None, list_of_files=fnm.get_list_of_files_peds_aver())

        self.grid = QtGui.QGridLayout()
        self.grid_row = 0
        #self.grid.addWidget(self.tit_path,     self.grid_row,   0)
        self.grid.addWidget(self.cbx_dark,      self.grid_row,   0, 1, 6)
        self.grid.addWidget(self.but_path,      self.grid_row+1, 0)
        self.grid.addWidget(self.edi_path,      self.grid_row+1, 1, 1, 7)
        self.grid.addWidget(self.cbx_all_chunks,self.grid_row+2, 1, 1, 6)
        self.grid.addWidget(self.lab_batch,     self.grid_row+3, 0)
        self.grid.addWidget(self.lab_status,    self.grid_row+3, 1, 1, 2)
        self.grid.addWidget(self.lab_start,     self.grid_row+3, 3)
        self.grid.addWidget(self.lab_end,       self.grid_row+3, 4)
        self.grid.addWidget(self.lab_total,     self.grid_row+3, 5)
        self.grid.addWidget(self.lab_time,      self.grid_row+3, 6)
        self.grid.addWidget(self.but_scan,      self.grid_row+4, 0)
        self.grid.addWidget(self.but_status,    self.grid_row+4, 1, 2, 2)
        self.grid.addWidget(self.edi_bat_start, self.grid_row+4, 3)
        self.grid.addWidget(self.edi_bat_end,   self.grid_row+4, 4)
        self.grid.addWidget(self.edi_bat_total, self.grid_row+4, 5)
        self.grid.addWidget(self.edi_bat_time,  self.grid_row+4, 6, 1, 2)
        self.grid.addWidget(self.but_aver,      self.grid_row+5, 0)
        self.grid.addWidget(self.but_browse,    self.grid_row+5, 3) #, 1, 2)
        self.grid.addWidget(self.but_plot,      self.grid_row+5, 4)
        self.grid.addWidget(self.but_remove,    self.grid_row+5, 7)
        self.grid.addWidget(self.table_scan,    self.grid_row+6, 0, 4, 8)
        self.grid.addWidget(self.table_aver,    self.grid_row+10,0, 4, 8)

        #self.grid.setRowMinimumHeight(self.grid_row+3, 5)

        self.connect(self.but_path,      QtCore.SIGNAL('clicked()'),          self.on_but_path      )
        self.connect(self.but_status,    QtCore.SIGNAL('clicked()'),          self.on_but_status    )
        self.connect(self.but_aver,      QtCore.SIGNAL('clicked()'),          self.on_but_aver    )
        self.connect(self.but_scan,      QtCore.SIGNAL('clicked()'),          self.on_but_scan   )
        self.connect(self.but_browse,    QtCore.SIGNAL('clicked()'),          self.on_but_browse    )
        self.connect(self.but_plot,      QtCore.SIGNAL('clicked()'),          self.on_but_plot      )
        self.connect(self.but_remove,    QtCore.SIGNAL('clicked()'),          self.on_but_remove    )
        self.connect(self.edi_bat_start, QtCore.SIGNAL('editingFinished()'),  self.on_edi_bat_start )
        self.connect(self.edi_bat_end,   QtCore.SIGNAL('editingFinished()'),  self.on_edi_bat_end   )
        self.connect(self.cbx_dark,      QtCore.SIGNAL('stateChanged(int)'),  self.on_cbx           ) 
        self.connect(self.cbx_all_chunks,QtCore.SIGNAL('stateChanged(int)'),  self.on_cbx_all_chunks) 
 
        self.setLayout(self.grid)

        self.showToolTips()
        self.setStyle()
        self.setButtonState()

    #-------------------
    #  Public methods --
    #-------------------

    def connectToThread1(self):
        try : self.connect   ( cp.thread1, QtCore.SIGNAL('update(QString)'), self.check_status )
        except : logger.warning('connectToThread1 is failed', __name__)


    def disconnectFromThread1(self):
        try : self.disconnect( cp.thread1, QtCore.SIGNAL('update(QString)'), self.check_status )
        except : pass


    def showToolTips(self):
        #self           .setToolTip('Use this GUI to work with xtc file.')
        self.edi_path   .setToolTip('The path to the xtc file for processing in this GUI')
        self.but_path   .setToolTip('Push this button and select \nthe xtc file for dark run')
        self.but_status .setToolTip('Print batch job status \nin the logger')
        self.but_aver   .setToolTip('Submit job in batch for dark run \nand produces files with:\n1) pedestals \n2) RMS \n3) hot-pixel mask')
        self.but_scan   .setToolTip('Submit job in batch for dark run scanner, \ncounts number of events in the file,\n and average time interval between events')
        self.but_browse .setToolTip('View files for this GUI')
        self.but_plot   .setToolTip('Plot image and spectrum for pedestals')
        self.but_remove .setToolTip('Remove all pedestal work\nfiles for selected run')
        self.cbx_dark   .setToolTip('Check this box \nto process and use \ndark run correction')
        self.cbx_all_chunks.setToolTip('Switch between using one \nor all chunks of xtc file')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        width = 60
        self.setMinimumWidth(530)
        self.setStyleSheet(cp.styleBkgd)
        #tit0   .setStyleSheet (cp.styleTitle)

        self.cbx_all_chunks.setStyleSheet (cp.styleLabel)
        self.cbx_dark      .setStyleSheet (cp.styleLabel)
        self.lab_status    .setStyleSheet (cp.styleLabel)
        self.lab_batch     .setStyleSheet (cp.styleLabel)
        self.lab_start     .setStyleSheet (cp.styleLabel)
        self.lab_end       .setStyleSheet (cp.styleLabel)
        self.lab_total     .setStyleSheet (cp.styleLabel)
        self.lab_time      .setStyleSheet (cp.styleLabel)
        self.lab_time      .setFixedHeight(10)

        self.edi_path      .setStyleSheet (cp.styleEditInfo) # cp.styleEditInfo
        self.edi_path      .setAlignment  (QtCore.Qt.AlignRight)

        self.edi_bat_start .setStyleSheet(cp.styleEdit)
        self.edi_bat_end   .setStyleSheet(cp.styleEdit)
        self.edi_bat_total .setStyleSheet(cp.styleEditInfo)
        self.edi_bat_time  .setStyleSheet(cp.styleEditInfo)

        self.edi_bat_start .setFixedWidth(width)
        self.edi_bat_end   .setFixedWidth(width)
        self.edi_bat_total .setFixedWidth(width)
        self.edi_bat_time  .setMinimumWidth(140)

        self.edi_bat_start .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_end   .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_total .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_time  .setAlignment(QtCore.Qt.AlignLeft)

        self.edi_bat_total .setReadOnly( True ) 
        self.edi_bat_time  .setReadOnly( True ) 

        self.but_path   .setStyleSheet (cp.styleButton)
        self.but_status .setStyleSheet (cp.styleButton)
        self.but_aver   .setStyleSheet (cp.styleButton) 
        self.but_scan   .setStyleSheet (cp.styleButton) 
        self.but_browse .setStyleSheet (cp.styleButton) 
        self.but_plot   .setStyleSheet (cp.styleButton) 
        self.but_remove .setStyleSheet (cp.styleButtonBad) 
  
        self.but_path   .setFixedWidth(width)
        self.but_aver   .setFixedWidth(width)
        self.but_scan   .setFixedWidth(width)
        self.but_plot   .setFixedWidth(width)
        self.but_browse .setFixedWidth(width) 
        self.but_remove .setFixedWidth(width)
        #self.but_status .setFixedWidth(width)

        self.on_but_status()
    
    #def setParent(self,parent) :
    #    self.parent = parent

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

        #try    : cp.plotimgspe.close()
        #except : pass

        try    : cp.guifilebrowser.close()
        except : pass

        #try    : del cp.guidark # GUIDark
        #except : pass # silently ignore


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def on_but_path(self):
        logger.debug('Dark file browser', __name__ )
        path = str(self.edi_path.text())        
        path = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',path) )
        dname, fname = os.path.split(path)

        if dname == '' or fname == '' :
            logger.info('Input directiry name or file name is empty... keep file path unchanged...')
            return

        cp.in_dir_dark .setValue(dname)
        cp.in_file_dark.setValue(fname)
        #self.edi_path.setText(path)
        self.edi_path.setText( fnm.path_dark_xtc_cond() )
        logger.info('selected file: ' + str(fnm.path_dark_xtc()), __name__ )
        self.set_default_pars()
        blp.parse_batch_log_peds_scan()
        self.set_fields()


    def set_default_pars(self):
        cp.bat_dark_start .setDefault()
        cp.bat_dark_end   .setDefault()
        cp.bat_dark_total .setDefault()
        cp.bat_dark_time  .setDefault()
        cp.bat_dark_dt_ave.setDefault()
        cp.bat_dark_dt_rms.setDefault()


    def set_fields(self):
        self.edi_bat_start.setText( str( cp.bat_dark_start.value() ) )        
        self.edi_bat_end  .setText( str( cp.bat_dark_end  .value() ) )        
        self.edi_bat_total.setText( str( cp.bat_dark_total.value() ) )        
        self.edi_bat_time .setText( str( cp.bat_dark_dt_ave.value() ) + u'\u00B1' + str( cp.bat_dark_dt_rms.value() ) )        
        self.set_style_for_edi_bat_end()


    def set_style_for_edi_bat_end(self):
        if(cp.bat_dark_end.value() == cp.bat_dark_end.value_def()) :
            self.edi_bat_end.setStyleSheet(cp.styleEditBad)
        else :
            self.edi_bat_end.setStyleSheet(cp.styleEdit)


    def on_but_aver(self):
        logger.debug('on_but_aver', __name__)

        if(cp.bat_dark_end.value() == cp.bat_dark_end.value_def()) :
            self.edi_bat_end.setStyleSheet(cp.styleEditBad)
            logger.warning('JOB IS NOT SUBMITTED !!!\nFirst, set the number of events for pedestal avaraging.', __name__)
            return
        else :
            self.edi_bat_end.setStyleSheet(cp.styleEdit)
        bjpeds.submit_batch_for_peds_aver()
        self.connectToThread1()


    def on_but_scan(self):
        logger.debug('on_but_scan', __name__)
        bjpeds.submit_batch_for_peds_scan()
        self.connectToThread1()


    def on_but_status(self):
        logger.debug('on_but_status', __name__)
        logger.info('='*110, __name__)
        bjpeds.check_work_files_for_pedestals()
        bjpeds.check_batch_job_for_peds_scan()
        bjpeds.check_batch_job_for_peds_aver()

        self.check_status()


    def check_status(self):
        self.check_status_scan()
        self.check_status_aver()
        if cp.procDarkStatus == 0 : self.disconnectFromThread1()


    def check_status_scan(self):
        bstatus, bstatus_str = bjpeds.status_batch_job_for_peds_scan()
        fstatus, fstatus_str = bjpeds.status_for_peds_scan_files()
        msg = 'Scan: ' + bstatus_str + '   ' + fstatus_str
        self.set_for_status(fstatus, msg, self.but_scan, self.table_scan)
        if fstatus :
            blp.parse_batch_log_peds_scan()
            self.set_fields()
        if cp.procDarkStatus & 1 : logger.info(msg, __name__) 


    def check_status_aver(self):
        bstatus, bstatus_str = bjpeds.status_batch_job_for_peds_aver()
        fstatus, fstatus_str = bjpeds.status_for_peds_aver_files()
        msg = 'Peds: ' + bstatus_str + '   ' + fstatus_str
        self.set_for_status(fstatus, msg, self.but_aver, self.table_aver)
        if cp.procDarkStatus & 2 : logger.info(msg, __name__) 


    def set_for_status(self, status, msg, but, table):
        """Sets the status string above the table and color of the submit button"""
        if status :
            but  .setStyleSheet(cp.styleButtonGood)
            table.setStatus(0, msg)
        else :
            but  .setStyleSheet(cp.styleButtonBad)
            table.setStatus(2, msg)


    def on_edi_bat_start(self):
        if(not cp.bat_dark_is_used.value()) : return
        cp.bat_dark_start.setValue( int(self.edi_bat_start.displayText()) )
        logger.info('Set bat_dark_start =' + str(cp.bat_dark_start.value()), __name__)

    def on_edi_bat_end(self):
        if(not cp.bat_dark_is_used.value()) : return
        cp.bat_dark_end.setValue( int(self.edi_bat_end.displayText()) )
        logger.info('Set bat_dark_end =' + str(cp.bat_dark_end.value()), __name__)
        self.set_fields()


    def on_but_browse(self):
        logger.debug('on_but_browse', __name__)
        try    :
            cp.guifilebrowser.close()
            self.but_browse.setStyleSheet(cp.styleButtonBad)
        except :
            self.but_browse.setStyleSheet(cp.styleButtonGood)
            cp.guifilebrowser = GUIFileBrowser(None, fnm.get_list_of_files_pedestals(), selected_file=fnm.path_pedestals_ave())
            cp.guifilebrowser.move(cp.guimain.pos().__add__(QtCore.QPoint(740,140))) # open window with offset w.r.t. parent
            cp.guifilebrowser.show()

    def on_but_plot(self):
        logger.debug('on_but_plot', __name__)
        try :
            cp.plotimgspe.close()
            try    : del cp.plotimgspe
            except : pass
        except :
            arr = gu.get_array_from_file( fnm.path_pedestals_ave() )
            if arr is None : return
            #print arr.shape,'\n', arr
            cp.plotimgspe = PlotImgSpe(None, arr, ifname=fnm.path_pedestals_ave(), ofname=fnm.path_peds_aver_plot())
            #cp.plotimgspe.setParent(self)
            cp.plotimgspe.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.plotimgspe.show()
            #but.setStyleSheet(cp.styleButtonGood)

    def on_but_remove(self):
        logger.debug('on_but_remove', __name__)
        bjpeds.remove_files_pedestals()
        self.on_but_status()

    def on_cbx(self):
        #if self.cbx_dark.hasFocus() :
        par = cp.bat_dark_is_used
        par.setValue( self.cbx_dark.isChecked() )
        msg = 'on_cbx - set status of parameter bat_dark_is_used: ' + str(par.value())
        logger.info(msg, __name__ )
        self.setButtonState()

    def on_cbx_all_chunks(self):
        #if self.cbx_all_chunks.hasFocus() :
        par = cp.use_dark_xtc_all
        par.setValue( self.cbx_all_chunks.isChecked() )
        msg = 'on_cbx - set status of parameter use_dark_xtc_all: ' + str(par.value())
        logger.info(msg, __name__ )
        self.setButtonState()
        self.edi_path.setText( fnm.path_dark_xtc_cond() )

    def setButtonState(self):

        is_used = cp.bat_dark_is_used.value()

        self.but_path   .setEnabled(is_used)
        self.but_status .setEnabled(is_used)
        self.but_aver   .setEnabled(is_used)
        self.but_scan   .setEnabled(is_used)
        self.but_browse .setEnabled(is_used)
        self.but_plot   .setEnabled(is_used)
        self.but_remove .setEnabled(is_used)
        self.cbx_all_chunks.setEnabled(is_used)
        
        #self.but_path   .setFlat(not is_used)
        #self.but_status .setFlat(not is_used)
        #self.but_aver   .setFlat(not is_used)
        #self.but_scan   .setFlat(not is_used)
        #self.but_browse .setFlat(not is_used)
        #self.but_plot   .setFlat(not is_used)
        #self.but_remove .setFlat(not is_used)

        if is_used :
            self.edi_bat_start.setStyleSheet(cp.styleEdit)
            self.edi_bat_end  .setStyleSheet(cp.styleEdit)
        else :
            self.edi_bat_start.setStyleSheet(cp.styleEditInfo)
            self.edi_bat_end  .setStyleSheet(cp.styleEditInfo)

        self.edi_bat_start.setReadOnly( not is_used ) 
        self.edi_bat_end  .setReadOnly( not is_used ) 

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIDark ()
    widget.show()
    app.exec_()

#-----------------------------
