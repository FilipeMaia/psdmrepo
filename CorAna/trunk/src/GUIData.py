#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIData...
#
#------------------------------------------------------------------------

"""GUI sets the data file"""

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
from PlotTime               import *
import GlobalUtils          as     gu
from BatchLogParser         import blp
from GUIFileBrowser         import *
from BatchJobData           import bjdata
from EventTimeRecords       import *
from GUIFilesStatusTable    import *

#---------------------
#  Class definition --
#---------------------
class GUIData ( QtGui.QWidget ) :
    """GUI sets the data file"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 530, 30)
        self.setWindowTitle('Data file')
        self.setFrame()

        self.cbx_data = QtGui.QCheckBox('Activate / protect buttons', self)
        self.cbx_data.setChecked( cp.is_active_data_gui.value() )

        self.cbx_all_chunks = QtGui.QCheckBox('Use all xtc chunks', self)
        self.cbx_all_chunks.setChecked( cp.use_data_xtc_all.value() )

        self.edi_path = QtGui.QLineEdit( fnm.path_data_xtc_cond() )        
        self.edi_path.setReadOnly( True )   

        self.lab_status = QtGui.QLabel('Status')
        self.lab_batch  = QtGui.QLabel('Batch')
        self.lab_start  = QtGui.QLabel('Start')
        self.lab_end    = QtGui.QLabel('End')
        self.lab_total  = QtGui.QLabel('Total')
        self.lab_time   = QtGui.QLabel(u'\u0394t(sec)')

        self.edi_bat_start  = QtGui.QLineEdit ( str( cp.bat_data_start.value() ) )        
        self.edi_bat_end    = QtGui.QLineEdit ( str( cp.bat_data_end  .value() ) )        
        self.edi_bat_total  = QtGui.QLineEdit ( str( cp.bat_data_total.value() ) )        
        self.edi_bat_time   = QtGui.QLineEdit ( str( cp.bat_data_time .value() ) )        
 
        self.but_path   = QtGui.QPushButton('File:')
        self.but_plot   = QtGui.QPushButton('img-Plot')
        self.but_tspl   = QtGui.QPushButton('t-Plot')
        self.but_brow   = QtGui.QPushButton('View')
        self.but_scan   = QtGui.QPushButton('Scan')
        self.but_aver   = QtGui.QPushButton('Average')
        self.but_status = QtGui.QPushButton('Check status')
        self.but_remove = QtGui.QPushButton('Remove')

        self.table_scan = GUIFilesStatusTable (parent=self, list_of_files=fnm.get_list_of_files_data_scan())
        self.table_aver = GUIFilesStatusTable (parent=self, list_of_files=fnm.get_list_of_files_data_aver())
        #self.table_aver = GUIFilesStatusTable (parent=self, list_of_files=fnm.get_list_of_files_data_aver_short())
        #self.table_scan.setMinimumHeight(285)
        #self.table_scan.table.setFixedWidth(self.table_scan.table.horizontalHeader().length() + 4)
        #self.table_scan.table.setFixedHeight(self.table_scan.table.verticalHeader().length() + 29)

        self.grid = QtGui.QGridLayout()
        self.grid_row = 1
        self.grid.addWidget(self.cbx_data,      self.grid_row,   0, 1, 6)          

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
        self.grid.addWidget(self.but_brow,      self.grid_row+5, 3)
        self.grid.addWidget(self.but_plot,      self.grid_row+5, 4)
        self.grid.addWidget(self.but_tspl,      self.grid_row+5, 5)
        self.grid.addWidget(self.but_remove,    self.grid_row+5, 7)

        self.grid.addWidget(self.table_scan,    self.grid_row+6, 0, 7, 8)
        self.grid.addWidget(self.table_aver,    self.grid_row+13,0, 4, 8)

        self.connect(self.but_path,      QtCore.SIGNAL('clicked()'),         self.on_but_path )
        self.connect(self.but_plot,      QtCore.SIGNAL('clicked()'),         self.on_but_plot )
        self.connect(self.but_tspl,      QtCore.SIGNAL('clicked()'),         self.on_but_tspl )
        self.connect(self.but_brow,      QtCore.SIGNAL('clicked()'),         self.on_but_brow )
        self.connect(self.but_aver,      QtCore.SIGNAL('clicked()'),         self.on_but_aver )
        self.connect(self.but_scan,      QtCore.SIGNAL('clicked()'),         self.on_but_scan )
        self.connect(self.but_status,    QtCore.SIGNAL('clicked()'),         self.on_but_status )
        self.connect(self.but_remove,    QtCore.SIGNAL('clicked()'),         self.on_but_remove )
        self.connect(self.edi_bat_start, QtCore.SIGNAL('editingFinished()'), self.on_edi_bat_start )
        self.connect(self.edi_bat_end,   QtCore.SIGNAL('editingFinished()'), self.on_edi_bat_end   )
        self.connect(self.cbx_data,      QtCore.SIGNAL('stateChanged(int)'), self.on_cbx ) 
        self.connect(self.cbx_all_chunks,QtCore.SIGNAL('stateChanged(int)'), self.on_cbx_all_chunks ) 
  
        self.setLayout(self.grid)

        self.showToolTips()
        self.setStyle()
        self.setButtonState()
        #self.connectToThread1()


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
        #self          .setToolTip('Use this GUI to work with xtc file.')
        self.edi_path  .setToolTip('The path to the xtc data file for processing')
        self.but_path  .setToolTip('Push this button and select the xtc data file')
        self.but_plot  .setToolTip('Plot image and spectrum for averaged data image')
        self.but_tspl  .setToolTip('Plot for time stamps quality check')
        self.but_brow  .setToolTip('View files for this procedure')
        self.but_scan  .setToolTip('Scan entire run and \n1) count number of events' + \
                                   '\n2) save time stamps' + \
                                   '\n3) save intensity monitors')
        self.but_aver  .setToolTip('1) Average image for selected event range' + \
                                   '\n2) Evaluate saturated pixel mask')
        self.but_status.setToolTip('Print batch job status \nin the logger')
        self.but_remove.setToolTip('Remove all data work\nfiles for selected run')
        self.cbx_data  .setToolTip('Lock / unlock buttons and fields')
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

        self.cbx_data      .setStyleSheet (cp.styleLabel)
        self.cbx_all_chunks.setStyleSheet (cp.styleLabel)
        self.lab_batch     .setStyleSheet (cp.styleLabel)
        self.lab_status    .setStyleSheet (cp.styleLabel)
        self.lab_start     .setStyleSheet (cp.styleLabel)
        self.lab_end       .setStyleSheet (cp.styleLabel)
        self.lab_total     .setStyleSheet (cp.styleLabel)
        self.lab_time      .setStyleSheet (cp.styleLabel)
        self.lab_time      .setFixedHeight(10)

        self.edi_path      .setStyleSheet (cp.styleEditInfo)
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

        self.but_path  .setStyleSheet (cp.styleButton)
        self.but_plot  .setStyleSheet (cp.styleButton) 
        self.but_tspl  .setStyleSheet (cp.styleButton) 
        self.but_brow  .setStyleSheet (cp.styleButton) 
        self.but_scan  .setStyleSheet (cp.styleButton) 
        self.but_aver  .setStyleSheet (cp.styleButton) 
        self.but_status.setStyleSheet (cp.styleButton)
        self.but_remove.setStyleSheet (cp.styleButtonBad) 
     
        self.but_path  .setFixedWidth (width)
        self.but_plot  .setFixedWidth (width)
        self.but_tspl  .setFixedWidth (width)
        self.but_brow  .setFixedWidth (width)
        self.but_scan  .setFixedWidth (width)
        self.but_aver  .setFixedWidth (width)
        self.but_remove.setFixedWidth(width)

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

        #try    : del cp.guidata # GUIData
        #except : pass # silently ignore


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def on_but_path(self):
        logger.debug('Data file browser', __name__ )
        path = str(self.edi_path.text())        
        path = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',path) )
        dname, fname = os.path.split(path)

        if dname == '' or fname == '' :
            logger.info('Input directiry name or file name is empty... keep file path unchanged...')
            return

        cp.in_dir_data .setValue(dname)
        cp.in_file_data.setValue(fname)
        #self.edi_path.setText(path)
        self.edi_path.setText( fnm.path_data_xtc_cond() )
        logger.info('selected file: ' + str(fnm.path_data_xtc()), __name__ )
        self.set_default_pars()
        blp.parse_batch_log_data_scan()
        self.set_fields()


    def set_default_pars(self):
        cp.bat_data_start .setDefault()
        cp.bat_data_end   .setDefault()
        cp.bat_data_total .setDefault()
        cp.bat_data_time  .setDefault()
        cp.bat_data_dt_ave.setDefault()
        cp.bat_data_dt_rms.setDefault()


    def set_fields(self):
        self.edi_bat_start.setText( str( cp.bat_data_start.value() ) )        
        self.edi_bat_end  .setText( str( cp.bat_data_end  .value() ) )        
        self.edi_bat_total.setText( str( cp.bat_data_total.value() ) )        
        self.edi_bat_time .setText( str( cp.bat_data_dt_ave.value() ) + u'\u00B1' + str( cp.bat_data_dt_rms.value() ) )        
        self.set_style_for_edi_bat_end()


    def set_style_for_edi_bat_end(self):
        if(cp.bat_data_end.value() == cp.bat_data_end.value_def()) :
            self.edi_bat_end.setStyleSheet(cp.styleEditBad)
        else :
            self.edi_bat_end.setStyleSheet(cp.styleEdit)


    def on_but_aver(self):
        logger.debug('on_but_aver', __name__)

        if(cp.bat_data_end.value() == cp.bat_data_end.value_def()) :
            self.edi_bat_end.setStyleSheet(cp.styleEditBad)
            logger.warning('JOB IS NOT SUBMITTED !!!\nFirst, set the number of events for data avaraging.', __name__)
            return
        else :
            self.edi_bat_end.setStyleSheet(cp.styleEdit)
        bjdata.submit_batch_for_data_aver()
        self.connectToThread1()


    def on_but_scan(self):
        logger.debug('on_but_scan', __name__)
        bjdata.submit_batch_for_data_scan()
        self.connectToThread1()


    def on_but_status(self):
        logger.debug('on_but_status', __name__)
        logger.info('='*110, __name__)
        bjdata.check_work_files_for_data_aver()
        bjdata.check_batch_job_for_data_scan()
        bjdata.check_batch_job_for_data_aver()

        self.check_status()


    def check_status(self):
        self.check_status_scan()
        self.check_status_aver()
        if cp.procDataStatus == 0 : self.disconnectFromThread1()


    def check_status_scan(self):
        bstatus, bstatus_str = bjdata.status_batch_job_for_data_scan()
        fstatus, fstatus_str = bjdata.status_for_data_scan_files()
        msg = 'Scan: ' + bstatus_str + '   ' + fstatus_str
        self.set_for_status(fstatus, msg, self.but_scan, self.table_scan)
        if fstatus :
            blp.parse_batch_log_data_scan()
            self.set_fields()
        if cp.procDataStatus & 1 : logger.info(msg, __name__) 


    def check_status_aver(self):
        bstatus, bstatus_str = bjdata.status_batch_job_for_data_aver()
        fstatus, fstatus_str = bjdata.status_for_data_aver_files()
        msg = 'Aver: ' + bstatus_str + '   ' + fstatus_str
        self.set_for_status(fstatus, msg, self.but_aver, self.table_aver)
        if fstatus :
            blp.parse_batch_log_data_aver()
            self.set_fields()
        if cp.procDataStatus & 2 : logger.info(msg, __name__) 


    def set_for_status(self, status, msg, but, table):
        if status :
            but  .setStyleSheet(cp.styleButtonGood)
            table.setStatus(0, msg)
        else :
            but  .setStyleSheet(cp.styleButtonBad)
            table.setStatus(2, msg)


    def on_edi_bat_start(self):
        if(not cp.is_active_data_gui.value()) : return
        cp.bat_data_start.setValue( int(self.edi_bat_start.displayText()) )
        logger.info('Set bat_data_start =' + str(cp.bat_data_start.value()), __name__)


    def on_edi_bat_end(self):
        if(not cp.is_active_data_gui.value()) : return
        cp.bat_data_end.setValue( int(self.edi_bat_end.displayText()) )
        logger.info('Set bat_data_end =' + str(cp.bat_data_end.value()), __name__)
        self.set_fields()


    def on_but_brow (self):       
        logger.debug('on_but_brow', __name__)
        try    :
            cp.guifilebrowser.close()
            self.but_brow.setStyleSheet(cp.styleButtonBad)
        except :
            self.but_brow.setStyleSheet(cp.styleButtonGood)
            cp.guifilebrowser = GUIFileBrowser(None, fnm.get_list_of_files_data(), fnm.path_data_ave())
            cp.guifilebrowser.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.guifilebrowser.show()


    def on_but_remove(self):
        logger.debug('on_but_remove', __name__)
        bjdata.remove_files_data_aver()
        self.on_but_status()


    def on_but_plot(self):
        logger.debug('on_but_plot', __name__)
        try :
            cp.plotimgspe.close()
            try    : del cp.plotimgspe
            except : pass
        except :
            ifname =  fnm.path_data_raw_ave()
            arr = gu.get_array_from_file(ifname)
            if arr is None : return
            logger.debug('Array shape: ' + str(arr.shape), __name__)
            cp.plotimgspe = PlotImgSpe(None, arr, ifname, ofname=fnm.path_data_aver_plot())
            cp.plotimgspe.move(cp.guimain.pos().__add__(QtCore.QPoint(740,140)))
            cp.plotimgspe.show()


    def on_but_tspl(self):
        logger.debug('on_but_tspl', __name__)
        try :
            cp.plottime.close()
            try    : del cp.plottime
            except : pass
        except :
            if not os.path.lexists(fnm.path_data_scan_tstamp_list()) :
                msg = 'Requested the time plot for NON-EXISTANT FILE: ' \
                    + str(fnm.path_data_scan_tstamp_list()) + '\nUse "Scan" first...'
                logger.warning(msg, __name__ )
                return
            
            cp.plottime = PlotTime(None, ifname = fnm.path_data_scan_tstamp_list(),\
                                   ofname = fnm.path_data_time_plot())
            cp.plottime.move(cp.guimain.pos().__add__(QtCore.QPoint(760,160)))
            cp.plottime.show()


    def on_cbx(self):
        #if self.cbx_data .hasFocus() :
        par = cp.is_active_data_gui
        par.setValue( self.cbx_data.isChecked() )
        msg = 'on_cbx - set status of parameter is_active_data_gui: ' + str(par.value())
        logger.info(msg, __name__ )
        self.setButtonState()

    def on_cbx_all_chunks(self):
        #if self.cbx_all_chunks .hasFocus() :
        par = cp.use_data_xtc_all
        par.setValue( self.cbx_all_chunks.isChecked() )
        msg = 'on_cbx - set status of parameter use_data_xtc_all: ' + str(par.value())
        logger.info(msg, __name__ )
        self.setButtonState()
        self.edi_path.setText( fnm.path_data_xtc_cond() )

    def setButtonState(self):
        is_active = cp.is_active_data_gui.value()

        self.but_path  .setEnabled( is_active)
        self.but_plot  .setEnabled( is_active)
        self.but_tspl  .setEnabled( is_active)
        self.but_brow  .setEnabled( is_active)
        self.but_scan  .setEnabled( is_active)
        self.but_aver  .setEnabled( is_active)
        self.but_remove.setEnabled( is_active)
        self.but_status.setEnabled( is_active)

        self.cbx_all_chunks.setEnabled( is_active) 

        #self.but_path  .setFlat(not is_active)
        #self.but_plot  .setFlat(not is_active)
        #self.but_tspl  .setFlat(not is_active) 
        #self.but_brow  .setFlat(not is_active)
        #self.but_scan  .setFlat(not is_active)
        #self.but_aver  .setFlat(not is_active)
        #self.but_remove.setFlat(not is_active)
        #self.but_status.setFlat(not is_active)

        if is_active :
            self.edi_bat_start.setStyleSheet(cp.styleEdit)
            self.edi_bat_end  .setStyleSheet(cp.styleEdit)
            #self.cbx_data.setIcon(cp.icon_unlock) # (QtGui.QIcon())            
        else :
            self.edi_bat_start.setStyleSheet(cp.styleEditInfo)
            self.edi_bat_end  .setStyleSheet(cp.styleEditInfo)
            #self.cbx_data.setIcon(cp.icon_lock)            

        self.edi_bat_start.setReadOnly( not is_active ) 
        self.edi_bat_end  .setReadOnly( not is_active ) 


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIData ()
    widget.show()
    app.exec_()

#-----------------------------
