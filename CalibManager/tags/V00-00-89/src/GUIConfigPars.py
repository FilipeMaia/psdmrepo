#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIConfigPars...
#
#------------------------------------------------------------------------

"""GUI for Work/Result directories"""

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

from CalibManager.Frame     import Frame
from ConfigParametersForApp import cp
from Logger                 import logger
import GlobalUtils          as     gu

#---------------------
#  Class definition --
#---------------------
#class GUIConfigPars ( QtGui.QWidget ) :
class GUIConfigPars ( Frame ) :
    """GUI for management of configuration parameters"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        #QtGui.QWidget.__init__(self, parent)
        Frame.__init__(self, parent, mlw=1)

        self.setGeometry(200, 400, 500, 200)
        self.setWindowTitle('Configuration Parameters')
        #self.setFrame()

        #self.tit_dir_work = QtGui.QLabel('Parameters:')

        self.edi_dir_work = QtGui.QLineEdit( cp.dir_work.value() )        
        self.but_dir_work = QtGui.QPushButton('Dir work:')
        self.edi_dir_work.setReadOnly( True )  

        self.edi_dir_results = QtGui.QLineEdit( cp.dir_results.value() )        
        self.but_dir_results = QtGui.QPushButton('Dir results:')
        self.edi_dir_results.setReadOnly( True )  

        self.lab_fname_prefix = QtGui.QLabel('File prefix:')
        self.edi_fname_prefix = QtGui.QLineEdit( cp.fname_prefix.value() )        

        self.lab_bat_queue  = QtGui.QLabel('Queue:') 
        self.box_bat_queue  = QtGui.QComboBox( self ) 
        self.box_bat_queue.addItems(cp.list_of_queues)
        self.box_bat_queue.setCurrentIndex( cp.list_of_queues.index(cp.bat_queue.value()) )

        self.lab_dark_start = QtGui.QLabel('Event start:') 
        self.lab_dark_end   = QtGui.QLabel('end:') 
        self.lab_dark_scan  = QtGui.QLabel('scan:') 
        self.lab_rms_thr_min= QtGui.QLabel('Threshold RMS MIN, ADU:') 
        self.lab_rms_thr    = QtGui.QLabel('MAX:') 
        self.lab_min_thr    = QtGui.QLabel('Threshold MIN, ADU:') 
        self.lab_max_thr    = QtGui.QLabel('MAX:') 
        self.lab_dark_sele  = QtGui.QLabel('Selector evt code:') 

        self.but_show_vers  = QtGui.QPushButton('Soft Vers')
        self.but_lsf_status = QtGui.QPushButton('LSF status')

        self.edi_dark_start = QtGui.QLineEdit( str( cp.bat_dark_start.value() ) )
        self.edi_dark_end   = QtGui.QLineEdit( str( cp.bat_dark_end.value()) )
        self.edi_dark_scan  = QtGui.QLineEdit( str( cp.bat_dark_scan.value()) )
        self.edi_dark_sele  = QtGui.QLineEdit( str( cp.bat_dark_sele.value()) )
        self.edi_rms_thr_min= QtGui.QLineEdit( str( cp.mask_rms_thr_min.value()) )
        self.edi_rms_thr    = QtGui.QLineEdit( str( cp.mask_rms_thr.value()) )
        self.edi_min_thr    = QtGui.QLineEdit( str( cp.mask_min_thr.value()) )
        self.edi_max_thr    = QtGui.QLineEdit( str( cp.mask_max_thr.value()) )

        self.edi_dark_start .setValidator(QtGui.QIntValidator(0,9999999,self))
        self.edi_dark_end   .setValidator(QtGui.QIntValidator(1,9999999,self))
        self.edi_dark_scan  .setValidator(QtGui.QIntValidator(1,9999999,self))
        self.edi_dark_sele  .setValidator(QtGui.QIntValidator(-256,256,self))
        self.edi_rms_thr_min.setValidator(QtGui.QDoubleValidator(0,65000,3,self))
        self.edi_rms_thr    .setValidator(QtGui.QDoubleValidator(0,65000,3,self))
        self.edi_min_thr    .setValidator(QtGui.QDoubleValidator(0,65000,3,self))
        self.edi_max_thr    .setValidator(QtGui.QDoubleValidator(0,65000,3,self))
        #self.edi_events.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]\\d{0,3}|end$"),self))

        self.cbx_deploy_hotpix = QtGui.QCheckBox('Deploy hotpix mask')
        self.cbx_deploy_hotpix.setChecked( cp.dark_deploy_hotpix.value() )

        self.grid = QtGui.QGridLayout()
        self.grid_row = 0
        #self.grid.addWidget(self.tit_dir_work,      self.grid_row,   0, 1, 9)
        self.grid.addWidget(self.but_dir_work,      self.grid_row+1, 0)
        self.grid.addWidget(self.edi_dir_work,      self.grid_row+1, 1, 1, 8)
        self.grid.addWidget(self.but_dir_results,   self.grid_row+2, 0)
        self.grid.addWidget(self.edi_dir_results,   self.grid_row+2, 1, 1, 8)
        self.grid.addWidget(self.lab_fname_prefix,  self.grid_row+3, 0)
        self.grid.addWidget(self.edi_fname_prefix,  self.grid_row+3, 1, 1, 4)
        self.grid.addWidget(self.lab_bat_queue,     self.grid_row+4, 0)
        self.grid.addWidget(self.box_bat_queue,     self.grid_row+4, 1)
        self.grid.addWidget(self.but_lsf_status,    self.grid_row+4, 4) #, 1, 2)
        self.grid.addWidget(self.but_show_vers,     self.grid_row+4, 5) #, 1, 2)
        self.grid.addWidget(self.lab_dark_start,    self.grid_row+5, 0)
        self.grid.addWidget(self.edi_dark_start,    self.grid_row+5, 1)
        self.grid.addWidget(self.lab_dark_end,      self.grid_row+5, 3)
        self.grid.addWidget(self.edi_dark_end,      self.grid_row+5, 4)
        self.grid.addWidget(self.lab_dark_scan,     self.grid_row+5, 5)
        self.grid.addWidget(self.edi_dark_scan,     self.grid_row+5, 6)
        self.grid.addWidget(self.lab_rms_thr_min,   self.grid_row+6, 0)
        self.grid.addWidget(self.edi_rms_thr_min,   self.grid_row+6, 1, 1, 4)
        self.grid.addWidget(self.lab_rms_thr,       self.grid_row+6, 3)
        self.grid.addWidget(self.edi_rms_thr,       self.grid_row+6, 4, 1, 4)
        self.grid.addWidget(self.cbx_deploy_hotpix, self.grid_row+6, 5, 1, 4)
        self.grid.addWidget(self.lab_min_thr,       self.grid_row+7, 0)
        self.grid.addWidget(self.edi_min_thr,       self.grid_row+7, 1, 1, 2)
        self.grid.addWidget(self.lab_max_thr,       self.grid_row+7, 3)
        self.grid.addWidget(self.edi_max_thr,       self.grid_row+7, 4, 1, 2)

        self.grid.addWidget(self.lab_dark_sele,     self.grid_row+7, 5)
        self.grid.addWidget(self.edi_dark_sele,     self.grid_row+7, 6)

        #self.setLayout(self.grid)

        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addLayout(self.grid)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.connect( self.but_dir_work,     QtCore.SIGNAL('clicked()'),          self.onButDirWork )
        self.connect( self.but_dir_results,  QtCore.SIGNAL('clicked()'),          self.onButDirResults )
        self.connect( self.box_bat_queue,    QtCore.SIGNAL('currentIndexChanged(int)'), self.onBoxBatQueue )
        self.connect( self.edi_fname_prefix, QtCore.SIGNAL('editingFinished ()'), self.onEditPrefix )
        self.connect( self.edi_dark_start,   QtCore.SIGNAL('editingFinished()'),  self.onEdiDarkStart )
        self.connect( self.edi_dark_end,     QtCore.SIGNAL('editingFinished()'),  self.onEdiDarkEnd )
        self.connect( self.edi_dark_scan,    QtCore.SIGNAL('editingFinished()'),  self.onEdiDarkScan )
        self.connect( self.edi_dark_sele,    QtCore.SIGNAL('editingFinished()'),  self.onEdiDarkSele )
        self.connect( self.edi_rms_thr_min,  QtCore.SIGNAL('editingFinished()'),  self.onEdiRmsThrMin )
        self.connect( self.edi_rms_thr,      QtCore.SIGNAL('editingFinished()'),  self.onEdiRmsThr )
        self.connect( self.edi_min_thr,      QtCore.SIGNAL('editingFinished()'),  self.onEdiMinThr )
        self.connect( self.edi_max_thr,      QtCore.SIGNAL('editingFinished()'),  self.onEdiMaxThr )
        self.connect( self.cbx_deploy_hotpix,QtCore.SIGNAL('stateChanged(int)'),  self.on_cbx ) 
        self.connect( self.but_show_vers,    QtCore.SIGNAL('clicked()'),          self.onButShowVers )
        self.connect( self.but_lsf_status,   QtCore.SIGNAL('clicked()'),          self.onButLsfStatus )
 
        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        self.edi_dir_work    .setToolTip('Click on "Dir work:" button\nto change the directory')
        self.but_dir_work    .setToolTip('Click on this button\nand select the directory')
        self.edi_dir_results .setToolTip('Click on "Dir results:" button\nto change the directory')
        self.but_dir_results .setToolTip('Click on this button\nand select the directory')
        self.edi_fname_prefix.setToolTip('Edit the common file prefix in this field')
        self.but_show_vers   .setToolTip('Show current package tags')
        self.but_lsf_status  .setToolTip('Show LSF status')
        self.edi_dark_sele   .setToolTip('Selector event code;\n0=off, +N=select, -N=discard')

#    def setFrame(self):
#        self.frame = QtGui.QFrame(self)
#        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
#        self.frame.setLineWidth(0)
#        self.frame.setMidLineWidth(1)
#        self.frame.setGeometry(self.rect())
#        #self.frame.setVisible(False)


    def setStyle(self):
        self.                 setStyleSheet (cp.styleBkgd)
        self.setMinimumSize(500,240)
        self.setMaximumSize(700,240)

        #self.tit_dir_work     .setStyleSheet (cp.styleTitle)
        self.edi_dir_work     .setStyleSheet (cp.styleEditInfo)       
        self.but_dir_work     .setStyleSheet (cp.styleButton) 
        self.edi_dir_results  .setStyleSheet (cp.styleEditInfo)       
        self.but_dir_results  .setStyleSheet (cp.styleButton) 
        self.lab_fname_prefix .setStyleSheet (cp.styleLabel)
        self.edi_fname_prefix .setStyleSheet (cp.styleEdit)
        self.lab_bat_queue    .setStyleSheet (cp.styleLabel)
        self.lab_dark_start   .setStyleSheet (cp.styleLabel)
        self.lab_dark_end     .setStyleSheet (cp.styleLabel)
        self.lab_dark_scan    .setStyleSheet (cp.styleLabel)
        self.lab_dark_sele    .setStyleSheet (cp.styleLabel)
        self.lab_rms_thr_min  .setStyleSheet (cp.styleLabel)
        self.lab_rms_thr      .setStyleSheet (cp.styleLabel)
        self.lab_min_thr      .setStyleSheet (cp.styleLabel)
        self.lab_max_thr      .setStyleSheet (cp.styleLabel)
        self.cbx_deploy_hotpix.setStyleSheet (cp.styleLabel)
        self.but_show_vers    .setStyleSheet (cp.styleButton) 
        self.but_lsf_status   .setStyleSheet (cp.styleButton) 

        #self.tit_dir_work    .setAlignment (QtCore.Qt.AlignLeft)
        self.edi_dir_work    .setAlignment (QtCore.Qt.AlignRight)
        self.edi_dir_results .setAlignment (QtCore.Qt.AlignRight)
        self.lab_fname_prefix.setAlignment (QtCore.Qt.AlignRight)
        self.lab_bat_queue   .setAlignment (QtCore.Qt.AlignRight)
        self.lab_dark_start  .setAlignment (QtCore.Qt.AlignRight)
        self.lab_dark_end    .setAlignment (QtCore.Qt.AlignRight)
        self.lab_dark_scan   .setAlignment (QtCore.Qt.AlignRight)
        self.lab_dark_sele   .setAlignment (QtCore.Qt.AlignRight)
        self.lab_rms_thr_min .setAlignment (QtCore.Qt.AlignRight)
        self.lab_rms_thr     .setAlignment (QtCore.Qt.AlignRight)
        self.lab_min_thr     .setAlignment (QtCore.Qt.AlignRight)
        self.lab_max_thr     .setAlignment (QtCore.Qt.AlignRight)

        self.edi_dir_work    .setMinimumWidth(300)
        self.but_dir_work    .setFixedWidth(80)
        self.edi_dir_results .setMinimumWidth(300)
        self.but_dir_results .setFixedWidth(80)
        self.box_bat_queue   .setFixedWidth(100)
        self.edi_dark_start  .setFixedWidth(80)
        self.edi_dark_end    .setFixedWidth(80)
        self.edi_dark_scan   .setFixedWidth(80)
        self.edi_dark_sele   .setFixedWidth(80)
        self.edi_rms_thr_min .setFixedWidth(80)
        self.edi_rms_thr     .setFixedWidth(80)
        self.edi_min_thr     .setFixedWidth(80)
        self.edi_max_thr     .setFixedWidth(80)
        self.but_show_vers   .setFixedWidth(100)
        self.but_lsf_status  .setFixedWidth(100)


    def setParent(self,parent) :
        self.parent = parent


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        #self.frame.setGeometry(self.rect())
        pass


    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        #try    : del cp.guiworkresdirs # GUIConfigPars
        #except : pass # silently ignore


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def onButShowVers(self):
        #list_of_pkgs = ['CalibManager', 'ImgAlgos'] #, 'CSPadPixCoords', 'PSCalib', 'pdscalibdata']
        #msg = 'Package versions:\n'
        #for pkg in list_of_pkgs :
        #    msg += '%s  %s\n' % (gu.get_pkg_version(pkg).ljust(10), pkg.ljust(32))

        #msg = cp.package_versions.text_version_for_all_packages()
        msg = cp.package_versions.text_rev_and_tag_for_all_packages()
        logger.info(msg, __name__ )


    def onButLsfStatus(self):
        queue = cp.bat_queue.value()
        farm = cp.dict_of_queue_farm[queue]
        msg, status = gu.msg_and_status_of_lsf(farm)
        msgi = '\nLSF status for queue %s on farm %s: \n%s\nLSF status for %s is %s' % \
               (queue, farm, msg, queue, {False:'bad',True:'good'}[status])
        logger.info(msgi, __name__ )

        cmd, msg = gu.text_status_of_queues(cp.list_of_queues)
        msgq = '\nStatus of queues for command: %s \n%s' % (cmd, msg)       
        logger.info(msgq, __name__ )


    def onButDirWork(self):
        self.selectDirectory(cp.dir_work, self.edi_dir_work, 'work')


    def onButDirResults(self):
        self.selectDirectory(cp.dir_results, self.edi_dir_results, 'results')


    def selectDirectory(self, par, edi, label=''):        
        logger.debug('Select directory for ' + label, __name__)
        dir0 = par.value()
        path, name = os.path.split(dir0)
        dir = str( QtGui.QFileDialog.getExistingDirectory(None,'Select directory for '+label,path) )

        if dir == dir0 or dir == '' :
            logger.info('Directiry for ' + label + ' has not been changed.', __name__)
            return
        edi.setText(dir)        
        par.setValue(dir)
        logger.info('Set directory for ' + label + str(par.value()), __name__)

        gu.create_directory(dir)


    def onBoxBatQueue(self):
        queue_selected = self.box_bat_queue.currentText()
        cp.bat_queue.setValue( queue_selected ) 
        logger.info('onBoxBatQueue - queue_selected: ' + queue_selected, __name__)


    def onEditPrefix(self):
        logger.debug('onEditPrefix', __name__)
        cp.fname_prefix.setValue( str(self.edi_fname_prefix.displayText()) )
        logger.info('Set file name common prefix: ' + str( cp.fname_prefix.value()), __name__ )


    def onEdiDarkStart(self):
        str_value = str( self.edi_dark_start.displayText() )
        cp.bat_dark_start.setValue(int(str_value))      
        logger.info('Set start event for dark run: %s' % str_value, __name__ )


    def onEdiDarkEnd(self):
        str_value = str( self.edi_dark_end.displayText() )
        cp.bat_dark_end.setValue(int(str_value))      
        logger.info('Set last event for dark run: %s' % str_value, __name__ )


    def onEdiDarkScan(self):
        str_value = str( self.edi_dark_scan.displayText() )
        cp.bat_dark_scan.setValue(int(str_value))      
        logger.info('Set the number of events to scan: %s' % str_value, __name__ )


    def onEdiDarkSele(self):
        str_value = str( self.edi_dark_sele.displayText() )
        cp.bat_dark_sele.setValue(int(str_value))      
        logger.info('Set the event code for selector: %s' % str_value, __name__ )


    def onEdiRmsThrMin(self):
        str_value = str( self.edi_rms_thr_min.displayText() )
        cp.mask_rms_thr_min.setValue(float(str_value))  
        logger.info('Set hot pixel RMS MIN threshold: %s' % str_value, __name__ )


    def onEdiRmsThr(self):
        str_value = str( self.edi_rms_thr.displayText() )
        cp.mask_rms_thr.setValue(float(str_value))  
        logger.info('Set hot pixel RMS MAX threshold: %s' % str_value, __name__ )


    def onEdiMinThr(self):
        str_value = str( self.edi_min_thr.displayText() )
        cp.mask_min_thr.setValue(float(str_value))  
        logger.info('Set hot pixel MIN threshold: %s' % str_value, __name__ )


    def onEdiMaxThr(self):
        str_value = str( self.edi_max_thr.displayText() )
        cp.mask_max_thr.setValue(float(str_value))  
        logger.info('Set hot pixel MAX threshold: %s' % str_value, __name__ )



    def on_cbx(self):
        #if self.cbx.hasFocus() :
        par = cp.dark_deploy_hotpix
        cbx = self.cbx_deploy_hotpix

        par.setValue( cbx.isChecked() )
        msg = 'check box ' + cbx.text()  + ' is set to: ' + str( par.value())
        logger.info(msg, __name__ )


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIConfigPars ()
    widget.show()
    app.exec_()

#-----------------------------
