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

from ConfigParametersForApp import cp
from Logger                 import logger
import GlobalUtils          as     gu

#---------------------
#  Class definition --
#---------------------
class GUIConfigPars ( QtGui.QWidget ) :
    """GUI for Work/Result directories"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 250)
        self.setWindowTitle('Configuration Parameters')
        self.setFrame()

        self.tit_dir_work = QtGui.QLabel('Parameters:')

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
        self.lab_rms_thr    = QtGui.QLabel('Threshold RMS, ADU:') 
        self.lab_min_thr    = QtGui.QLabel('Threshold MIN, ADU:') 
        self.lab_max_thr    = QtGui.QLabel('MAX:') 

        self.but_show_vers  = QtGui.QPushButton('Soft Vers')
        self.but_lsf_status = QtGui.QPushButton('LSF status')

        self.edi_dark_start = QtGui.QLineEdit  ( str( cp.bat_dark_start.value() ) )
        self.edi_dark_end   = QtGui.QLineEdit  ( str( cp.bat_dark_end.value()) )
        self.edi_rms_thr    = QtGui.QLineEdit  ( str( cp.mask_rms_thr.value()) )
        self.edi_min_thr    = QtGui.QLineEdit  ( str( cp.mask_min_thr.value()) )
        self.edi_max_thr    = QtGui.QLineEdit  ( str( cp.mask_max_thr.value()) )

        self.edi_dark_start.setValidator(QtGui.QIntValidator(0,9999999,self))
        self.edi_dark_end  .setValidator(QtGui.QIntValidator(0,9999999,self))
        self.edi_rms_thr   .setValidator(QtGui.QDoubleValidator(0,65000,3,self))
        self.edi_min_thr   .setValidator(QtGui.QDoubleValidator(0,65000,3,self))
        self.edi_max_thr   .setValidator(QtGui.QDoubleValidator(0,65000,3,self))
        #self.edi_events.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]\\d{0,3}|end$"),self))

        self.cbx_deploy_hotpix = QtGui.QCheckBox('Deploy hotpix mask')
        self.cbx_deploy_hotpix.setChecked( cp.dark_deploy_hotpix.value() )

        self.grid = QtGui.QGridLayout()
        self.grid_row = 0
        self.grid.addWidget(self.tit_dir_work,      self.grid_row,   0, 1, 9)
        self.grid.addWidget(self.but_dir_work,      self.grid_row+1, 0)
        self.grid.addWidget(self.edi_dir_work,      self.grid_row+1, 1, 1, 8)
        self.grid.addWidget(self.but_dir_results,   self.grid_row+2, 0)
        self.grid.addWidget(self.edi_dir_results,   self.grid_row+2, 1, 1, 8)
        self.grid.addWidget(self.lab_fname_prefix,  self.grid_row+3, 0)
        self.grid.addWidget(self.edi_fname_prefix,  self.grid_row+3, 1, 1, 4)
        self.grid.addWidget(self.lab_bat_queue,     self.grid_row+4, 0)
        self.grid.addWidget(self.box_bat_queue,     self.grid_row+4, 1)
        self.grid.addWidget(self.lab_dark_start,    self.grid_row+5, 0)
        self.grid.addWidget(self.edi_dark_start,    self.grid_row+5, 1)
        self.grid.addWidget(self.lab_dark_end,      self.grid_row+5, 3)
        self.grid.addWidget(self.edi_dark_end,      self.grid_row+5, 4)
        self.grid.addWidget(self.lab_rms_thr,       self.grid_row+6, 0)
        self.grid.addWidget(self.edi_rms_thr,       self.grid_row+6, 1, 1, 4)
        self.grid.addWidget(self.cbx_deploy_hotpix, self.grid_row+6, 3, 1, 4)
        self.grid.addWidget(self.lab_min_thr,       self.grid_row+7, 0)
        self.grid.addWidget(self.edi_min_thr,       self.grid_row+7, 1, 1, 2)
        self.grid.addWidget(self.lab_max_thr,       self.grid_row+7, 3)
        self.grid.addWidget(self.edi_max_thr,       self.grid_row+7, 4, 1, 2)
        self.grid.addWidget(self.but_show_vers,     self.grid_row+8, 0, 1, 2)
        self.grid.addWidget(self.but_lsf_status,    self.grid_row+8, 1, 1, 2)

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


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def setStyle(self):
        self.                 setStyleSheet (cp.styleBkgd)
        self.setMinimumSize(500,240)
        self.setMaximumSize(700,240)

        self.tit_dir_work     .setStyleSheet (cp.styleTitle)
        self.edi_dir_work     .setStyleSheet (cp.styleEditInfo)       
        self.but_dir_work     .setStyleSheet (cp.styleButton) 
        self.edi_dir_results  .setStyleSheet (cp.styleEditInfo)       
        self.but_dir_results  .setStyleSheet (cp.styleButton) 
        self.lab_fname_prefix .setStyleSheet (cp.styleLabel)
        self.edi_fname_prefix .setStyleSheet (cp.styleEdit)
        self.lab_bat_queue    .setStyleSheet (cp.styleLabel)
        self.lab_dark_start   .setStyleSheet (cp.styleLabel)
        self.lab_dark_end     .setStyleSheet (cp.styleLabel)
        self.lab_rms_thr      .setStyleSheet (cp.styleLabel)
        self.lab_min_thr      .setStyleSheet (cp.styleLabel)
        self.lab_max_thr      .setStyleSheet (cp.styleLabel)
        self.cbx_deploy_hotpix.setStyleSheet (cp.styleLabel)
        self.but_show_vers    .setStyleSheet (cp.styleButton) 
        self.but_lsf_status   .setStyleSheet (cp.styleButton) 

        self.tit_dir_work    .setAlignment (QtCore.Qt.AlignLeft)
        self.edi_dir_work    .setAlignment (QtCore.Qt.AlignRight)
        self.edi_dir_results .setAlignment (QtCore.Qt.AlignRight)
        self.lab_fname_prefix.setAlignment (QtCore.Qt.AlignRight)
        self.lab_bat_queue   .setAlignment (QtCore.Qt.AlignRight)
        self.lab_dark_start  .setAlignment (QtCore.Qt.AlignRight)
        self.lab_dark_end    .setAlignment (QtCore.Qt.AlignRight)
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
        self.edi_rms_thr     .setFixedWidth(80)
        self.edi_min_thr     .setFixedWidth(80)
        self.edi_max_thr     .setFixedWidth(80)
        self.but_show_vers   .setFixedWidth(100)
        self.but_lsf_status  .setFixedWidth(100)


    def setParent(self,parent) :
        self.parent = parent


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())


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

        msg = cp.package_versions.text_version_for_all_packages()
        logger.info(msg, __name__ )


    def onButLsfStatus(self):
        queue = cp.bat_queue.value()
        msg, status = gu.msg_and_status_of_lsf(queue)
        msgi = '\nLSF status for queue %s: \n%s\nLSF status for %s is %s' % (queue, msg, queue, {False:'bad',True:'good'}[status])
        logger.info(msgi, __name__ )


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


    def onEdiRmsThr(self):
        str_value = str( self.edi_rms_thr.displayText() )
        cp.mask_rms_thr.setValue(float(str_value))  
        logger.info('Set hot pixel RMS threshold: %s' % str_value, __name__ )

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
