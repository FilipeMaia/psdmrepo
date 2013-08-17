#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDarkListItemRun ...
#
#------------------------------------------------------------------------

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
from FileNameManager        import fnm
from BatchJobPedestals      import *

#---------------------
#  Class definition --
#---------------------
class GUIDarkListItemRun ( QtGui.QWidget ) :
    """GUI sets the source dark run number, validity range, and starts calibration of pedestals"""

    #char_expand    = u' \u25BE' # down-head triangle

    def __init__ ( self, parent=None, str_run_number='0000') :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(100, 100, 600, 70)
        self.setWindowTitle('GUI Dark Run Item')
        #try : self.setWindowIcon(cp.icon_help)
        #except : pass
        self.setFrame()

        #self.list_of_runs   = None
        self.parent = parent
        
        self.str_run_number = str_run_number # cp.str_run_number
        self.str_run_from   = str_run_number # cp.str_run_from
        self.str_run_to     = 'end'          # cp.str_run_to
        self.calib_dir      = cp.calib_dir
        self.det_name       = cp.det_name

        self.bjpeds = BatchJobPedestals(parent=self) 

        self.lab_run  = QtGui.QLabel('Run')
        self.lab_from = QtGui.QLabel('valid from')
        self.lab_to   = QtGui.QLabel('to')

        #self.lab_rnum = QtGui.QPushButton( self.str_run_number )
        self.lab_rnum = QtGui.QLabel( self.str_run_number )
        self.but_go   = QtGui.QPushButton( 'Go' )
        self.but_depl = QtGui.QPushButton( 'Deploy' )
        self.edi_from = QtGui.QLineEdit  ( self.str_run_from )
        self.edi_to   = QtGui.QLineEdit  ( self.str_run_to )

        #self.but_stop.setVisible(False)

        self.edi_from.setValidator(QtGui.QIntValidator(0,9999,self))

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.lab_run)
        self.hbox.addWidget(self.lab_rnum)
        #self.hbox.addStretch(1)     
        self.hbox.addWidget(self.lab_from)
        self.hbox.addWidget(self.edi_from)
        self.hbox.addWidget(self.lab_to)
        self.hbox.addWidget(self.edi_to)
        self.hbox.addSpacing(150)     
        self.hbox.addWidget(self.but_go)
        self.hbox.addWidget(self.but_depl)
        self.hbox.addStretch(1)     
        #self.hbox.addWidget(self.but_stop)

        self.setLayout(self.hbox)

        self.connect( self.but_go  , QtCore.SIGNAL('clicked()'),         self.onButGo )
        self.connect( self.but_depl, QtCore.SIGNAL('clicked()'),         self.onButDeploy )
        self.connect( self.edi_from, QtCore.SIGNAL('editingFinished()'), self.onEdiFrom )
        self.connect( self.edi_to  , QtCore.SIGNAL('editingFinished()'), self.onEdiTo )
   
        self.showToolTips()

        self.setStatusMessage()
        self.setFieldsEnabled(cp.det_name.value() != 'None')

        #cp.guidarkrunitem = self


    def showToolTips(self):
        self.lab_rnum.setToolTip('Data run for calibration.')
        self.but_go  .setToolTip('Begin data processing for calibration.')
        self.edi_from.setToolTip('Type in the run number \nas a lower limit of the validity range.')
        self.edi_to  .setToolTip('Type in the run number or "end"\nas an upper limit of the validity range.')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setFieldsEnabled(self, is_enabled=True):

        logger.info('Set fields enabled: %s' %  is_enabled, __name__)

        #self.lab_rnum .setEnabled(is_enabled)
        self.but_go   .setEnabled(is_enabled)
        self.but_depl .setEnabled(is_enabled)

        self.edi_from .setReadOnly(not is_enabled)
        self.edi_to   .setReadOnly(not is_enabled)

        self.edi_from .setEnabled(is_enabled)
        self.edi_to   .setEnabled(is_enabled)

        #if self.str_run_number == 'None' : ...

        self.setStyle()



    def setStyle(self):
        self.setMinimumSize(600,34)
        self.           setStyleSheet (cp.styleBkgd)
        #self.           setStyleSheet (cp.styleYellowish)

        self.lab_from.setStyleSheet(cp.styleLabel)
        self.lab_to  .setStyleSheet(cp.styleLabel)
        self.lab_run .setStyleSheet(cp.styleLabel)
        self.but_depl.setStyleSheet(cp.styleButtonGood)

        self.lab_rnum .setFixedWidth(80)        
        self.edi_from .setFixedWidth(40)
        self.edi_to   .setFixedWidth(40)
        self.but_go   .setFixedWidth(60)        
        self.but_depl .setFixedWidth(60)

        self.edi_from .setAlignment (QtCore.Qt.AlignRight)
        self.edi_to   .setAlignment (QtCore.Qt.AlignRight)

        #self.but_go.setEnabled(self.str_run_number != 'None' and self.lab_rnum.isEnabled())

        self.setContentsMargins (QtCore.QMargins(0,-9,0,-9))
        #self.setContentsMargins (QtCore.QMargins(0,5,0,0))

        self.setStatusStyleOfButtons()


    def setStatusStyleOfButtons(self):
        logger.info('setStyleOfButtons', __name__)

        files_are_available = self.bjpeds.status_for_peds_files_essential()

        #print 'Work files for run %s status: %s' % (self.str_run_number, files_are_available)

        if self.edi_from.isReadOnly() : self.edi_from.setStyleSheet (cp.styleEditInfo)
        else                          : self.edi_from.setStyleSheet (cp.styleEdit)

        if self.edi_to.isReadOnly()   : self.edi_to.setStyleSheet (cp.styleEditInfo)
        else                          : self.edi_to.setStyleSheet (cp.styleEdit)

        self.but_depl.setVisible(files_are_available)
        self.but_go  .setVisible(self.but_go.isEnabled())

        if files_are_available : self.but_go.setStyleSheet(cp.styleButton)
        else                   : self.but_go.setStyleSheet(cp.styleButtonGood)

        #if self.but_go.text() == 'Stop' : 
            #self.but_go.setText('Go')



    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())
        #self.box_txt.setGeometry(self.contentsRect())

        
    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        #self.saveLogTotalInFile() # It will be saved at closing of GUIMain

        #try    : cp.guimain.butLogger.setStyleSheet(cp.styleButtonBad)
        #except : pass

        #self.box_txt.close()

        #try    : del cp.guistatus # GUIDarkListItemRun
        #except : pass


    def onClose(self):
        logger.info('onClose', __name__)
        self.close()


    def onEdiFrom(self):
        logger.debug('onEdiFrom', __name__ )
        self.str_run_from = str( self.edi_from.displayText() )        
        msg = 'Set the run validity range from %s' % self.str_run_from
        logger.info(msg, __name__ )
        self.setStatusMessage()

 
    def onEdiTo(self):
        logger.debug('onEdiTo', __name__ )
        self.str_run_to = str( self.edi_to.displayText() )        
        msg = 'Set the run validity range up to %s' % self.str_run_to
        logger.info(msg, __name__ )
        self.setStatusMessage()


    def exportLocalPars(self):
        """Export local parameters to configuration current"""
        cp.str_run_number.setValue(self.str_run_number)
        #cp.str_run_from  .setValue(self.str_run_from  )
        #cp.str_run_to    .setValue(self.str_run_to    )

 

    def onButGo(self):
        self.exportLocalPars()
        self.but_depl.setVisible(False)

        but = self.but_go
        but.setStyleSheet(cp.styleDefault)

        if   but.text() == 'Go' : 
            logger.info('onButGo for run %s' % self.str_run_number, __name__ )
            self.bjpeds.start_auto_processing()
            but.setText('Stop')
            
        elif but.text() == 'Stop' : 
            logger.info('onButStop for run %s' % self.str_run_number, __name__ )
            self.bjpeds.stop_auto_processing()
            but.setText('Go')

        but.setStyleSheet(cp.styleButtonBad)


    def onStop(self):
        self.but_go.setText('Go')
        self.setStatusStyleOfButtons()


    def setStatusMessage(self):
        if cp.guistatus is None : return
        #msg = 'From %s to %s use dark run %s' % (self.str_run_from.value(), self.str_run_to.value(), self.str_run_number.value())
        #msg = gu.get_text_content_of_calib_dir_for_detector(path=self.calib_dir.value(), det=self.det_name.value(), calib_type='pedestals')
        #cp.guistatus.setStatusMessage(msg)


    def onButDeploy(self):
        """Deploys the calibration file(s)"""
        list_of_deploy_commands = self.get_list_of_deploy_commands()
        msg = 'Deploy calibration file(s):'

        if list_of_deploy_commands == [] :
            msg += 'List of commands IS EMPTY !!!'  

        for cmd in list_of_deploy_commands :
            msg += '\n' + cmd
        logger.info(msg, __name__)


    def get_list_of_deploy_commands(self):

        self.exportLocalPars()
        cp.blsp.parse_batch_log_peds_scan()
        #cp.blsp.print_list_of_types_and_sources()

        list_of_sources = cp.blsp.get_list_of_sources()
        list_of_types   = cp.blsp.get_list_of_types()
        list_of_files   = cp.blsp.get_list_of_files_for_all_sources(fnm.path_peds_ave())

        list_of_deploy_commands = []

        for file, type, source in zip(list_of_files, list_of_types, list_of_sources) :

            pos1 = source.find('DetInfo(') + 8
            src  = source[pos1:-1]

            pos1 = type.find('::')
            typ  = type[:pos1] + '::CalibV1'

            fname = '%s-%s.data' % ( self.str_run_from.lstrip('0'),
                                     self.str_run_to  .lstrip('0') )

            calib_path = os.path.join(self.calib_dir.value(), typ, src, 'pedestals', fname)
            cmd = 'cp %s  %s' % (file, calib_path)

            list_of_deploy_commands.append(cmd)

        return list_of_deploy_commands
   

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    w = GUIDarkListItemRun(parent=None, str_run_number='0024')
    w.setFieldsEnabled(True)
    w.show()
    app.exec_()

#-----------------------------
