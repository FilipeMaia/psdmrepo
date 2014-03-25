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
import time   # for sleep(sec)
from time import time 

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersForApp import cp
from Logger                 import logger
import GlobalUtils          as     gu
from FileNameManager        import fnm
from BatchJobPedestals      import *
from FileDeployer           import fd
from BatchLogScanParser     import blsp # Just in order to instatiate it

#---------------------
#  Class definition --
#---------------------
class GUIDarkListItemRun ( QtGui.QWidget ) :
    """GUI sets the source dark run number, validity range, and starts calibration of pedestals"""

    def __init__ ( self, parent=None, str_run_number='0000', str_run_type='Type N/A', comment='') :

        self.t0_sec = time()

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(100, 100, 600, 35)
        self.setWindowTitle('GUI Dark Run Item')
        #try : self.setWindowIcon(cp.icon_help)
        #except : pass
        self.setFrame()

        #self.list_of_runs   = None
        self.parent = parent

        self.run_number     = int(str_run_number) # int run number
        self.str_run_number = str_run_number # cp.str_run_number
        self.str_run_type   = str_run_type   # cp.str_run_number
        self.str_run_from   = str_run_number # cp.str_run_from
        self.str_run_to     = 'end'          # cp.str_run_to
        self.comment        = comment
        self.calib_dir      = cp.calib_dir
        self.det_name       = cp.det_name
        self.dict_bjpeds    = cp.dict_bjpeds

        self.create_or_use_butch_object()

        self.lab_run  = QtGui.QLabel('Run')
        self.lab_from = QtGui.QLabel('valid from')
        self.lab_to   = QtGui.QLabel('to')

        #self.lab_rnum = QtGui.QPushButton( self.str_run_number )
        self.lab_rnum = QtGui.QLabel( self.str_run_number )
        self.lab_type = QtGui.QLabel( self.str_run_type + '  ' + comment)
        self.but_go   = QtGui.QPushButton( 'Go' )
        self.but_depl = QtGui.QPushButton( 'Deploy' )
        self.edi_from = QtGui.QLineEdit  ( self.str_run_from )
        self.edi_to   = QtGui.QLineEdit  ( self.str_run_to )

        #self.but_stop.setVisible(False)

        self.edi_from.setValidator(QtGui.QIntValidator(0,9999,self))
        self.edi_to  .setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]\\d{0,3}|end$"),self))

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.lab_run)
        self.hbox.addWidget(self.lab_rnum)
        #self.hbox.addStretch(1)     
        self.hbox.addWidget(self.lab_from)
        self.hbox.addWidget(self.edi_from)
        self.hbox.addWidget(self.lab_to)
        self.hbox.addWidget(self.edi_to)
        #self.hbox.addSpacing(150)     
        self.hbox.addStretch(1)     
        self.hbox.addWidget(self.but_go)
        self.hbox.addWidget(self.but_depl)
        self.hbox.addStretch(1)     
        self.hbox.addWidget(self.lab_type)
        #self.hbox.addWidget(self.but_stop)

        self.setLayout(self.hbox)

        self.connect( self.but_go,   QtCore.SIGNAL('clicked()'),         self.onButGo )
        self.connect( self.but_depl, QtCore.SIGNAL('clicked()'),         self.onButDeploy )
        self.connect( self.edi_from, QtCore.SIGNAL('editingFinished()'), self.onEdiFrom )
        self.connect( self.edi_to,   QtCore.SIGNAL('editingFinished()'), self.onEdiTo )
   
        self.showToolTips()

        self.setStatusMessage()
        self.setFieldsEnabled(cp.det_name.value() != '')

        #cp.guidarkrunitem = self
        #print '  GUIDarkListItemRun Consumed time (sec) =', time()-self.t0_sec


    def create_or_use_butch_object(self) :
        """Creates BatchJobPedestals object for the 1st time or use existing in the dictionary
        """
        self.bjpeds = BatchJobPedestals(self.run_number) 

#        if self.run_number in self.dict_bjpeds.keys() :
#            #print 'Use existing BatchJobPedestals object for run %s' % self.str_run_number
#            self.bjpeds = self.dict_bjpeds[self.run_number]
#        else :
#            #print 'Create new BatchJobPedestals object for run %s' % self.str_run_number
#            self.bjpeds = self.dict_bjpeds[self.run_number] = BatchJobPedestals(self.run_number) 


    def showToolTips(self):
        self.lab_rnum.setToolTip('Data run for calibration.')
        self.lab_type.setToolTip('Type of file (data, dark, etc)')
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

        logger.debug('Set fields enabled: %s' %  is_enabled, __name__)

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
        self.setMinimumSize(600,35)
        self.           setStyleSheet (cp.styleBkgd)
        #self.           setStyleSheet (cp.styleYellowish)

        self.lab_from.setStyleSheet(cp.styleLabel)
        self.lab_to  .setStyleSheet(cp.styleLabel)
        self.lab_run .setStyleSheet(cp.styleLabel)
        self.but_depl.setStyleSheet(cp.styleButtonGood)

        #self.lab_rnum .setFixedWidth(60)        
        self.lab_type .setMinimumWidth(250)        
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
        #logger.debug('setStyleOfButtons - update buttons status for %s' % self.str_run_number, __name__)
        files_are_available = self.bjpeds.status_for_peds_files_essential()
        #print 'setStatusStyleOfButtons: files_are_available = ', files_are_available        
        #print 'Work files for run %s status: %s' % (self.str_run_number, files_are_available)

        if self.edi_from.isReadOnly() : self.edi_from.setStyleSheet (cp.styleEditInfo)
        else                          : self.edi_from.setStyleSheet (cp.styleEdit)

        if self.edi_to.isReadOnly()   : self.edi_to.setStyleSheet (cp.styleEditInfo)
        else                          : self.edi_to.setStyleSheet (cp.styleEdit)

        self.but_depl.setVisible(self.but_depl.isEnabled() and files_are_available )
        self.but_go  .setVisible(self.but_go  .isEnabled())

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

        logger.info('onButGo', __name__)

        but = self.but_go
        but.setStyleSheet(cp.styleDefault)

        if   but.text() == 'Go' : 
            logger.info('onButGo for run %s' % self.str_run_number, __name__ )
            but.setEnabled(False)
            self.bjpeds.start_auto_processing()
            if self.bjpeds.autoRunStage :            
                but.setText('Stop')
            but.setEnabled(True)

        elif but.text() == 'Stop' : 
            logger.info('onButStop for run %s' % self.str_run_number, __name__ )
            self.bjpeds.stop_auto_processing()
            but.setEnabled(True)
            but.setText('Go')

        but.setStyleSheet(cp.styleButtonWarning)


    def onStop(self):
        msg = 'onStop - buttons status should be updated now for %s' % (self.str_run_number)
        logger.info(msg, __name__)
        self.but_go.setEnabled(True)        
        self.but_go.setText('Go')
        self.updateButtons()


    def setStatusMessage(self):
        pass
        #if cp.guistatus is None : return
        #cp.guistatus.setStatusMessage(msg)


    def onButDeploy(self):
        """Deploys the calibration file(s)"""
        list_of_deploy_commands, list_of_sources = self.get_list_of_deploy_commands_and_sources()
        msg = 'Deploy calibration file(s):'

        if list_of_deploy_commands == [] :
            msg += 'List of commands IS EMPTY !!!'  
            logger.info(msg, __name__)
            return


        list_src_cbx = [[src,True] for src in list_of_sources]
        resp = gu.changeCheckBoxListInPopupMenu(list_src_cbx, win_title='Confirm Deployment')

        if resp != 1 :
            logger.info('Deployment is cancelled!', __name__)
            return

        #for cmd,[src,cbx] in zip(list_of_deploy_commands, list_src_cbx) :
        #    print cbx, src, '\n', cmd, '\n'
        #    if cbx : fd.procDeployCommand(cmd)

        for cmd in list_of_deploy_commands :
            #print 'cmd: ', cmd
            if self.is_allowed_command(cmd, list_src_cbx) : fd.procDeployCommand(cmd,'calib-dark')

        if cp.guistatus is not None : cp.guistatus.updateStatusInfo()


    def is_allowed_command(self, cmd, list_src_cbx):
        """Check the deployment command is for selected src"""
        destination = cmd.split()[2] # for example: /reg/d/psdm/CXI/cxib2313/calib/CsPad::CalibV1/CxiDsd.0:Cspad.0/pedestals/1-1.data
        for src,cbx in list_src_cbx :
            if not cbx : continue
            if src in destination : return True
        return False


    def get_list_of_deploy_commands_and_sources(self):
        """Get list of deploy commands for all detectors of the same type"""
        self.exportLocalPars()
        #cp.blsp.print_list_of_types_and_sources()
        list_of_types, list_of_sources = cp.blsp.list_of_types_and_sources_for_selected_detectors()
        # list_of_types  : ['CsPad::DataV1',    'CsPad::DataV1']
        # list_of_sources: ['CxiDs1.0:Cspad.0', 'CxiDsd.0:Cspad.0']

        list_of_deploy_commands  = self.get_list_of_deploy_commands_for_calibtype(list_of_types, list_of_sources, fnm.path_peds_ave(),    'pedestals')
        list_of_deploy_commands += self.get_list_of_deploy_commands_for_calibtype(list_of_types, list_of_sources, fnm.path_peds_rms(),    'pixel_rms')

        if cp.dark_deploy_hotpix.value() :
            list_of_deploy_commands += self.get_list_of_deploy_commands_for_calibtype(list_of_types, list_of_sources, fnm.path_hotpix_mask(), 'pixel_status')

        return list_of_deploy_commands, list_of_sources


    def get_list_of_deploy_commands_for_calibtype(self, list_of_types, list_of_sources, base_path, calibtype='pedestals'):
        """Get list of deploy commands for lists of type and sources for calibtype"""
        list_of_files = gu.get_list_of_files_for_list_of_insets( base_path, list_of_sources )

        list_of_deploy_commands = []

        for file, type, source in zip(list_of_files, list_of_types, list_of_sources) :

            src  = source

            pos1 = type.find('::')
            typ  = type[:pos1] + '::CalibV1'

            fname = '%s-%s.data' % ( self.str_run_from.lstrip('0'),
                                     self.str_run_to  .lstrip('0') )

            calib_path = os.path.join(self.calib_dir.value(), typ, src, calibtype, fname)
            cmd = 'cp %s %s' % (file, calib_path)

            list_of_deploy_commands.append(cmd)

        return list_of_deploy_commands
   

    def updateButtons(self) :
        #logger.info('update', __name__)
        self.setFieldsEnabled(cp.det_name.value() != '')
        #self.setStatusStyleOfButtons()


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    w = GUIDarkListItemRun(parent=None, str_run_number='0016')
    w.setFieldsEnabled(True)
    w.show()
    app.exec_()

#-----------------------------
