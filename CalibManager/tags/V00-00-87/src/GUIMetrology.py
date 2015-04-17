
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMetrology...
#
#------------------------------------------------------------------------

"""Renders the main GUI for the CalibManager.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""

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

from Logger               import logger
from FileNameManager      import fnm
from GUIFileBrowser       import *
from GUIRange             import *
from FileDeployer         import fd
import GlobalUtils        as     gu

from xlsx_parser              import convert_xlsx_to_text
from OpticAlignmentCspadV1    import OpticAlignmentCspadV1
from OpticAlignmentCspadV2    import OpticAlignmentCspadV2
from OpticAlignmentCspad2x2V1 import OpticAlignmentCspad2x2V1

#---------------------
#  Class definition --
#---------------------
#class GUIMetrology ( QtGui.QWidget ) :
class GUIMetrology ( Frame ) :
    """GUI for metrology processing.

    @see BaseClass
    @see OtherClass
    """
    def __init__ (self, parent=None, app=None) :

        self.name = 'GUIMetrology'
        self.myapp = app
        #QtGui.QWidget.__init__(self, parent)
        Frame.__init__(self, parent, mlw=1)

        self.instr_name    = cp.instr_name # for comments in geometry file
        self.fname_prefix  = cp.fname_prefix
        self.fname_metrology_xlsx = cp.fname_metrology_xlsx
        self.fname_metrology_text = cp.fname_metrology_text
        self.img_arr = None
        self.list_of_calib_types = ['center', 'tilt', 'geometry']
        
        cp.setIcons()

        self.setGeometry(10, 25, 725, 200)
        self.setWindowTitle('Metrology')
        #self.setWindowIcon(cp.icon_monitor)
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        #self.setFrame()

        self.setParams()
  
        #self.titFileXlsx = QtGui.QLabel('File xlsx:')

        self.ediFileXlsx = QtGui.QLineEdit ( fnm.path_metrology_xlsx() )
        self.ediFileXlsx.setReadOnly(True)

        self.ediFileText = QtGui.QLineEdit ( fnm.path_metrology_text() ) # cp.fname_metrology_text.value_def() )
        self.ediFileText.setReadOnly(True)

        self.butFileXlsx  = QtGui.QPushButton(' 1. Select xlsx file:')
        self.butConvert   = QtGui.QPushButton(' 2. Convert xlsx to text file(s)')
        self.butFileText  = QtGui.QPushButton(' 3. Select text file:')
        self.butEvaluate  = QtGui.QPushButton(' 4. Evaluate')
        self.butDeploy    = QtGui.QPushButton(' 5. Deploy')
        self.butList      = QtGui.QPushButton('List')
        self.butRemove    = QtGui.QPushButton('Remove')
        self.butViewOffice= QtGui.QPushButton('View xlsx')
        self.butViewText  = QtGui.QPushButton('View text')
        self.butScript    = QtGui.QPushButton(self.script + cp.char_expand )
        self.butSrc       = QtGui.QPushButton(self.source_name + cp.char_expand )
        self.labSrc       = QtGui.QLabel('for detector')
        self.labScript    = QtGui.QLabel('using script')
        self.guirange     = GUIRange()

        self.butViewOffice .setIcon(cp.icon_monitor)
        self.butViewText   .setIcon(cp.icon_monitor)
        #self.butConvert    .setIcon(cp.icon_convert)

        self.grid = QtGui.QGridLayout()
        self.grid_row = 0
        self.grid.addWidget(self.butFileXlsx,   self.grid_row,   0)
        self.grid.addWidget(self.ediFileXlsx,   self.grid_row,   1, 1, 8)
        self.grid.addWidget(self.butViewOffice, self.grid_row,   8)

        self.grid.addWidget(self.butConvert,    self.grid_row+1, 0)
        self.grid.addWidget(self.butList,       self.grid_row+1, 1, 1, 1)
        self.grid.addWidget(self.butRemove,     self.grid_row+1, 2, 1, 1)

        self.grid.addWidget(self.butFileText,   self.grid_row+2, 0)
        self.grid.addWidget(self.ediFileText,   self.grid_row+2, 1, 1, 8)
        self.grid.addWidget(self.butViewText,   self.grid_row+2, 8)

        self.grid.addWidget(self.butEvaluate,   self.grid_row+3, 0)
        self.grid.addWidget(self.labScript,     self.grid_row+3, 1)
        self.grid.addWidget(self.butScript,     self.grid_row+3, 2)

        self.grid.addWidget(self.butDeploy,     self.grid_row+4, 0)
        self.grid.addWidget(self.labSrc,        self.grid_row+4, 1)
        self.grid.addWidget(self.butSrc,        self.grid_row+4, 2)
        self.grid.addWidget(self.guirange,      self.grid_row+4, 3, 1, 5)
        #self.setLayout(self.grid)
          
        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addLayout(self.grid)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.connect( self.butFileXlsx,   QtCore.SIGNAL('clicked()'), self.onButFileXlsx   ) 
        self.connect( self.butFileText,   QtCore.SIGNAL('clicked()'), self.onButFileText   ) 
        self.connect( self.butViewOffice, QtCore.SIGNAL('clicked()'), self.onButViewOffice )
        self.connect( self.butViewText,   QtCore.SIGNAL('clicked()'), self.onButViewText   )
        self.connect( self.butConvert,    QtCore.SIGNAL('clicked()'), self.onButConvert    )
        self.connect( self.butRemove,     QtCore.SIGNAL('clicked()'), self.onButRemove     )
        self.connect( self.butList,       QtCore.SIGNAL('clicked()'), self.onButList       )
        self.connect( self.butEvaluate,   QtCore.SIGNAL('clicked()'), self.onButEvaluate   )
        self.connect( self.butDeploy,     QtCore.SIGNAL('clicked()'), self.onButDeploy     )
        self.connect( self.butScript,     QtCore.SIGNAL('clicked()'), self.onButScript     )
        self.connect( self.butSrc,        QtCore.SIGNAL('clicked()'), self.onButSrc        )
 
        self.showToolTips()
        self.setStyle()

        cp.guimetrology = self
        #self.move(10,25)
        
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------


    def showToolTips(self):
        #pass
        self.ediFileXlsx  .setToolTip('Persistent path to xlsx file') 
        self.butFileXlsx  .setToolTip('Open file browser dialog window\nand select xlsx file. This file is\nusually e-mailed from detector group.') 
        self.butViewOffice.setToolTip('Open openoffice.org window')
        self.butViewText  .setToolTip('Open file viewer window')
        self.butFileText  .setToolTip('Open file browser dialog window\nand select metrology text file') 
        self.ediFileText  .setToolTip('Path to the text metrology file which\nis used to evaluate calibration constants.') 
        self.butConvert   .setToolTip('Convert xlsx to text metrology file(s)')
        self.butList      .setToolTip('List temporarty metrology text file(s)')
        self.butRemove    .setToolTip('Remove temporarty metrology text file(s)')
        self.butEvaluate  .setToolTip('Run quality check script and\nevaluate geometry alignment parameters')
        self.butDeploy    .setToolTip('Deploy geometry alignment parameters')
        self.butScript    .setToolTip('Select the script to process optic metrology file')
        self.butSrc       .setToolTip('Select name of the detector')
 

#    def setFrame(self):
#        self.frame = QtGui.QFrame(self)
#        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
#        self.frame.setLineWidth(0)
#        self.frame.setMidLineWidth(1)
#        self.frame.setGeometry(self.rect())
#        #self.frame.setVisible(False)

    def setParams(self) :
        #if self.path_fm_selected != '' :
        #    self.path_fm_selected = os.path.dirname(self.path_fm_selected)
        self.str_run_from     = '0'
        self.str_run_to       = 'end'
        self.source_name      = 'Select'
        self.script           = 'Select'
        self.calib_type       = 'Select'


    def setStyle(self):

        self.setMinimumSize(725,200)
        self.setMaximumSize(800,200)
        
        self.              setStyleSheet(cp.styleBkgd)
        self.butViewOffice.setStyleSheet(cp.styleButton)
        self.butViewText  .setStyleSheet(cp.styleButton)
        #self.butViewOffice.setFixedWidth(200)
        #self.butViewOffice.setMinimumHeight(60)
        #self.butViewOffice.setMinimumSize(180,60)

        self.butFileXlsx .setStyleSheet(cp.styleButtonLeft)
        self.butConvert  .setStyleSheet(cp.styleButtonLeft) 
        self.butFileText .setStyleSheet(cp.styleButtonLeft) 
        self.butEvaluate .setStyleSheet(cp.styleButtonLeft) 
        self.butDeploy   .setStyleSheet(cp.styleButtonLeft) 

        self.ediFileXlsx.setFixedWidth(400)
        self.ediFileXlsx.setStyleSheet(cp.styleEditInfo) 
        self.ediFileXlsx.setEnabled(False)            

        self.ediFileText.setFixedWidth(400)
        self.ediFileText.setStyleSheet(cp.styleEditInfo) 
        self.ediFileText.setEnabled(False)            

        self.labSrc     .setStyleSheet(cp.styleLabel)
        self.labScript  .setStyleSheet(cp.styleLabel)

        #self.butFBrowser.setVisible(False)
        #self.butSave.setText('')
        #self.butExit.setText('')
        #self.butExit.setFlat(True)

        self.setStyleButtons()


    def setStyleButtons(self):
        if self.source_name == 'Select' : self.butSrc.setStyleSheet(cp.stylePink)
        else                            : self.butSrc.setStyleSheet(cp.styleButton)

        if self.script == 'Select' : self.butScript.setStyleSheet(cp.stylePink)
        else                       : self.butScript.setStyleSheet(cp.styleButton)

  
    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self.name) 
        #self.frame.setGeometry(self.rect())
        #print 'GUIMetrology.resizeEvent: %s' % str(self.size())
        pass


    def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', self.name)

        try    : cp.guifilebrowser.close()
        except : pass


    def onExit(self):
        logger.debug('onExit', self.name)
        self.close()


    def onButFileXlsx(self):
        logger.debug('onButFileXlsx', __name__)
        but = self.butFileXlsx
        edi = self.ediFileXlsx
        par = self.fname_metrology_xlsx
        #prefix = self.fname_prefix.value()
        filter = 'Text files (*.xlsx )\nAll files (*)'

        self.onButFile(but, edi, par, filter, set_path=True)


    def onButFileText(self):
        logger.debug('onButFileText', __name__)
        but = self.butFileText
        edi = self.ediFileText
        par = self.fname_metrology_text        
        basename  = os.path.basename( fnm.path_metrology_text() )
        fname, ext = os.path.splitext(basename)
        filter = 'Text files (*' + ext + ')\nAll files (*)'
        self.onButFile(but, edi, par, filter, set_path=True)


    def onButFile(self, but, edi, par, filter, set_path=True):
        logger.debug('onButFile', __name__)
        path = str( edi.displayText() )
        dname, fname = os.path.split(path)
        msg = 'dir : %s   file : %s' % (dname, fname)
        logger.info(msg, __name__)
        path = str( QtGui.QFileDialog.getOpenFileName(self, 'Open file', dname, filter=filter) )
        dname, fname = os.path.split(path)

        if dname == '' or fname == '' :
            logger.info('Input directiry name or file name is empty... use default values', __name__)
            return
        else :
            edi.setText(path)
            if set_path : par.setValue(path)
            else        : par.setValue(fname)
            logger.info('Selected file: %s' % path, __name__)


    def onButViewOffice(self):       
        logger.debug('onLogger', self.name)
        try    :
            #cp.viewoffice.close()
            #del cp.viewoffice
            self.butViewOffice.setStyleSheet(cp.styleButton)
            #self.butViewOffice.setText('Open openoffice')

            cmd = 'openoffice.org %s &' % fnm.path_metrology_xlsx()
            msg = 'Confirm command: %s' % cmd

            resp = gu.confirm_or_cancel_dialog_box(parent=self.butViewOffice, text=msg, title='Please confirm or cancel!')
            if resp :
                logger.info('Approved command:\n' + cmd, __name__)
                self.commandInSubproc(cmd)
            else :
                logger.info('Command is cancelled', __name__)

        except :
            self.butViewOffice.setStyleSheet(cp.styleButtonGood)
            #self.butViewOffice.setText('Close openoffice')

            #cp.viewoffice = MaskEditor(**pars)
            #cp.viewoffice.move(self.pos().__add__(QtCore.QPoint(820,-7))) # open window with offset w.r.t. parent
            #cp.viewoffice.show()


    def onButViewText(self):
        logger.debug('onButViewText', __name__)
        try    :
            cp.guifilebrowser.close()
            #self.but_view.setStyleSheet(cp.styleButtonBad)
        except :
            #self.but_view.setStyleSheet(cp.styleButtonGood)

            list_of_files = fnm.get_list_of_metrology_text_files()
            if self.script != 'Select' :
                list_of_files += self.list_metrology_alignment_const_fnames()

            cp.guifilebrowser = GUIFileBrowser(None, list_of_files, fnm.path_metrology_text())
            cp.guifilebrowser.move(self.pos().__add__(QtCore.QPoint(880,40))) # open window with offset w.r.t. parent
            cp.guifilebrowser.show()



    def checkTextFileName(self):

        edi = self.ediFileText
        par = self.fname_metrology_text        

        if fnm.path_metrology_text() != fnm.path_metrology_text_def() :

            msg = 'TEXT FILE WILL BE OVERWRITTEN!\nUse default name %s\n for output file' % fnm.path_metrology_text_def()
            resp = gu.confirm_or_cancel_dialog_box(parent=self.butConvert, text=msg, title='Please confirm or cancel!')
            if resp :
                logger.info('Approved:\n' + msg.replace('\n',' '), __name__)
                par.setDefault()
                edi.setText(fnm.path_metrology_text_def())
            else :
                logger.info('Selected current file name: %s' % fnm.path_metrology_text(), __name__)
 

    def onButConvert(self):
        logger.debug('onButConvert', __name__)
        
        if not os.path.exists(fnm.path_metrology_xlsx()) :
            msg = 'Input file %s DOES NOT exist!' % fnm.path_metrology_xlsx() 
            logger.warning(msg, __name__)
            return

        self.checkTextFileName()

        #ifname = fnm.path_metrology_xlsx()
        #ofname = fnm.path_metrology_text()
        list_ofnames = convert_xlsx_to_text(fnm.path_metrology_xlsx(), fnm.path_metrology_text(), print_bits=0)

        msg = 'File %s is converted to the temporarty metrology text file(s):\n' % fnm.path_metrology_xlsx()
        for name in list_ofnames : msg += '    %s\n' % name
        logger.info(msg, __name__)


    def onButRemove(self):
        #logger.debug('onButRemove', __name__)        
        cmd = 'rm'
        for fname in fnm.get_list_of_metrology_text_files() : cmd += ' %s' % fname
        msg = 'Confirm command: %s' % cmd

        resp = gu.confirm_or_cancel_dialog_box(parent=self.butViewOffice, text=msg, title='Please confirm or cancel!')
        if resp :
            logger.info('Approved command:\n' + cmd, __name__)
            self.commandInSubproc(cmd)
        else :
            logger.info('Command is cancelled', __name__)


    def onButList(self):
        msg = 'List of metrology text files in %s\n' % fnm.path_dir_work()
        for fname in fnm.get_list_of_metrology_text_files() : msg += '    %s\n' % fname
        logger.info(msg, __name__)        


    def get_detector_selected(self):
        lst = cp.list_of_dets_selected()
        len_lst = len(lst)
        msg = '%d detector(s) selected: %s' % (len_lst, str(lst))
        #logger.info(msg, __name__ )

        if len_lst !=1 :
            msg += ' Select THE ONE!'
            logger.warning(msg, __name__ )
            return None

        return lst[0]


    def onButScript(self):
        logger.debug('onButScript', __name__ )

        det = self.get_detector_selected()
        if det is None : return

        if det != cp.list_of_dets[0] :
            logger.warning('Scripts are implemented for CSPAD ONLY !!!: ', __name__)
        
        lst = cp.dict_of_metrology_scripts[det]

        selected = gu.selectFromListInPopupMenu(lst)

        if selected is None : return        # selection is cancelled
        if selected is self.script : return # the same
        
        txt = str(selected)

        self.setScript(txt)
        self.setSrc()
        self.setStyleButtons()


    def onButSrc(self):
        logger.debug('onButSrc', __name__ )

        det = self.get_detector_selected()
        if det is None : return

        try    :
            lst = ru.list_of_sources_for_det(det)
        except :
            lst = cp.dict_of_det_sources[det]

        selected = gu.selectFromListInPopupMenu(lst)

        if selected is None : return             # selection is cancelled
        if selected is self.source_name : return # the same

        txt = str(selected)
        self.setSrc(txt)
        self.setStyleButtons()


    def setScript(self,txt='Select'):
        self.script = txt
        self.butScript.setText( txt + cp.char_expand )
        logger.info('Script is selected: ' + txt, __name__)

  
    def setSrc(self,txt='Select'):
        self.source_name = txt
        self.butSrc.setText( txt + cp.char_expand )
        logger.info('Source selected: ' + txt, __name__)

  
    def onButEvaluate(self):
        logger.debug('onButEvaluate', __name__)
        det = self.get_detector_selected()
        if det is None : return

        if not os.path.exists(fnm.path_metrology_text()) :
            msg = 'Input file %s DOES NOT exist!' % fnm.path_metrology_text() 
            logger.warning(msg, __name__)
            return

        list_of_metrology_scripts = cp.dict_of_metrology_scripts[det]

        if self.script == 'Select' :
            msg = 'Script for processing metrology file is not selected. Select it first...'
            logger.warning(msg, __name__)
            return

        fname_metrology = fnm.path_metrology_text()

        #print 'list_of_metrology_scripts', list_of_metrology_scripts
        #for CSPAD script CSPADV1 CXI-like
        if det == cp.list_of_dets[0] and self.script == list_of_metrology_scripts[0] :            
            msg = 'Evaluate parameters for %s using script %s' % (det, self.script)
            logger.info(msg, __name__)
            optal = OpticAlignmentCspadV1(fname_metrology, print_bits=0, plot_bits=0, \
                                      exp=self.instr_name.value(), det=det)
            self.procCspad(optal)

        #for CSPAD script CSPADV2 XPP-like
        elif det == cp.list_of_dets[0] and self.script == list_of_metrology_scripts[1] :            
            msg = 'Evaluate parameters for %s using script %s' % (det, self.script)
            logger.info(msg, __name__)
            optal = OpticAlignmentCspadV2(fname_metrology, print_bits=0, plot_bits=0, \
                                      exp=self.instr_name.value(), det=det)
            self.procCspad(optal)

        #for CSPAD2x2 script CSPAD2X2V1
        elif det == cp.list_of_dets[1] and self.script == list_of_metrology_scripts[0] :            
            msg = 'Evaluate parameters for %s using script %s' % (det, self.script)
            logger.info(msg, __name__)
            optal = OpticAlignmentCspad2x2V1(fname_metrology, print_bits=0, plot_bits=0, \
                                      exp=self.instr_name.value(), det=det)
            self.procCspad(optal)

        # for other detectors and scripts for now...
        else :            
            msg = 'Script %s is not yet implemented for detector %s...' % (self.script, det)
            logger.warning(msg, __name__)
            return
        
 
    def procCspad(self, optal):
        """Create and save interim files for calibration types"""
        self.list_of_calib_types = ['center', 'tilt', 'geometry']

        fname_metrology = fnm.path_metrology_text()
        msg = 'procCspad(V1,V2,2x2V1) for metrology data in file %s' % fname_metrology
        logger.info(msg, __name__)       

        txt_qc_table_xy = optal.txt_qc_table_xy()
        txt_qc_table_z  = optal.txt_qc_table_z()

        txt_center      = optal.txt_center_pix_formatted_array()
        txt_tilt        = optal.txt_tilt_formatted_array()
        txt_geometry    = optal.txt_geometry()

        logger.info('Quality check in X-Y plane:\n'+txt_qc_table_xy, __name__)       
        logger.info('Quality check in Z:\n'+txt_qc_table_z, __name__)       
        logger.info('parameters of type "center":\n'+txt_center, __name__)       
        logger.info('parameters of type "tilt":\n'+txt_tilt, __name__)       
        logger.info('parameters of type "geometry":\n'+txt_geometry, __name__)       
        
        # Save calibration files in work directory

        dic_type_fname = self.dict_metrology_alignment_const_fname_for_type()

        gu.save_textfile(txt_center,   dic_type_fname['center']) 
        gu.save_textfile(txt_tilt,     dic_type_fname['tilt']) 
        gu.save_textfile(txt_geometry, dic_type_fname['geometry']) 

        msg = 'Save interim metrology alignment files:'
        for type in self.list_of_calib_types :
            msg += '\n  %s   %s' % (type.ljust(16), dic_type_fname[type])

        logger.info(msg, __name__)       


    def dict_metrology_alignment_const_fname_for_type(self) : 
        #lst_const_types = cp.const_types_cspad # ex. ['center', 'tilt',...]
        lst_const_types = self.list_of_calib_types
        lst_of_insets = ['%s-%s' % (self.script,type) for type in lst_const_types] # ex. ['CSPADV1-tilt', ...]
        lst_of_const_fnames = gu.get_list_of_files_for_list_of_insets(fnm.path_metrology_alignment_const(), lst_of_insets)
        return dict(zip(lst_const_types, lst_of_const_fnames))


    def list_metrology_alignment_const_fnames(self) : 
        return self.dict_metrology_alignment_const_fname_for_type().values()


    def onButDeploy(self):
        logger.debug('onButDeploy', __name__)        

        if self.script == 'Select' :
            msg = 'Script for processing metrology file is not selected.... Select it first and evaluate constants (Item 4)'
            logger.warning(msg, __name__)
            return

        if self.source_name == 'Select' :
            msg = 'Detector is not selected. Select it first...'
            logger.warning(msg, __name__)
            return

        list_of_cmds = self.list_of_copy_cmds()


        txt = '\nList of commands for tentetive file deployment:'
        for cmd in list_of_cmds :
            txt += '\n' + cmd
        logger.info(txt, __name__)


        msg = 'Approve commands \njust printed in the logger'
        if self.approveCommand(self.butDeploy, msg) :

            for cmd in list_of_cmds :
                fd.procDeployCommand(cmd, 'metrology-alignment')
                #print 'Command for deployer: ', cmd

            if cp.guistatus is not None : cp.guistatus.updateStatusInfo()




    def approveCommand(self, but, msg):
        resp = gu.confirm_or_cancel_dialog_box(parent=but, text=msg, title='Please confirm or cancel!')
        if resp : logger.info('Commands approved', __name__)
        else    : logger.info('Command is cancelled', __name__)
        return resp


    def list_of_copy_cmds(self):

        det = self.get_detector_selected()
        if det is None : return

        dst_calib_dir  = fnm.path_to_calib_dir()
        dst_calib_type = cp.dict_of_det_calib_types[det]
        dst_source     = self.source_name
        dst_fname      = '%s.data' % self.guirange.getRange()

        #print 'dst_calib_dir  ', dst_calib_dir
        #print 'dst_calib_type ', dst_calib_type
        #print 'dst_source     ', dst_source
        #print 'dst_fname      ', dst_fname

        list_of_cmds = []
        for type, fname in self.dict_metrology_alignment_const_fname_for_type().iteritems() :
            dst_path = os.path.join(dst_calib_dir, dst_calib_type, dst_source, type, dst_fname)
            cmd = 'cp %s %s' % (fname, dst_path)
            list_of_cmds.append(cmd)

        return list_of_cmds

 
    def commandInSubproc(self, cmd):
        
        cmd_seq = cmd.split()
        msg = 'Command: ' + cmd

        #out, err = gu.subproc(cmd_seq)
        #if err != '' : msg += '\nERROR: ' + err
        #if out != '' : msg += '\nRESPONCE: ' + out

        os.system(cmd)
        logger.info(msg, __name__)

        #os.system('chmod 670 %s' % path)


#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIMetrology()
    ex.show()
    app.exec_()
#-----------------------------
