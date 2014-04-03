
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

@version $Id:$

@author Mikhail S. Dubrovin
"""


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

import matplotlib
#matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
if matplotlib.get_backend() != 'Qt4Agg' : matplotlib.use('Qt4Agg')

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersForApp import cp

from Logger               import logger
from FileNameManager      import fnm
from GUIFileBrowser       import *
#from CorAna.MaskEditor import MaskEditor
import GlobalUtils        as     gu
from xlsx_parser          import convert_xlsx_to_text
#---------------------
#  Class definition --
#---------------------
class GUIMetrology ( QtGui.QWidget ) :
    """Main GUI for main button bar.

    @see BaseClass
    @see OtherClass
    """
    def __init__ (self, parent=None, app=None) :

        self.name = 'GUIMetrology'
        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.fname_prefix  = cp.fname_prefix
        self.fname_metrology_xlsx = cp.fname_metrology_xlsx
        self.fname_metrology_text = cp.fname_metrology_text
        self.img_arr = None

        cp.setIcons()

        self.setGeometry(10, 25, 650, 30)
        self.setWindowTitle('Metrology')
        #self.setWindowIcon(cp.icon_monitor)
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        self.setFrame()

        self.setParams()
  
        #self.titFileXlsx = QtGui.QLabel('File xlsx:')

        self.ediFileXlsx = QtGui.QLineEdit ( fnm.path_metrology_xlsx() )
        self.ediFileXlsx.setReadOnly(True)

        self.ediFileText = QtGui.QLineEdit ( fnm.path_metrology_text() )
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
        self.butSrc       = QtGui.QPushButton(self.source_name + cp.char_expand )

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
        self.grid.addWidget(self.butSrc,        self.grid_row+3, 1)
        self.grid.addWidget(self.butDeploy,     self.grid_row+4, 0)
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
        self.butSrc       .setToolTip('Select name of the detector')
 

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setParams(self) :
        #if self.path_fm_selected != '' :
        #    self.path_fm_selected = os.path.dirname(self.path_fm_selected)
        self.str_run_from     = '0'
        self.str_run_to       = 'end'
        self.source_name      = 'Select'
        self.calib_type       = 'Select'


    def setStyle(self):
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

        #self.butFBrowser.setVisible(False)
        #self.butSave.setText('')
        #self.butExit.setText('')
        #self.butExit.setFlat(True)

        self.setStyleButtons()


    def setStyleButtons(self):
        if self.source_name == 'Select' : self.butSrc.setStyleSheet(cp.stylePink)
        else                            : self.butSrc.setStyleSheet(cp.styleButton)

  
    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self.name) 
        self.frame.setGeometry(self.rect())


    def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', self.name)

        #try    : cp.guimain.close()
        #except : pass

        #try    : del cp.guimain
        #except : pass


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
        basename  = os.path.basename( fnm.path_metrology_ptrn() )
        fname, ext = os.path.splitext(basename)
        filter = 'Text files (' + fname + '*' + ext + ')\nAll files (*)'
        self.onButFile(but, edi, par, filter, set_path=False)


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
            
            cp.guifilebrowser = GUIFileBrowser(None, fnm.get_list_of_metrology_text_files(), fnm.path_metrology_text())
            cp.guifilebrowser.move(self.pos().__add__(QtCore.QPoint(880,40))) # open window with offset w.r.t. parent
            cp.guifilebrowser.show()


    def onButConvert(self):
        logger.debug('onButConvert', __name__)
        
        #ifname = fnm.path_metrology_xlsx()
        #ofname = fnm.path_metrology_text()
        list_ofnames = convert_xlsx_to_text(fnm.path_metrology_xlsx(), fnm.path_metrology_ptrn(), print_bits=0)

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


    def onButSrc(self):
        logger.info('onButSrc', __name__ )

        det = self.get_detector_selected()
        if det is None : return

        try    :
            lst = ru.list_of_sources_for_det(det)
        except :
            lst = cp.dict_of_det_sources[det]

        selected = gu.selectFromListInPopupMenu(lst)

        if selected is None : return            # selection is cancelled

        txt = str(selected)
        self.source_name = txt
        self.butSrc.setText( txt + cp.char_expand )
        logger.info('Source selected: ' + txt, __name__)

        self.setStyleButtons()

  
    def onButEvaluate(self):
        logger.info('onButEvaluate - NON IMPLEMENTED YET', __name__)        
 

    def onButDeploy(self):
        logger.info('onButDeploy - NON IMPLEMENTED YET', __name__)        

 
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
