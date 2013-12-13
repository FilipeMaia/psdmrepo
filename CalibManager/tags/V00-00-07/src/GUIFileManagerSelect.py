
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIFileManagerSelect...
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

from CorAna.MaskEditor import MaskEditor
import GlobalUtils     as     gu
import RegDBUtils      as     ru
from GUIFileBrowser         import *
from PlotImgSpe             import *
from FileDeployer           import fd

#---------------------
#  Class definition --
#---------------------
class GUIFileManagerSelect ( QtGui.QWidget ) :
    """Main GUI for main button bar.

    @see BaseClass
    @see OtherClass
    """
    char_expand    = u' \u25BE' # down-head triangle

    def __init__ (self, parent=None, app=None) :

        self.name = 'GUIFileManagerSelect'
        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(10, 25, 630, 120)
        self.setWindowTitle('File Maneger Select & Action GUI')
        #self.setWindowIcon(cp.icon_monitor)
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        self.setFrame()

        #cp.setIcons()

        self.path_fm_selected = ''
        self.setParams()
 
        self.lab_src        = QtGui.QLabel('for detector')
        self.lab_type       = QtGui.QLabel('calib type')
        self.lab_from       = QtGui.QLabel('valid from run')
        self.lab_to         = QtGui.QLabel('to')
        self.edi_from       = QtGui.QLineEdit  ( self.str_run_from )
        self.edi_to         = QtGui.QLineEdit  ( self.str_run_to )
        self.but_src        = QtGui.QPushButton(self.source_name + self.char_expand )
        self.but_type       = QtGui.QPushButton(self.calib_type + self.char_expand )

        self.edi_from.setValidator(QtGui.QIntValidator(0,9999,self))
        self.edi_to  .setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]\\d{0,3}|end$"),self))
 
        self.lab_file = QtGui.QLabel('File:')
        self.edi_file = QtGui.QLineEdit ( self.path_fm_selected ) # fnm.path_to_calib_dir() )
        self.edi_file.setReadOnly(True)
 
        self.but_copy   = QtGui.QPushButton('Copy')
        self.but_delete = QtGui.QPushButton('Delete')
        self.but_view   = QtGui.QPushButton('View')
        self.but_plot   = QtGui.QPushButton('Plot')
        #self.but_copy  .setIcon(cp.icon_monitor)

        self.but_browse = QtGui.QPushButton( 'Browse' )

        self.hboxB = QtGui.QHBoxLayout() 
        #self.hboxB.addStretch(1)     
        self.hboxB.addWidget( self.lab_file )
        self.hboxB.addWidget( self.edi_file )
        self.hboxB.addWidget( self.but_browse )
        self.hboxB.addStretch(1)     

        self.hboxD = QtGui.QHBoxLayout() 
        #self.hboxD.addSpacing(50)
        self.hboxD.addWidget( self.but_view   )
        self.hboxD.addWidget( self.but_plot   )
        self.hboxD.addWidget( self.but_delete )
        self.hboxD.addStretch(1)     

        self.hboxC = QtGui.QHBoxLayout() 
        #self.hboxC.addStretch(1)     
        self.hboxC.addWidget( self.but_copy )
        self.hboxC.addWidget( self.lab_src  )
        self.hboxC.addWidget( self.but_src  )
        self.hboxC.addWidget( self.lab_type )
        self.hboxC.addWidget( self.but_type )
        self.hboxC.addWidget( self.lab_from )
        self.hboxC.addWidget( self.edi_from )
        self.hboxC.addWidget( self.lab_to   )
        self.hboxC.addWidget( self.edi_to   )
        self.hboxC.addStretch(1)     


        self.vboxW = QtGui.QVBoxLayout() 
        self.vboxW.addStretch(1)
        self.vboxW.addLayout( self.hboxB ) 
        self.vboxW.addLayout( self.hboxD ) 
        self.vboxW.addLayout( self.hboxC ) 
        self.vboxW.addStretch(1)
        
        self.setLayout(self.vboxW)

        self.connect( self.but_browse, QtCore.SIGNAL('clicked()'), self.onButBrowse ) 
        self.connect( self.but_copy,   QtCore.SIGNAL('clicked()'), self.onButCopy   ) 
        self.connect( self.but_view,   QtCore.SIGNAL('clicked()'), self.onButView   ) 
        self.connect( self.but_plot,   QtCore.SIGNAL('clicked()'), self.onButPlot   ) 
        self.connect( self.but_delete, QtCore.SIGNAL('clicked()'), self.onButDelete ) 
        self.connect( self.but_src,    QtCore.SIGNAL('clicked()'), self.onButSrc    ) 
        self.connect( self.but_type,   QtCore.SIGNAL('clicked()'), self.onButType   ) 
        self.connect( self.edi_from,   QtCore.SIGNAL('editingFinished()'), self.onEdiFrom )
        self.connect( self.edi_to,     QtCore.SIGNAL('editingFinished()'), self.onEdiTo )
  
        self.showToolTips()
        self.setStyle()

        cp.guifilemanagerselect = self
        self.move(10,25)
        
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        #pass
        self.edi_file  .setToolTip('Path to the file with image data') 
        self.but_browse.setToolTip('Open file browser dialog\nwindow and select the file') 
        self.but_src   .setToolTip('Select name of the detector')
        self.but_type  .setToolTip('Select type of calibration parameters')
        self.but_copy  .setToolTip('Copy selected file')
        self.but_view  .setToolTip('Launch file browser')
        self.but_plot  .setToolTip('Launch plot browser')
        self.but_delete.setToolTip('Delete selected file\nDelete  is allowed for\nWORK or CALIB directories only')
        self.edi_from  .setToolTip('Enter run number in range [0,9999]')
        self.edi_to    .setToolTip('Enter run number in range [0,9999] or "end"')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setStyle(self):
        self.          setStyleSheet(cp.styleBkgd)
        self.but_copy  .setStyleSheet(cp.styleButton)
        self.but_delete.setStyleSheet(cp.styleButton)
        self.but_view  .setStyleSheet(cp.styleButton)
        self.but_plot  .setStyleSheet(cp.styleButton)
        
        self.setMinimumSize(630,100)
        self.setFixedHeight(100)
        self.setContentsMargins (QtCore.QMargins(0,-9,0,-9))

        #self.but_copy.setFixedWidth(100)
        #self.but_delete.setFixedWidth(100)
        #self.but_copy.setMinimumHeight(60)

        self.edi_from.setFixedWidth(40)
        self.edi_to  .setFixedWidth(40)

        self.edi_file.setFixedWidth(490)
        self.edi_file.setStyleSheet(cp.styleEditInfo) 
        self.edi_file.setEnabled(False)            

        self.lab_file  .setStyleSheet(cp.styleLabel)
        self.lab_src   .setStyleSheet(cp.styleLabel)
        self.lab_type  .setStyleSheet(cp.styleLabel)
        self.lab_from  .setStyleSheet(cp.styleLabel)
        self.lab_to    .setStyleSheet(cp.styleLabel)
 
        #self.butViewwser.setVisible(False)
        #self.butSave.setText('')
        #self.butExit.setText('')
        #self.butExit.setFlat(True)

        self.setStyleButtons()


    def setStyleButtons(self):
        """Set buttons enabled or disabled depending on status of other fields
        """

        file_is_enable = True
        if self.str_path() == '' :
            file_is_enable = False
            self.but_browse.setStyleSheet(cp.styleButtonGood)
        else :
            self.but_browse.setStyleSheet(cp.styleButton)

        self.but_view.setEnabled(file_is_enable)
        self.but_plot.setEnabled(file_is_enable)

        #print '\nself.str_path()', self.str_path()
        #print 'fnm.path_dir_work()', fnm.path_dir_work().lstrip('.')

        is_enable_delete = file_is_enable \
                           and (self.str_path().find(fnm.path_to_calib_dir()) != -1
                                or self.str_path().find(fnm.path_dir_work().lstrip('.')) != -1
                                )
        self.but_delete.setEnabled(is_enable_delete)

        is_enable_copy = file_is_enable \
                         and self.source_name != 'Select' \
                         and self.calib_type != 'Select' 
        self.but_copy  .setEnabled(is_enable_copy)
        
        if self.source_name == 'Select' : self.but_src .setStyleSheet(cp.stylePink)
        else                            : self.but_src .setStyleSheet(cp.styleButton)
        if self.calib_type  == 'Select' : self.but_type.setStyleSheet(cp.stylePink)
        else                            : self.but_type.setStyleSheet(cp.styleButton)

        #self.setButtonDelete()
        #self.setButtonCopy()


    #def setButtonDelete(self):

    #def setButtonCopy(self):
        #self.but.setVisible(False)
        #self.but.setEnabled(True)
 

    def setParams(self) :
        if self.path_fm_selected != '' :
            self.path_fm_selected = os.path.dirname(self.path_fm_selected)
        self.str_run_from     = '0'
        self.str_run_to       = 'end'
        self.source_name      = 'Select'
        self.calib_type       = 'Select'


    def resetFields(self) :
        self.setParams()
        self.edi_file  .setText(self.path_fm_selected)
        self.edi_from  .setText(self.str_run_from)
        self.edi_to    .setText(self.str_run_to)
        self.but_src   .setText(self.source_name + self.char_expand )
        self.but_type  .setText(self.calib_type + self.char_expand )
        self.setStyleButtons()


    def resetFieldsOnDelete(self) :
        self.path_fm_selected = os.path.dirname(self.path_fm_selected) # ''
        self.edi_file  .setText(self.path_fm_selected)
        self.setStyleButtons()


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self.name) 
        self.frame.setGeometry(self.rect())
        #print 'GUIFileManagerSelect resizeEvent: %s' % str(self.size())


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

        try    : cp.plotimgspe.close()
        except : pass


    def onExit(self):
        logger.debug('onExit', self.name)
        self.close()


    def onButCopy(self):
        #logger.info('onButCopy', __name__)
        cmd = 'cp %s %s' % (self.str_path(), self.get_out_path())
        if self.approveCommand(self.but_copy, cmd) :
            #os.system(cmd)
            fd.procDeployCommand(cmd)


    def onButDelete(self):
        #logger.info('onButDelete', __name__)
        cmd = 'rm %s' % self.str_path()
        if self.approveCommand(self.but_delete, cmd) :
            os.system(cmd)
            self.resetFieldsOnDelete()
            

    def approveCommand(self, but, cmd):
        msg = 'Approve command:\n' + cmd
        resp = gu.confirm_or_cancel_dialog_box(parent=but, text=msg, title='Please confirm or cancel!')
        if resp :
            logger.info('Approved command:\n' + cmd, __name__)
        return resp

    def selectDirFromPopupMenu(self, dir_current='.'):

        list_of_opts = [  'Use WORK directory'
                         ,'Use CALIB directory'
                         ,'Use CURRENT FILE directory'
                         ,'Use OTHER EXPERIMENT directory'
                         ,'Use ./'
                         ,'Reset'
                         ,'Cancel'
                        ]

        selected = gu.selectFromListInPopupMenu(list_of_opts)
        logger.info('selected option: %s' % selected, __name__ )

        if   selected == list_of_opts[0] : return fnm.path_dir_work()
        elif selected == list_of_opts[1] : return fnm.path_to_calib_dir()
        elif selected == list_of_opts[2] : return dir_current
        elif selected == list_of_opts[3] : return os.path.join(fnm.path_to_calib_dir(),'../../')
        elif selected == list_of_opts[4] : return './'
        elif selected == list_of_opts[5] : return ''
        elif selected == list_of_opts[6] : return None
        else                             : return None
 

    def get_out_path(self):
        det = self.get_detector_selected()
        if det is None : return
        calib_dir = fnm.path_to_calib_dir()
        calib_type = cp.dict_of_det_calib_types[det]
        fname = '%s-%s.data' % (self.str_run_from, self.str_run_to)
        path = os.path.join(calib_dir, calib_type, self.source_name, self.calib_type, fname)
        return path


    def str_path(self):
        return str(self.edi_file.displayText())


    def onButBrowse(self):
        logger.debug('onButBrowse', __name__)
        path0 = self.selectDirFromPopupMenu( self.str_path() )
        if path0 is None : return
        if path0 is '' :
            path = path0
        else :
            file_filter = 'Files (*.txt *.data *.dat HISTORY)\nAll files (*)'
            path = gu.get_open_fname_through_dialog_box(self, path0, 'Select file', filter=file_filter)
            if path == None or path == '' :
                logger.debug('File selection is cancelled...', __name__ )
                return

        self.edi_file.setText(path)
        self.path_fm_selected = path # .setValue(path)
        logger.debug('Selected file:\n' + path, __name__)

        self.setStyleButtons()


    def onEdiFrom(self):
        logger.debug('onEdiFrom', __name__ )
        self.str_run_from = str( self.edi_from.displayText() )        
        msg = 'Set the run validity range from %s' % self.str_run_from
        logger.info(msg, __name__ )


    def onEdiTo(self):
        logger.debug('onEdiTo', __name__ )
        self.str_run_to = str( self.edi_to.displayText() )        
        msg = 'Set the run validity range up to %s' % self.str_run_to
        logger.info(msg, __name__ )


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


    def onButType(self):
        logger.debug('onButType', __name__ )

        det = self.get_detector_selected()
        if det is None : return

        lst = cp.dict_of_det_const_types[det]
        selected = gu.selectFromListInPopupMenu(lst)
        if selected is None : return            # selection is cancelled

        txt = str(selected)
        self.calib_type = txt
        self.but_type.setText( txt + self.char_expand )
        logger.debug('Type selected: ' + txt, __name__)

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

        if selected is None : return            # selection is cancelled

        txt = str(selected)
        self.source_name = txt
        self.but_src.setText( txt + self.char_expand )
        logger.info('Source selected: ' + txt, __name__)

        self.setStyleButtons()


    def onButView(self):
        #self.exportLocalPars()
        logger.debug('onButView', __name__)
        try    :
            cp.guifilebrowser.close()
        except :
            cp.guifilebrowser = GUIFileBrowser(None, [self.str_path()], self.str_path())
            cp.guifilebrowser.move(self.pos().__add__(QtCore.QPoint(880,40))) # open window with offset w.r.t. parent
            cp.guifilebrowser.show()


    def onButPlot(self):
        logger.debug('onButPlot', __name__)

        logger.debug('onButPlot', __name__)
        try :
            cp.plotimgspe.close()
            try    : del cp.plotimgspe
            except : pass

        except :
            ifname = self.str_path()
            ofname = os.path.join(fnm.path_dir_work(),'image.png')
            tit = 'Plot for %s' % os.path.basename(ifname)            
            cp.plotimgspe = PlotImgSpe(None, ifname=ifname, ofname=ofname, title=tit, load_is_visible=True)
            cp.plotimgspe.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.plotimgspe.show()

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIFileManagerSelect()
    ex.show()
    app.exec_()
#-----------------------------
