#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDarkListItem ...
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

from GUIDarkRunItem         import *
from GUIDarkListItemExpandWidg import *
#from GUIAnyFilesStatusTable import *

#---------------------
#  Class definition --
#---------------------
class GUIDarkListItem ( QtGui.QWidget ) :
    """GUI sets the source dark run number, validity range, and starts calibration of pedestals"""

    char_expand    = u'\u25BC' # solid down-head triangle
    char_shrink    = u'\u25B6' # solid right-head triangle
    #char_expand    = u'\u25BD' # open down-head triangle
    #char_shrink    = u'\u25B7' # open right-head triangle

    def __init__ ( self, parent=None, run_number='0000') :

        QtGui.QWidget.__init__(self, parent)

        self.parent = parent

        self.setGeometry(100, 100, 600, 70)
        self.setWindowTitle('GUI Dark List Item')
        #try : self.setWindowIcon(cp.icon_help)
        #except : pass
        self.setFrame()

        self.list_of_runs    = None

        self.run_number = run_number # cp.str_run_number

        
        self.str_run_number = cp.str_run_number # cp.str_run_number
        #self.calib_dir      = cp.calib_dir
        #self.det_name       = cp.det_name
        
        self.but_exp = QtGui.QPushButton(self.char_expand)

        self.gui_add = None
        self.gui_run = GUIDarkRunItem(self, self.run_number)
        self.hboxTT = QtGui.QHBoxLayout()
        self.hboxTT.addSpacing(5)     
        self.hboxTT.addWidget(self.but_exp)
        self.hboxTT.addWidget(self.gui_run)
        self.hboxTT.addStretch(1)     

        self.hboxW = QtGui.QHBoxLayout()
        self.hboxWW = QtGui.QHBoxLayout()
        self.hboxWW.addStretch(1)     
        self.hboxWW.addLayout(self.hboxW)
        self.hboxWW.addStretch(1)     

        self.vbox = QtGui.QVBoxLayout()
        #self.vbox.addWidget(self.gui_run)
        self.vbox.addLayout(self.hboxTT)
        self.vbox.addLayout(self.hboxWW)
        self.vbox.addStretch(1)     

        self.setLayout(self.vbox)

        self.connect( self.but_exp,     QtCore.SIGNAL('clicked()'),          self.onButExpand  )
 
        #self.connect( self.but_go  , QtCore.SIGNAL('clicked()'),         self.onButGo )
        #self.connect( self.edi_from, QtCore.SIGNAL('editingFinished()'), self.onEdiFrom )
        #self.connect( self.edi_to  , QtCore.SIGNAL('editingFinished()'), self.onEdiTo )
   
        self.showToolTips()

        #self.setStatusMessage()
        #self.setFieldsEnabled(cp.det_name.value() != 'None')

        #cp.guidarkrunitem = self

        self.setStyle()


    def showToolTips(self):
        pass
        #self.lab_rnum.setToolTip('Data run for calibration.')
        #self.but_go  .setToolTip('Begin data processing for calibration.')
        #self.edi_from.setToolTip('Type in the run number \nas a lower limit of the validity range.')
        #self.edi_to  .setToolTip('Type in the run number or "end"\nas an upper limit of the validity range.')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def setFieldsEnabled(self, is_enabled=True):

        logger.info('Set fields enabled: %s' %  is_enabled, __name__)

        #self.lab_rnum .setEnabled(is_enabled)
        #self.but_go   .setEnabled(is_enabled)

        #self.edi_from .setReadOnly(not is_enabled)
        #self.edi_to   .setReadOnly(not is_enabled)

        #self.edi_from .setEnabled(is_enabled)
        #self.edi_to   .setEnabled(is_enabled)

        #self.run_number == 'None'

        #self.setStyle()


    def setStyle(self):
        self.setMinimumSize(600,30)
        self.           setStyleSheet (cp.styleBkgd)
        #self.           setStyleSheet (cp.styleYellowish)

        self.but_exp.setFixedSize(20,22)
        #self.but_exp.setStyleSheet(cp.styleButtonGood)
        self.but_exp.setStyleSheet(cp.stylePink)

        #self.lab_from.setStyleSheet(cp.styleLabel)
        #self.lab_to  .setStyleSheet(cp.styleLabel)
        #self.lab_run .setStyleSheet(cp.styleLabel)

        #self.lab_rnum .setFixedWidth(80)        
        #self.edi_from .setFixedWidth(40)
        #self.edi_to   .setFixedWidth(40)
        #self.but_go   .setFixedWidth(80)        

        #self.edi_from .setAlignment (QtCore.Qt.AlignRight)
        #self.edi_to   .setAlignment (QtCore.Qt.AlignRight)

        #if self.edi_from.isReadOnly() : self.edi_from.setStyleSheet (cp.styleEditInfo)
        #else                          : self.edi_from.setStyleSheet (cp.styleEdit)

        #if self.edi_to.isReadOnly()   : self.edi_to.setStyleSheet (cp.styleEditInfo)
        #else                          : self.edi_to.setStyleSheet (cp.styleEdit)

        #self.but_go.setEnabled(self.run_number != 'None' and self.lab_rnum.isEnabled())


        #self.setContentsMargins (QtCore.QMargins(0,0,0,0))
        #self.setContentsMargins (QtCore.QMargins(-5,-5,-5,-5))
        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))
        #self.setContentsMargins (QtCore.QMargins(0,5,0,0))


    def setParent(self,parent) :
        self.parent = parent


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

        try    : self.gui_add.close()
        except : pass

        try    : self.gui_run.close()
        except : pass


    def onClose(self):
        logger.info('onClose', __name__)
        self.close()


    def setStatusMessage(self):
        if cp.guistatus is None : return
        #msg = 'From %s to %s use dark run %s' % (self.str_run_from.value(), self.str_run_to.value(), self.str_run_number.value())
        #msg = gu.get_text_content_of_calib_dir_for_detector(path=self.calib_dir.value(), det=self.det_name.value(), calib_type='pedestals')
        #cp.guistatus.setStatusMessage(msg)


    def onButExpand(self):
        #logger.debug('onButExpand', __name__)
        but = self.but_exp

        if but.text() == self.char_expand :
            #print 'Click on expand button'
            but.setText(self.char_shrink)
            self.onClickExpand()
            self.parent.onItemExpand(self)

        elif but.text() == self.char_shrink :
            #print 'Click on shrink button'
            but.setText(self.char_expand)
            self.onClickShrink()
            self.parent.onItemShrink(self)
        

    def onClickExpand(self):
        logger.debug('onClickExpand', __name__)

        #self.gui_add = QtGui.QLabel('Additional information')
  
        #dir_xtc = fnm.path_to_xtc_dir()
        #list_of_files = gu.get_list_of_files_in_dir_for_part_fname(dir_xtc, pattern='-r'+self.run_number)
        #self.gui_add = GUIAnyFilesStatusTable(self, list_of_files)

        self.gui_add = GUIDarkListItemExpandWidg(self, self.run_number) 
        self.hboxW.addWidget(self.gui_add)

        #self.gui_add.setStyleSheet(cp.styleYellowish)
        #self.gui_run.setStyleSheet(cp.styleYellowish)
        #self.gui_add.setMinimumHeight(100)



    def onClickShrink(self):
        logger.debug('onClickShrink', __name__)

        #self.gui_add.setVisible(False)

        try    : self.gui_add.close()
        except : pass

        try    : del self.gui_add
        except : pass

        self.gui_add = None

        self.gui_run.setStyleSheet(cp.styleBkgd)



    def getHeight(self):
        logger.debug('getHeight', __name__)
        h = self.gui_run.height()
        if self.gui_add is not None :
            h += self.gui_add.getHeight()
        return h + 10
        
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    w = GUIDarkListItem(parent=None, run_number='0024')
    w.setFieldsEnabled(True)
    w.show()
    app.exec_()

#-----------------------------
