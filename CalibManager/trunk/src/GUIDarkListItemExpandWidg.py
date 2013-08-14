#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDarkListItemExpandWidg ...
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

from GUIAnyFilesStatusTable import *
from GUIDarkMoreOpts        import *

#---------------------
#  Class definition --
#---------------------
class GUIDarkListItemExpandWidg ( QtGui.QWidget ) :
    """GUI sets the source dark run number, validity range, and starts calibration of pedestals"""

    def __init__ ( self, parent=None, run_number='0000') :

        QtGui.QWidget.__init__(self, parent)

        self.parent     = parent
        self.run_number = run_number

        self.setGeometry(100, 100, 600, 70)
        self.setWindowTitle('GUI Dark List Item')
        self.setFrame()

        dir_xtc = fnm.path_to_xtc_dir()
        list_of_files = gu.get_list_of_files_in_dir_for_part_fname(dir_xtc, pattern='-r'+self.run_number)
        #print 'list_of_files =', list_of_files

        #self.lab_dir = QtGui.QLabel('xtc dir: %s' % dir_xtc)
        self.gui_table = GUIAnyFilesStatusTable(self, list_of_files)
        self.gui_more  = GUIDarkMoreOpts(self, self.run_number)
        
        self.vbox = QtGui.QVBoxLayout()
        #self.vbox.addWidget(self.lab_dir)
        self.vbox.addWidget(self.gui_table)
        self.vbox.addWidget(self.gui_more)
        #self.vbox.addStretch(1)     

        self.setLayout(self.vbox)

        self.showToolTips()

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
        self.frame.setVisible(False)


    def setStyle(self):
        #self.gui_table.setStyleSheet(cp.styleYellowish)
        #self.gui_table.setMinimumHeight(100)

        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))


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

        try    : self.gui_table.close()
        except : pass

        try    : del self.gui_table
        except : pass

        self.gui_table = None


    def onClose(self):
        logger.info('onClose', __name__)
        self.close()


    def setStatusMessage(self):
        if cp.guistatus is None : return
        #msg = 'From %s to %s use dark run %s' % (self.str_run_from.value(), self.str_run_to.value(), self.str_run_number.value())
        #msg = gu.get_text_content_of_calib_dir_for_detector(path=self.calib_dir.value(), det=self.det_name.value(), calib_type='pedestals')
        #cp.guistatus.setStatusMessage(msg)


    def getHeight(self):
        logger.debug('getHeight', __name__)
        h=0
        #h += 30 # self.lab_dir.height()
        if self.gui_table is not None :
            h += self.gui_table.height()
        h += 50 # for self.gui_more
        return h 
        
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    w = GUIDarkListItemExpandWidg(parent=None, run_number='0005')
    w.show()
    app.exec_()

#-----------------------------
