#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDark...
#
#------------------------------------------------------------------------

"""GUI works with dark run"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#-------------------
#  Import modules --
#-------------------
import sys
import os

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

from ConfigParametersForApp import cp
from Logger                 import logger

from GUIInsExpDirDet        import *
from GUIStatus              import *
from GUIDarkRunGo           import *
from GUIDarkMoreOpts        import *

#-----------------------------

class GUIDark ( QtGui.QWidget ) :
    """GUI works with dark run"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 750, 300)
        self.setWindowTitle('Dark run processing')
        self.setFrame()

        self.guiinsexpdirdet  = GUIInsExpDirDet(self)
        self.guistatus        = GUIStatus(self)
        self.guidarkrungo     = GUIDarkRunGo(self)
        self.guidarkmoreopts  = GUIDarkMoreOpts(self)
        
        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addWidget(self.guiinsexpdirdet)
        self.vbox.addStretch(1)
        self.vbox.addWidget(self.guistatus)
        self.vbox.addStretch(1)
        self.vbox.addWidget(self.guidarkrungo)
        self.vbox.addStretch(1)
        self.vbox.addWidget(self.guidarkmoreopts)
        self.vbox.addStretch(1)

        self.setLayout(self.vbox)

        self.showToolTips()
        self.setStyle()

        cp.guidark = self
 
    def showToolTips(self):
        pass
        #self           .setToolTip('Use this GUI to work with xtc file.')
        #self.edi_path   .setToolTip('The path to the xtc file for processing in this GUI')

    def setStyle(self):
        pass
        #width = 60
        #self.setMinimumWidth(700)
        #self.setStyleSheet(cp.styleBkgd)
        #tit0   .setStyleSheet (cp.styleTitle)
        #self.guidarkmoreopts.setFixedHeight(100)

        #self.cbx_all_chunks.setStyleSheet (cp.styleLabel)
        #self.lab_status    .setStyleSheet (cp.styleLabel)
        #self.lab_batch     .setStyleSheet (cp.styleLabel)

  
    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

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
        logger.info('closeEvent', __name__)

        try    : self.guiinsexpdirdet.close()
        except : pass

        try    : self.guistatus.close()
        except : pass

        try    : self.guidarkrungo.close()        
        except : pass

        try    : self.guidarkmoreopts.close()        
        except : pass

        #if cp.res_save_log : 
        #    logger.saveLogInFile     ( fnm.log_file() )
        #    logger.saveLogTotalInFile( fnm.log_file_total() )

        #try    : self.gui_win.close()
        #except : pass

        #try    : del cp.guimain
        #except : pass

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIDark ()
    widget.show()
    app.exec_()

#-----------------------------
#-----------------------------
