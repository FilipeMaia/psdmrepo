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

from GUIStatus              import *
from GUIDarkList            import *
from GUIDarkMoreOpts        import *
from GUIDarkControlBar      import *

#-----------------------------

class GUIDark ( QtGui.QWidget ) :
    """GUI works with dark run"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 800, 300)
        self.setWindowTitle('Dark run processing')
        self.setFrame()

        self.guistatus        = GUIStatus(self)
        self.guidarklist      = GUIDarkList(self)
        self.guidarkcbar      = GUIDarkControlBar(self)
        #self.guidarkmoreopts  = GUIDarkMoreOpts(self)

        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addWidget(self.guidarkcbar)
        self.vbox.addWidget(self.guidarklist)
        self.vwidg = QtGui.QWidget(self)
        self.vwidg.setLayout(self.vbox) 

        self.vsplit = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.vsplit.addWidget(self.guistatus)
        self.vsplit.addWidget(self.vwidg)
        
        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addWidget(self.vsplit)
        #self.hbox.addWidget(self.guistatus)
        #self.hbox.addStretch(1)
        #self.vbox.addWidget(self.guidarklist)
        #self.vbox.addStretch(1)
        #self.vbox.addWidget(self.guidarkmoreopts)

        self.setLayout(self.hbox)

        self.showToolTips()
        self.setStyle()

        cp.guidark = self
        self.guistatus.updateStatusInfo()

    def showToolTips(self):
        pass
        #self           .setToolTip('Use this GUI to work with xtc file.')
        #self.edi_path   .setToolTip('The path to the xtc file for processing in this GUI')

    def setStyle(self):

        self.setContentsMargins (QtCore.QMargins(-5,-5,-5,2))

        #self.vsplit.setMinimumHeight(200)
        self.vsplit.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Ignored)
        #self.vsplit.setHandleWidth(150)

        #self.vsplit.moveSplitter(200, self.vsplit.indexOf(self.guidarklist))

        #self.setMinimumHeight(500)
        #self.setBaseSize(750,700)

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
        #self.setGeometry(self.contentsRect())
        #w, h = self.size().width(), self.size().height()
        #self.guistatus  .setMinimumHeight(0.3*h)
        #self.guidarklist.setMinimumHeight(0.5*h)


    def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)

        try    : self.guistatus.close()
        except : pass

        try    : self.guidarklist.close()        
        except : pass

        try    : self.guidarkcbar.close()        
        except : pass

        #try    : self.guidarkmoreopts.close()        
        #except : pass

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
