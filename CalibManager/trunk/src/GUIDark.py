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
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
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
from GUIDarkControlBar      import *
from GUIDarkList            import *

#-----------------------------

class GUIDark ( QtGui.QWidget ) :
    """GUI works with dark run"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 800, 300)
        self.setWindowTitle('Dark run processing')
        self.setFrame()

        self.guistatus   = GUIStatus(self)
        self.guidarkcbar = GUIDarkControlBar(self)
        self.guidarklist = GUIDarkList(self)

        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addWidget(self.guidarkcbar)
        self.vbox.addWidget(self.guidarklist)
        self.vwidg = QtGui.QWidget(self)
        self.vwidg.setLayout(self.vbox) 

        self.vsplit = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.vsplit.addWidget(self.guistatus)
        self.vsplit.addWidget(self.vwidg)

        self.hbox = QtGui.QHBoxLayout(self) 
        self.hbox.addWidget(self.vsplit)
        #self.hbox.addStretch(1)

        self.setLayout(self.hbox)

        self.showToolTips()
        self.setStyle()

        cp.guidark = self
        self.guistatus.updateStatusInfo()


    def showToolTips(self):
        self.setToolTip('Dark run GUI')
        pass


    def setStyle(self):

        self.setContentsMargins (QtCore.QMargins(-5,-5,-5,2))
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        #self.vsplit.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Ignored)
        #self.setMinimumSize(790,210)
        #self.setMinimumHeight(320)
        #self.vsplit.setMinimumHeight(200)
        #self.vsplit.setHandleWidth(150)
        #self.vsplit.moveSplitter(10, self.vsplit.indexOf(self.guistatus))
        #self.vsplit.moveSplitter(300, self.vsplit.indexOf(self.vwidg))
        #self.setBaseSize(750,700)
        #self.setStyleSheet(cp.styleBkgd)

  
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
        #print 'GUIDark resizeEvent: %s' % str(self.size())


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

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIDark ()
    widget.show()
    app.exec_()

#-----------------------------
#-----------------------------
