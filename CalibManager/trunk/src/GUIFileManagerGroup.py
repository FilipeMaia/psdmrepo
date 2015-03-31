#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIFileManagerGroup...
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

from CalibManager.Frame     import Frame
from ConfigParametersForApp import cp
from Logger                 import logger

from GUIStatus                  import *
from GUIFileManagerGroupControl import *
from GUIDirTree                 import *
from GUIExpCalibDir             import *

#-----------------------------

#class GUIFileManagerGroup ( QtGui.QWidget ) :
class GUIFileManagerGroup ( Frame ) :
    """GUI works with dark run"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        Frame.__init__(self, parent, mlw=1)

        self.setGeometry(200, 400, 800, 300)
        self.setWindowTitle('Group File Manager')
        #self.setFrame()

        self.guistatus   = GUIStatus(self)
        self.guifilemanagergroupcontrol = GUIFileManagerGroupControl(self)
        self.guidirtree      = GUIDirTree(None, cp.calib_dir_src.value())
        self.guiexpcalibdir  = GUIExpCalibDir()
        #self.guisrcfile  = QtGui.QTextEdit('Source file GUI is not implemented.') # GUIDark(self)
        
        #self.hbox = QtGui.QHBoxLayout() 
        #self.hbox.addWidget(self.guifilemanagergroupcontrol)
        #self.hbox.addWidget(self.guistatus)

        self.hsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.hsplitter.addWidget(self.guidirtree)
        self.hsplitter.addWidget(self.guifilemanagergroupcontrol)
        self.hsplitter.addWidget(self.guistatus)
        
        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addWidget(self.guiexpcalibdir)
        self.vbox.addWidget(self.hsplitter)
        #self.vbox.addLayout(self.hbox)

        #self.vwidg = QtGui.QWidget(self)
        #self.vwidg.setLayout(self.box) 

        #self.vsplit = QtGui.QSplitter(QtCore.Qt.Vertical)
        #self.vsplit.addWidget(self.guistatus)
        #self.vsplit.addWidget(self.guisrcfile)

        #self.hbox.addStretch(1)

        self.setLayout(self.vbox)

        self.showToolTips()
        self.setStyle()
        self.guistatus.updateStatusInfo()

        cp.guifilemanagergroup = self


    def showToolTips(self):
        self.setToolTip('Dark run GUI')
        pass


    def setStyle(self):

        self.setContentsMargins (QtCore.QMargins(-5,-5,-5,2))
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        #self.hbox.moveSplitter(250, self.hbox.indexOf(self.guifilemanagergroupcontrol))
        self.hsplitter.setSizes([200,100,600])

        #self.vsplit.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Ignored)
        #self.setMinimumSize(790,210)
        #self.setMinimumHeight(320)
        #self.vsplit.setMinimumHeight(200)
        #self.vsplit.setHandleWidth(150)
        #self.vsplit.moveSplitter(10, self.vsplit.indexOf(self.guistatus))
        #self.vsplit.moveSplitter(300, self.vsplit.indexOf(self.vwidg))
        #self.setBaseSize(750,700)
        #self.setStyleSheet(cp.styleBkgd)

  
#    def setFrame(self):
#        self.frame = QtGui.QFrame(self)
#        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
#        self.frame.setLineWidth(0)
#        self.frame.setMidLineWidth(1)
#        self.frame.setGeometry(self.rect())
#        self.frame.setVisible(False)


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self.name)
        #self.frame.setGeometry(self.rect())
        #print 'GUIFileManagerGroup resizeEvent: %s' % str(self.size())
        pass


    def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass


    def resetFields(self) :
        #logger.info('resetFields', __name__)
        self.guiexpcalibdir.setStyleButtons()
        self.guifilemanagergroupcontrol.resetFields()


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)

        try    : self.guistatus.close()
        except : pass

        try    : self.guidirtree.close()
        except : pass

        try    : self.guifilemanagergroupcontrol.close()
        except : pass
 
        try    : self.guiexpcalibdir.close()
        except : pass

        cp.guifilemanagergroup = None

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIFileManagerGroup ()
    widget.show()
    app.exec_()

#-----------------------------
#-----------------------------
