#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIBackground...
#
#------------------------------------------------------------------------

"""GUI works with configuration parameters management"""

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
from shutil import copy

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

import ConfigParameters as cp
import GlobalMethods    as gm
import numpy            as np

#---------------------
#  Class definition --
#---------------------
class GUIBackground ( QtGui.QWidget ) :
    """GUI works with background parameters management"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)

        #self.parent = cp.confpars.guimain

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Background')

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())

        aveFileName = cp.confpars. aveDirName + '/' + cp.confpars. aveFileName        

        self.titFile     = QtGui.QLabel('Background file:')
        self.titCopy     = QtGui.QLabel(aveFileName + ' to the background file')

        self.butBrowse   = QtGui.QPushButton("Browse")
        self.butCopy     = QtGui.QPushButton("Copy")

        self.cboxApply   = QtGui.QCheckBox('Apply background subtraction',self)

        if cp.confpars.bkgdSubtractionIsOn : self.cboxApply.setCheckState(2)
        else :                               self.cboxApply.setCheckState(0)

        #cp.confpars.confParsDirName  = os.getenv('HOME')
        path          = cp.confpars.bkgdDirName + '/' + cp.confpars.bkgdFileName
        self.fileEdit = QtGui.QLineEdit(path)
        
        self.showToolTips()

        grid = QtGui.QGridLayout()
        grid.addWidget(self.cboxApply,     0, 0, 1, 5)    
        grid.addWidget(self.titFile,       2, 0)    
        grid.addWidget(self.fileEdit,      3, 0, 1, 5)    
        grid.addWidget(self.butBrowse,     3, 5)    
        grid.addWidget(self.butCopy,       4, 0, 1, 1)
        grid.addWidget(self.titCopy,       4, 1, 1, 5)

        vbox = QtGui.QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(grid)
        vbox.addStretch(1)
        self.setLayout(vbox)

        self.connect(self.butBrowse,    QtCore.SIGNAL('clicked()'),          self.processBrowse  )
        self.connect(self.butCopy,      QtCore.SIGNAL('clicked()'),          self.processCopy    )
        self.connect(self.fileEdit,     QtCore.SIGNAL('editingFinished ()'), self.processFileEdit)
        self.connect(self.cboxApply,    QtCore.SIGNAL('stateChanged(int)'),  self.processCBoxApply)
        cp.confpars.bkgdGUIIsOpen = True


    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        self.fileEdit  .setToolTip('Type the file path name here,\nor better use "Browse" button.')
        self.butBrowse .setToolTip('Select the file path name\nfor background image file.')
        self.butCopy   .setToolTip('Copy the latest averaged image as a background file.')

    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        cp.confpars.bkgdGUIIsOpen = False
            
    def processExit(self):
        #print 'Exit button'
        self.close()

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def refreshGUIWhatToDisplay(self):
        cp.confpars.guiwhat.processRefresh()

    def processCBoxApply(self, value):
        print 'CBoxApply'
        if self.cboxApply.isChecked():
            cp.confpars.bkgdSubtractionIsOn = True
            self.loadBackgroundArrayFromFile()
        else:
            cp.confpars.bkgdSubtractionIsOn = False

    def processCopy(self):
        src = cp.confpars. aveDirName + '/' + cp.confpars. aveFileName
        dst = cp.confpars.bkgdDirName + '/' + cp.confpars.bkgdFileName
        print 'Copy the file', src, 'to', dst       
        copy(src,dst);

    def processBrowse(self):
        print 'Browse'
        self.path = str(self.fileEdit.displayText())
        self.dirName,self.fileName = os.path.split(self.path)
        print 'dirName  : %s' % (self.dirName)
        print 'fileName : %s' % (self.fileName)
        self.path = QtGui.QFileDialog.getOpenFileName(self,'Open file',self.dirName)
        self.dirName,self.fileName = os.path.split(str(self.path))
        if self.dirName == '' or self.fileName == '' :
            print 'Input dirName or fileName is empty... use default values'  
        else :
            self.fileEdit.setText(self.path)
            cp.confpars.bkgdDirName  = self.dirName
            cp.confpars.bkgdFileName = self.fileName

    def processFileEdit(self):
        print 'FileEdit'
        self.path = str(self.fileEdit.displayText())
        cp.confpars.bkgdDirName,cp.confpars.bkgdFileName = os.path.split(self.path)
        print 'Set dirName  : %s' % (cp.confpars.bkgdDirName)
        print 'Set fileName : %s' % (cp.confpars.bkgdFileName)

    def loadBackgroundArrayFromFile(self):
        bkgdfname = cp.confpars.bkgdDirName + '/' + cp.confpars.bkgdFileName
        cp.confpars.arr_bkgd = gm.getNumpyArrayFromFile(fname=bkgdfname, datatype=np.float32)
 
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIBackground ()
    widget.show()
    app.exec_()

#-----------------------------
