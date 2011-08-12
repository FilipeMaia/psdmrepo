#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIGainCorrection...
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

import ConfigParameters        as cp
import GlobalMethods           as gm
import numpy                   as np
import FastArrayTransformation as fat

#---------------------
#  Class definition --
#---------------------
class GUIGainCorrection ( QtGui.QWidget ) :
    """GUI works with gain correction parameters management"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)

        #self.parent = cp.confpars.guimain

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Gain correction')

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())

        aveFileName = cp.confpars. aveDirName + '/' + cp.confpars. aveFileName        

        self.titFile     = QtGui.QLabel('Gain correction file:')
        self.titMake     = QtGui.QLabel('the gain correction file from' + aveFileName)

        self.butBrowse   = QtGui.QPushButton("Browse")
        self.butMake     = QtGui.QPushButton("Make")

        self.cboxApply   = QtGui.QCheckBox('Apply gain correction',self)

        if cp.confpars.gainCorrectionIsOn : self.cboxApply.setCheckState(2)
        else :                              self.cboxApply.setCheckState(0)

        #cp.confpars.confParsDirName  = os.getenv('HOME')
        path          = cp.confpars.gainDirName + '/' + cp.confpars.gainFileName
        self.fileEdit = QtGui.QLineEdit(path)
        
        self.showToolTips()

        grid = QtGui.QGridLayout()
        grid.addWidget(self.cboxApply,     0, 0, 1, 5)    
        grid.addWidget(self.titFile,       2, 0)    
        grid.addWidget(self.fileEdit,      3, 0, 1, 5)    
        grid.addWidget(self.butBrowse,     3, 5)    
        grid.addWidget(self.butMake,       4, 0)
        grid.addWidget(self.titMake,       4, 1, 1, 5)

        vbox = QtGui.QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(grid)
        vbox.addStretch(1)
        self.setLayout(vbox)

        self.connect(self.butBrowse,    QtCore.SIGNAL('clicked()'),          self.processBrowse  )
        self.connect(self.butMake,      QtCore.SIGNAL('clicked()'),          self.processMake    )
        self.connect(self.fileEdit,     QtCore.SIGNAL('editingFinished ()'), self.processFileEdit)
        self.connect(self.cboxApply,    QtCore.SIGNAL('stateChanged(int)'),  self.processCBoxApply)

        cp.confpars.gainGUIIsOpen = True


    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        self.fileEdit  .setToolTip('Type the file path name here,\nor better use "Browse" button.')
        self.butBrowse .setToolTip('Select the file path name\n for gain correction.')
        self.butMake   .setToolTip('Make the gain correction file from the latest averaged image.')

    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        cp.confpars.gainGUIIsOpen = False
            
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
            cp.confpars.gainCorrectionIsOn = True
            self.loadGainCorrectionArrayFromFile()
        else:
            cp.confpars.gainCorrectionIsOn = False

    def processMake(self):
        src = cp.confpars. aveDirName + '/' + cp.confpars. aveFileName
        dst = cp.confpars.gainDirName + '/' + cp.confpars.gainFileName
        print 'Make the file', dst, 'from', src       
        self.makeGainCorrectionFile(src,dst);

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
            cp.confpars.gainDirName  = self.dirName
            cp.confpars.gainFileName = self.fileName

    def processFileEdit(self):
        print 'FileEdit'
        self.path = str(self.fileEdit.displayText())
        cp.confpars.gainDirName,cp.confpars.gainFileName = os.path.split(self.path)
        print 'Set dirName  : %s' % (cp.confpars.gainDirName)
        print 'Set fileName : %s' % (cp.confpars.gainFileName)

    def makeGainCorrectionFile(self, src, dst):
        arr_ave       = gm.getNumpyArrayFromFile(fname=src, datatype=np.float32)
        arr_gain_corr = fat.getGainCorrectionArrayFromAverage(arr_ave)
        gm.saveNumpyArrayInFile(arr_gain_corr,  fname=dst , format='%f') # , format='%i')

    def loadGainCorrectionArrayFromFile(self):
        gcfname = cp.confpars.gainDirName + '/' + cp.confpars.gainFileName
        cp.confpars.arr_gain = gm.getNumpyArrayFromFile(fname=gcfname, datatype=np.float32)
 
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIGainCorrection ()
    widget.show()
    app.exec_()

#-----------------------------
