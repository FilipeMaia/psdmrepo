#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIConfiguration...
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

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

import ConfigParameters as cp

#---------------------
#  Class definition --
#---------------------
class GUIConfiguration ( QtGui.QWidget ) :
    """GUI works with configuration parameters management"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(370, 350, 500, 200)
        self.setWindowTitle('Configuration')

        self.titFile     = QtGui.QLabel('File with configuration parameters:')
        self.titPars     = QtGui.QLabel('Operations on configuration parameters:')
        self.butBrowse   = QtGui.QPushButton("Browse")
        self.butRead     = QtGui.QPushButton("Read")
        self.butWrite    = QtGui.QPushButton("Write")
        self.butPrint    = QtGui.QPushButton("Print current")
        self.butExit     = QtGui.QPushButton("Exit")
        
        #cp.confpars.confParsDirName  = os.getenv('HOME')
        path          = cp.confpars.confParsDirName + '/' + cp.confpars.confParsFileName
        self.fileEdit = QtGui.QLineEdit(path)
        
        hboxT1 = QtGui.QHBoxLayout()
        hboxT1.addWidget(self.titFile)

        hboxT2 = QtGui.QHBoxLayout()
        hboxT2.addWidget(self.titPars)

        hboxF = QtGui.QHBoxLayout()
        hboxF.addWidget(self.fileEdit)
        hboxF.addWidget(self.butBrowse)

        hboxA = QtGui.QHBoxLayout()
        hboxA.addWidget(self.butRead)
        hboxA.addStretch(1)
        hboxA.addWidget(self.butWrite)
        hboxA.addStretch(1)
        hboxA.addWidget(self.butPrint)

        hboxE = QtGui.QHBoxLayout()
        hboxE.addStretch(1)
        hboxE.addWidget(self.butExit)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hboxT1)
        vbox.addLayout(hboxF)
        vbox.addStretch(1)     
        vbox.addLayout(hboxT2)
        vbox.addLayout(hboxA)
        vbox.addStretch(1)     
        vbox.addLayout(hboxE)

        self.setLayout(vbox)

        self.connect(self.butExit,   QtCore.SIGNAL('clicked()'), self.processExit  )
        self.connect(self.butRead,   QtCore.SIGNAL('clicked()'), self.processRead  )
        self.connect(self.butWrite,  QtCore.SIGNAL('clicked()'), self.processWrite )
        self.connect(self.butPrint,  QtCore.SIGNAL('clicked()'), self.processPrint )
        self.connect(self.butBrowse, QtCore.SIGNAL('clicked()'), self.processBrowse)

    #-------------------
    #  Public methods --
    #-------------------

    def closeEvent(self, event):
        print 'closeEvent'
        self.processExit()
            
    def processExit(self):
        print 'Exit'
        self.close()
        cp.confpars.configGUIIsOpen = False

    def processRead(self):
        print 'Read'
        cp.confpars.readParameters(self.confParsFileName())
        
    def processWrite(self):
        print 'Write'
        cp.confpars.writeParameters(self.confParsFileName())

    def processPrint(self):
        print 'Print'
        cp.confpars.Print()

    def processBrowse(self):
        print 'Browse'
        self.path = str(self.fileEdit.displayText())
        self.dirName,self.fileName = os.path.split(self.path)
        print 'dirName  : %s' % (self.dirName)
        print 'fileName : %s' % (self.fileName)
        self.path = QtGui.QFileDialog.getOpenFileName(self,'Open file',self.dirName)
        #self.path = cp.confpars.dirName + '/' + cp.confpars.fileName
        self.path = self.dirName+'/'+self.fileName
        self.fileEdit.setText(self.path)
 
    def confParsFileName(self):
        #One have to check that file exists...
        fname = str(self.fileEdit.displayText())
        return fname

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIConfiguration ()
    widget.show()
    app.exec_()

#-----------------------------
