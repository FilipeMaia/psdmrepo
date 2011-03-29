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

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Configuration')

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        
        self.titFile     = QtGui.QLabel('File with configuration parameters:')
        self.titPars     = QtGui.QLabel('Operations on configuration parameters:')
        self.titRadio    = QtGui.QLabel('At program start:')

        self.butBrowse   = QtGui.QPushButton("Browse")
        self.butRead     = QtGui.QPushButton("Read")
        self.butWrite    = QtGui.QPushButton("Write")
        self.butDefault  = QtGui.QPushButton("Reset default")
        self.butPrint    = QtGui.QPushButton("Print current")
        self.butExit     = QtGui.QPushButton("Quit")

        self.radioRead   = QtGui.QRadioButton("Read parameters from file")
        self.radioDefault= QtGui.QRadioButton("Set default")
        self.radioGroup  = QtGui.QButtonGroup()
        self.radioGroup.addButton(self.radioRead)
        self.radioGroup.addButton(self.radioDefault)
        if cp.confpars.readParsFromFileAtStart : self.radioRead.setChecked(True)
        else :                                   self.radioDefault.setChecked(True)

        #cp.confpars.confParsDirName  = os.getenv('HOME')
        path          = cp.confpars.confParsDirName + '/' + cp.confpars.confParsFileName
        self.fileEdit = QtGui.QLineEdit(path)
        
        self.showToolTips()

        hboxT1 = QtGui.QHBoxLayout()
        hboxT1.addWidget(self.titFile)

        hboxF = QtGui.QHBoxLayout()
        hboxF.addWidget(self.fileEdit)
        hboxF.addWidget(self.butBrowse)

        grid = QtGui.QGridLayout()
        grid.addWidget(self.titPars,       0, 0, 1, 3)
        grid.addWidget(self.butRead,       1, 0)
        grid.addWidget(self.butWrite,      1, 1)
        grid.addWidget(self.butDefault,    1, 2)
        grid.addWidget(self.butPrint,      1, 3)
        #grid.addWidget(self.titRadio,      2, 0)
        #grid.addWidget(self.radioRead,     2, 1, 1, 2)
        #grid.addWidget(self.radioDefault,  3, 1, 1, 2)
        grid.addWidget(self.butExit,       4, 3)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hboxT1)
        vbox.addLayout(hboxF)
        vbox.addStretch(1)     
        vbox.addLayout(grid)

        self.setLayout(vbox)

        self.connect(self.butExit,      QtCore.SIGNAL('clicked()'),          self.processExit         )
        self.connect(self.butRead,      QtCore.SIGNAL('clicked()'),          self.processRead         )
        self.connect(self.butWrite,     QtCore.SIGNAL('clicked()'),          self.processWrite        )
        self.connect(self.butPrint,     QtCore.SIGNAL('clicked()'),          self.processPrint        )
        self.connect(self.butDefault,   QtCore.SIGNAL('clicked()'),          self.processDefault      )
        self.connect(self.butBrowse,    QtCore.SIGNAL('clicked()'),          self.processBrowse       )
        self.connect(self.radioRead,    QtCore.SIGNAL('clicked()'),          self.processRadioRead    )
        self.connect(self.radioDefault, QtCore.SIGNAL('clicked()'),          self.processRadioDefault )
        self.connect(self.fileEdit,     QtCore.SIGNAL('editingFinished ()'), self.processFileEdit     )

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        self.fileEdit  .setToolTip('Type the file path name here,\nor better use "Browse" button.')
        self.butBrowse .setToolTip('Select the file path name\nto read/write the configuration parameters.')
        self.butRead   .setToolTip('Read (retreave) the configuration parameters from file.')
        self.butWrite  .setToolTip('Write (save) the configuration parameters in file.')
        self.butDefault.setToolTip('Reset the configuration parameters\nto their default values.')
        self.butPrint  .setToolTip('Print current values of the configuration parameters.')
        self.butExit   .setToolTip('Quit this GUI and close the window.')
        self.radioRead .setToolTip('If this button is ON and configuration parameters are saved,\n' +
                                   'then they will be read from file at the next start of this program.')
        self.radioDefault.setToolTip('If this button is ON and configuration parameters are saved,\n' +
                                     'then they will be reset to the default values at the next start of this program.')

    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        print 'closeEvent'
        self.processExit()
            
    def processExit(self):
        print 'Exit'
        self.close()
        cp.confpars.configGUIIsOpen = False

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def processRead(self):
        print 'Read'
        cp.confpars.readParameters(self.confParsFileName())
        self.parent.fileEdit.setText(cp.confpars.dirName + '/' + cp.confpars.fileName)
        
    def processWrite(self):
        print 'Write'
        cp.confpars.writeParameters(self.confParsFileName())

    def processDefault(self):
        print 'Set default values of configuration parameters'
        cp.confpars.setDefaultParameters()
        self.parent.fileEdit.setText(cp.confpars.dirName + '/' + cp.confpars.fileName)

    def processPrint(self):
        print 'Print'
        cp.confpars.Print()

    def processRadioRead(self):
        print 'RadioRead'
        cp.confpars.readParsFromFileAtStart = True

    def processRadioDefault(self):
        print 'RadioDefault'
        cp.confpars.readParsFromFileAtStart = False

    def processBrowse(self):
        print 'Browse'
        self.path = str(self.fileEdit.displayText())
        self.dirName,self.fileName = os.path.split(self.path)
        print 'dirName  : %s' % (self.dirName)
        print 'fileName : %s' % (self.fileName)
        self.path = QtGui.QFileDialog.getOpenFileName(self,'Open file',self.dirName)
        self.dirName,self.fileName = os.path.split(str(self.path))
        #self.path = cp.confpars.confParsDirName + '/' + cp.confpars.confParsFileName
        #self.path = self.dirName+'/'+self.fileName
        if self.dirName == '' or self.fileName == '' :
            print 'Input dirName or fileName is empty... use default values'  
        else :
            self.fileEdit.setText(self.path)
            cp.confpars.confParsDirName  = self.dirName
            cp.confpars.confParsFileName = self.fileName

    def processFileEdit(self):
        print 'FileEdit'
        self.path = str(self.fileEdit.displayText())
        cp.confpars.confParsDirName,cp.confpars.confParsFileName = os.path.split(self.path)
        print 'Set dirName  : %s' % (cp.confpars.confParsDirName)
        print 'Set fileName : %s' % (cp.confpars.confParsFileName)
 
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
