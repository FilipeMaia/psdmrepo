#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIConfigParameters...
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

from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------
class GUIConfigParameters ( QtGui.QWidget ) :
    """GUI works with configuration parameters management"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)

        #self.parent = cp.guimain

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Configuration Parameters')
        self.setFrame()
        
        self.titFile     = QtGui.QLabel('File with configuration parameters:')
        self.titPars     = QtGui.QLabel('Operations on configuration parameters:')
        self.butBrowse   = QtGui.QPushButton("Browse")
        self.butRead     = QtGui.QPushButton("Read")
        self.butWrite    = QtGui.QPushButton("Save")
        self.butDefault  = QtGui.QPushButton("Reset default")
        self.butPrint    = QtGui.QPushButton("Print current")
        self.butClose    = QtGui.QPushButton("Close")
        self.fnameEdit   = QtGui.QLineEdit( cp.fname_cp.value() )        

        grid = QtGui.QGridLayout()
        grid.addWidget(self.titFile,       0, 0, 1, 4)
        grid.addWidget(self.fnameEdit,     1, 0, 1, 4)
        grid.addWidget(self.butBrowse,     1, 4)
        grid.addWidget(self.titPars,       2, 0, 1, 4)
        grid.addWidget(self.butRead,       3, 0)
        grid.addWidget(self.butWrite,      3, 1)
        grid.addWidget(self.butDefault,    3, 2)
        grid.addWidget(self.butPrint,      3, 3)
        grid.addWidget(self.butClose,      3, 4)
        self.setLayout(grid)

        self.connect(self.fnameEdit,    QtCore.SIGNAL('editingFinished ()'), self.processFileEdit     )
        self.connect(self.butRead,      QtCore.SIGNAL('clicked()'),          self.processRead         )
        self.connect(self.butWrite,     QtCore.SIGNAL('clicked()'),          self.processWrite        )
        self.connect(self.butPrint,     QtCore.SIGNAL('clicked()'),          self.processPrint        )
        self.connect(self.butDefault,   QtCore.SIGNAL('clicked()'),          self.processDefault      )
        self.connect(self.butBrowse,    QtCore.SIGNAL('clicked()'),          self.processBrowse       )
        self.connect(self.butClose,     QtCore.SIGNAL('clicked()'),          self.processClose        )

        self.showToolTips()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        self.fnameEdit .setToolTip('Type the file path name here,\nor better use "Browse" button.')
        self.butBrowse .setToolTip('Select the file path name\nto read/write the configuration parameters.')
        self.butRead   .setToolTip('Read the configuration parameters from file.')
        self.butWrite  .setToolTip('Save (write) the configuration parameters in file.')
        self.butDefault.setToolTip('Reset the configuration parameters\nto their default values.')
        self.butPrint  .setToolTip('Print current values of the configuration parameters.')
        self.butClose  .setToolTip('Close this window.')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp
            del cp.guiconfigparameters 
        except : # AttributeError:
            pass

    def processClose(self):
        #print 'Close button'
        self.close()

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def processRead(self):
        print 'Read'
        cp.readParametersFromFile( self.getFileNameFromEditField() )
        self.fnameEdit.setText( cp.fname_cp.value() )
        #self.parent.fnameEdit.setText( cp.fname_cp.value() )
        #self.refreshGUIWhatToDisplay()

    def processWrite(self):
        print 'Write'
        cp.saveParametersInFile( self.getFileNameFromEditField() )

    def processDefault(self):
        print 'Set default values of configuration parameters.'
        cp.setDefaultValues()
        self.fnameEdit.setText( cp.fname_cp.value() )
        #self.refreshGUIWhatToDisplay()

    def processPrint(self):
        print 'Print'
        cp.printParameters()

    def processBrowse(self):
        print 'Browse'
        self.path = self.getFileNameFromEditField()
        self.dname,self.fname = os.path.split(self.path)
        print 'dname : %s' % (self.dname)
        print 'fname : %s' % (self.fname)
        self.path = str( QtGui.QFileDialog.getOpenFileName(self,'Open file',self.dname) )
        self.dname,self.fname = os.path.split(self.path)

        if self.dname == '' or self.fname == '' :
            print 'Input directiry name or file name is empty... use default values'  
        else :
            self.fnameEdit.setText(self.path)
            cp.fname_cp.setValue(self.path)

    def processFileEdit(self):
        print 'FileEdit'
        self.path = self.getFileNameFromEditField()
        cp.fname_cp.setValue(self.path)
        dname,fname = os.path.split(self.path)
        print 'Set dname : %s' % (dname)
        print 'Set fname : %s' % (fname)

    def getFileNameFromEditField(self):
        return str( self.fnameEdit.displayText() )

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIConfigParameters ()
    widget.show()
    app.exec_()

#-----------------------------
