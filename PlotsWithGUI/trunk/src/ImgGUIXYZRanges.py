#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgGUIXYZRanges...
#
#------------------------------------------------------------------------

"""GUI for XYZ Ranges

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule
@version $Id: 
@author Mikhail S. Dubrovin
"""

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

#---------------------
#  Class definition --
#---------------------

class ImgGUIXYZRanges (QtGui.QWidget) :
    """GUI for XYZ Ranges"""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None ):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('GUI for XYZ Ranges')
        #self.setFrame()

        self.styleSheetGrey  = "background-color: rgb(170, 170, 170); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(240, 240, 240); color: rgb(0, 0, 0)"

        self.myXmin = None
        self.myXmax = None
        self.myYmin = None
        self.myYmax = None
        self.myZmin = None
        self.myZmax = None

        self.cboxXIsOn = QtGui.QCheckBox("X min, max:")
        self.cboxYIsOn = QtGui.QCheckBox("Y min, max:")
        self.cboxZIsOn = QtGui.QCheckBox("Z min, max:")

        self.editXmin  = QtGui.QLineEdit(self.stringOrNone(self.myXmin))
        self.editXmax  = QtGui.QLineEdit(self.stringOrNone(self.myXmax))
        self.editYmin  = QtGui.QLineEdit(self.stringOrNone(self.myYmin))
        self.editYmax  = QtGui.QLineEdit(self.stringOrNone(self.myYmax))
        self.editZmin  = QtGui.QLineEdit(self.stringOrNone(self.myZmin))
        self.editZmax  = QtGui.QLineEdit(self.stringOrNone(self.myZmax))

        width = 60
        self.editXmin.setMaximumWidth(width)
        self.editXmax.setMaximumWidth(width)
        self.editYmin.setMaximumWidth(width)
        self.editYmax.setMaximumWidth(width)
        self.editZmin.setMaximumWidth(width)
        self.editZmax.setMaximumWidth(width)

        self.editXmin.setValidator(QtGui.QIntValidator(0,100000,self))
        self.editXmax.setValidator(QtGui.QIntValidator(0,100000,self)) 
        self.editYmin.setValidator(QtGui.QIntValidator(0,100000,self))
        self.editYmax.setValidator(QtGui.QIntValidator(0,100000,self)) 
        self.editZmin.setValidator(QtGui.QIntValidator(-100000,100000,self))
        self.editZmax.setValidator(QtGui.QIntValidator(-100000,100000,self))
 
        self.connect(self.cboxXIsOn, QtCore.SIGNAL('stateChanged(int)'), self.processCBoxes)
        self.connect(self.cboxYIsOn, QtCore.SIGNAL('stateChanged(int)'), self.processCBoxes)
        self.connect(self.cboxZIsOn, QtCore.SIGNAL('stateChanged(int)'), self.processCBoxes)
        self.connect(self.editXmin,  QtCore.SIGNAL('editingFinished()'), self.processEditXmin)
        self.connect(self.editXmax,  QtCore.SIGNAL('editingFinished()'), self.processEditXmax)
        self.connect(self.editYmin,  QtCore.SIGNAL('editingFinished()'), self.processEditYmin)
        self.connect(self.editYmax,  QtCore.SIGNAL('editingFinished()'), self.processEditYmax)
        self.connect(self.editZmin,  QtCore.SIGNAL('editingFinished()'), self.processEditZmin)
        self.connect(self.editZmax,  QtCore.SIGNAL('editingFinished()'), self.processEditZmax)

        # GUI Layout
        grid = QtGui.QGridLayout()

        row = 0
        grid.addWidget(self.cboxXIsOn, row, 0)
        grid.addWidget(self.editXmin , row, 1)
        grid.addWidget(self.editXmax , row, 2)

        row = 1
        grid.addWidget(self.cboxYIsOn, row, 0)
        grid.addWidget(self.editYmin , row, 1)
        grid.addWidget(self.editYmax , row, 2)

        row = 2
        grid.addWidget(self.cboxZIsOn, row, 0)
        grid.addWidget(self.editZmin , row, 1)
        grid.addWidget(self.editZmax , row, 2)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(grid)
        self.setLayout(vbox)

        self.setEditFieldColors()    
        self.setEditFieldValues()


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def setXYZRanges(self, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax) :
        self.myXmin = Xmin
        self.myXmax = Xmax
        self.myYmin = Ymin
        self.myYmax = Ymax
        self.myZmin = Zmin
        self.myZmax = Zmax
        self.setEditFieldValues()


    def getXYZRanges(self) :
        return ( self.myXmin, self.myXmax, self.myYmin, self.myYmax, self.myZmin, self.myZmax )

     
    def setEditFieldValues(self) :
        self.editXmin.setText( str(self.intOrNone(self.myXmin)) )
        self.editXmax.setText( str(self.intOrNone(self.myXmax)) )

        self.editYmin.setText( str(self.intOrNone(self.myYmin)) )
        self.editYmax.setText( str(self.intOrNone(self.myYmax)) ) 

        self.editZmin.setText( str(self.intOrNone(self.myZmin)) )
        self.editZmax.setText( str(self.intOrNone(self.myZmax)) )

        self.setEditFieldColors()

       
    def setEditFieldColors(self) :
        
        if self.cboxXIsOn.isChecked(): self.styleSheet = self.styleSheetWhite
        else                         : self.styleSheet = self.styleSheetGrey
        self.editXmin.setStyleSheet('Text-align:left;' + self.styleSheet)
        self.editXmax.setStyleSheet('Text-align:left;' + self.styleSheet)

        if self.cboxYIsOn.isChecked(): self.styleSheet = self.styleSheetWhite
        else                         : self.styleSheet = self.styleSheetGrey
        self.editYmin.setStyleSheet('Text-align:left;' + self.styleSheet)
        self.editYmax.setStyleSheet('Text-align:left;' + self.styleSheet)

        if self.cboxZIsOn.isChecked(): self.styleSheet = self.styleSheetWhite
        else                         : self.styleSheet = self.styleSheetGrey
        self.editZmin.setStyleSheet('Text-align:left;' + self.styleSheet)
        self.editZmax.setStyleSheet('Text-align:left;' + self.styleSheet)

        self.editXmin.setReadOnly( not self.cboxXIsOn.isChecked() )
        self.editXmax.setReadOnly( not self.cboxXIsOn.isChecked() )

        self.editYmin.setReadOnly( not self.cboxYIsOn.isChecked() )
        self.editYmax.setReadOnly( not self.cboxYIsOn.isChecked() )

        self.editZmin.setReadOnly( not self.cboxZIsOn.isChecked() )
        self.editZmax.setReadOnly( not self.cboxZIsOn.isChecked() )


    def processCBoxes(self):
        self.setEditFieldColors()


    def stringOrNone(self,value):
        if value == None : return 'None'
        else             : return str(value)


    def intOrNone(self,value):
        if value == None : return None
        else             : return int(value)


    def processEditXmin(self):
        self.myXmin = self.editXmin.displayText()


    def processEditXmax(self):
        self.myXmax = self.editXmax.displayText()


    def processEditYmin(self):
        self.myYmin = self.editYmin.displayText()


    def processEditYmax(self):
        self.myYmax = self.editYmax.displayText()


    def processEditZmin(self):
        self.myZmin = self.editZmin.displayText()


    def processEditZmax(self):
        self.myZmax = self.editZmax.displayText()
 

    def processQuit(self):
        print 'Quit'
        self.close()


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        print 'Close application'
           
#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    w = ImgGUIXYZRanges(None)
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
