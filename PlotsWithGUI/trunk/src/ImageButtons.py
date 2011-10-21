#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImageButtons...
#
#------------------------------------------------------------------------

"""Plots for any 'image' record in the EventeDisplay project.

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

#class ImageButtons (QtGui.QMainWindow) :
class ImageButtons (QtGui.QWidget) :
    """A set of buttons for figure control."""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None ):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('GUI of buttons')

        self.setFrame()

        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"

        self.myXmin = None
        self.myXmax = None
        self.myYmin = None
        self.myYmax = None
        self.myZmin = None
        self.myZmax = None
        self.myNBins = 100
        self.myZoomIsOn = False

        self.but_draw  = QtGui.QPushButton("&Draw")
        self.but_quit  = QtGui.QPushButton("&Quit")
        self.cbox_grid = QtGui.QCheckBox("Show &Grid")
        self.cbox_grid.setChecked(False)
        self.cbox_log  = QtGui.QCheckBox("&Log")
        self.cbox_log.setChecked(False)

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
 
        #self.connect(self.but_draw,  QtCore.SIGNAL('clicked()'),         self.processDraw)
        self.connect(self.but_quit,  QtCore.SIGNAL('clicked()'),         self.processQuit)
        #self.connect(self.cbox_grid, QtCore.SIGNAL('stateChanged(int)'), self.processDraw)
        #self.connect(self.cbox_log,  QtCore.SIGNAL('stateChanged(int)'), self.processDraw)

        self.connect(self.cboxXIsOn, QtCore.SIGNAL('stateChanged(int)'), self.processCBoxes)
        self.connect(self.cboxYIsOn, QtCore.SIGNAL('stateChanged(int)'), self.processCBoxes)
        self.connect(self.cboxZIsOn, QtCore.SIGNAL('stateChanged(int)'), self.processCBoxes)

        self.connect(self.editXmin, QtCore.SIGNAL('editingFinished ()'), self.processEditXmin)
        self.connect(self.editXmax, QtCore.SIGNAL('editingFinished ()'), self.processEditXmax)
        self.connect(self.editYmin, QtCore.SIGNAL('editingFinished ()'), self.processEditYmin)
        self.connect(self.editYmax, QtCore.SIGNAL('editingFinished ()'), self.processEditYmax)
        self.connect(self.editZmin, QtCore.SIGNAL('editingFinished ()'), self.processEditZmin)
        self.connect(self.editZmax, QtCore.SIGNAL('editingFinished ()'), self.processEditZmax)

        # Layout with box sizers
        # 
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.but_draw)
        hbox.addWidget(self.cbox_grid)
        hbox.addStretch(1)
        hbox.addWidget(self.but_quit)
        #hbox.setAlignment(w, QtCore.Qt.AlignVCenter)

        hboxX = QtGui.QHBoxLayout()
        hboxX.addWidget(self.cboxXIsOn)
        hboxX.addWidget(self.editXmin)
        hboxX.addWidget(self.editXmax)
        hboxX.addStretch(1)

        hboxY = QtGui.QHBoxLayout()
        hboxY.addWidget(self.cboxYIsOn)
        hboxY.addWidget(self.editYmin)
        hboxY.addWidget(self.editYmax)
        hboxY.addStretch(1)

        hboxZ = QtGui.QHBoxLayout()
        hboxZ.addWidget(self.cboxZIsOn)
        hboxZ.addWidget(self.editZmin)
        hboxZ.addWidget(self.editZmax)
        hboxZ.addWidget(self.cbox_log)
        hboxZ.addStretch(1)

        vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        vbox.addLayout(hboxX)              # <=== Add buttons etc.
        vbox.addLayout(hboxY)
        vbox.addLayout(hboxZ)
        vbox.addLayout(hbox)

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

    w = ImageButtons(None)
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
