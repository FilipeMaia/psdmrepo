
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISelectQuadAndPair...
#
#------------------------------------------------------------------------

"""GUI selects the CSpad Quad and Pair.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.
@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $
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
import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp
import ConfigCSpad      as cs

#---------------------
#  Class definition --
#---------------------
class GUISelectQuadAndPair ( QtGui.QWidget ) :
    """GUI selects the CSpad Quad and Pair"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, parent=None, app=None) :
        """Constructor"""

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 500, 80, 25)
        self.setWindowTitle('Quad and pair selection')
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        # see http://www.riverbankcomputing.co.uk/static/Docs/PyQt4/html/qframe.html
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(True)

        self.char_expand = u'\u25BE' # down-head triangle

        self.titQuad  = QtGui.QLabel('Quad:')
        self.titPair  = QtGui.QLabel('Pair:')

#        self.editQuad = QtGui.QLineEdit(str(cp.confpars.cspadQuad))
#        self.editPair = QtGui.QLineEdit(str(cp.confpars.cspadPair))

        self.butMenuQuad = QtGui.QPushButton(str(cp.confpars.cspadQuad) + self.char_expand)
        self.butMenuPair = QtGui.QPushButton(str(cp.confpars.cspadPair) + self.char_expand)
        self.butMenuQuad.setMaximumWidth(25)
        self.butMenuPair.setMaximumWidth(25)

        self.listActMenuQuad = []
        self.popupMenuQuad = QtGui.QMenu()
        for quad in range(4) :
            self.listActMenuQuad.append(self.popupMenuQuad.addAction(str(quad)))

        self.setMenuForPairs()

        self.popupMenuPair = QtGui.QMenu()
        for pair in range(8) : # cs.confcspad.indPairsInQuads[cp.confpars.cspadQuad]
            self.listActMenuPair.append(self.popupMenuPair.addAction(str(pair)))

        #self.connect(self.butMenu1, QtCore.SIGNAL('clicked()'), self.processMenu )
        
        hbox = QtGui.QHBoxLayout() 
        hbox.addWidget(self.titQuad)
       #hbox.addWidget(self.editQuad)
        hbox.addWidget(self.butMenuQuad)
        hbox.addStretch(1)     
        hbox.addWidget(self.titPair)
       #hbox.addWidget(self.editPair)
        hbox.addWidget(self.butMenuPair)

        self.setLayout(hbox)

        self.connect(self.butMenuQuad,   QtCore.SIGNAL('clicked()'), self.processMenuQuad )
        self.connect(self.butMenuPair,   QtCore.SIGNAL('clicked()'), self.processMenuPair )

#        self.connect(self.editQuad,  QtCore.SIGNAL('editingFinished ()'), self.processEditQuad )
#        self.connect(self.editPair,  QtCore.SIGNAL('editingFinished ()'), self.processEditPair )

        self.showToolTips()


    #-------------------
    # Private methods --
    #-------------------

    def setMenuForPairs(self) :

        self.listActMenuPair = []
        self.popupMenuPair = QtGui.QMenu()
        for pair in range(8) : 
            if cs.confcspad.indPairsInQuads[cp.confpars.cspadQuad][pair] != -1:
                self.listActMenuPair.append(self.popupMenuPair.addAction(str(pair)))
            else :
                self.listActMenuPair.append(self.popupMenuPair.addAction('N/A'))

        if cs.confcspad.indPairsInQuads[cp.confpars.cspadQuad][cp.confpars.cspadPair] == -1:
            self.butMenuPair.setText( 'N/A' + self.char_expand )
            self.butMenuPair.setStyleSheet("background-color: rgb(255, 0, 0); color: rgb(255, 255, 255)")
            self.butMenuPair.setMaximumWidth(45)
        else :
            self.butMenuPair.setText( str(cp.confpars.cspadPair) + self.char_expand )
            self.butMenuPair.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0)")
            self.butMenuPair.setMaximumWidth(25)



    def showToolTips(self):
        # Tips for buttons and fields:
        #self            .setToolTip('Click on QUAD or PAIR number using mouse left button')
        self.butMenuQuad.setToolTip('Click mouse left on this button\nand select the QUAD number\nif it is necessary for the plot.')
        self.butMenuPair.setToolTip('Select the PAIR number.\nN/A means that the pair is not available in the dataset.')



    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def processQuit(self):
        print 'Quit'
        self.close()
        
    def mousePressEvent(self, event):
        print 'Click on Quad or Pair number using mouse left button\n'
        #print 'event.button() = %s at position' % (event.button()),        
        #print (event.pos()),
        #print ' x=%d, y=%d' % (event.x(),event.y()),        
        #print ' global x=%d, y=%d' % (event.globalX(),event.globalY())

    def processMenuQuad(self):
        #print 'MenuQuad'
        actionSelected = self.popupMenuQuad.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        for action in self.listActMenuQuad :
            cp.confpars.cspadQuad = int(actionSelected.text())
            if actionSelected == action : self.butMenuQuad.setText( str(cp.confpars.cspadQuad) + self.char_expand )

        self.setMenuForPairs()
        

    def processMenuPair(self):
        #print 'MenuPair'
        actionSelected = self.popupMenuPair.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        for action in self.listActMenuPair:
            if actionSelected != action : continue
            if actionSelected.text() == 'N/A' :
                self.butMenuPair.setText( 'N/A' + self.char_expand )
                self.butMenuPair.setStyleSheet("background-color: rgb(255, 0, 0); color: rgb(255, 255, 255)")
                self.butMenuPair.setMaximumWidth(45)
            else :
                cp.confpars.cspadPair = int(actionSelected.text())
                self.butMenuPair.setText( str(cp.confpars.cspadPair) + self.char_expand )
                self.butMenuPair.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0)")
                self.butMenuPair.setMaximumWidth(25)


#    def processEditQuad(self):    
#        cp.confpars.cspadQuad = int(self.editQuad.displayText())
#        print 'Set quad: ', cp.confpars.cspadQuad

#    def processEditPair(self):    
#        cp.confpars.cspadPair = int(self.editPair.displayText())
#        print 'Set pair: ', cp.confpars.cspadPair

#http://doc.qt.nokia.com/4.6/qt.html#Key-enum
    def keyPressEvent(self, event):
        print 'event.key() = %s' % (event.key())
        if event.key() == QtCore.Qt.Key_Escape:
    #        self.close()
            self.SHowIsOn = False    

        if event.key() == QtCore.Qt.Key_B:
            print 'event.key() = %s' % (QtCore.Qt.Key_B)

        if event.key() == QtCore.Qt.Key_Return:
            print 'event.key() = Return'

            #self.processFileEdit()
            #self.processNumbEdit()
            #self.processSpanEdit()
            #self.currentEventNo()

        if event.key() == QtCore.Qt.Key_Home:
            print 'event.key() = Home'

    def closeEvent(self, event):
        print 'closeEvent'
        self.processQuit()

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUISelectQuadAndPair()
    ex.show()
    app.exec_()
#-----------------------------
