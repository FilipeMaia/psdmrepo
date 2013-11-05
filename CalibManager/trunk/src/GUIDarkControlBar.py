#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDarkControlBar...
#
#------------------------------------------------------------------------

"""GUI sets the instrument, experiment, and run number for signal and dark data"""

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

from ConfigParametersForApp import cp
import GlobalUtils          as     gu
#from GUILogger              import *
#from FileNameManager        import fnm

#---------------------
#  Class definition --
#---------------------
class GUIDarkControlBar ( QtGui.QWidget ) :
    """GUI sets the configuration parameters for instrument, experiment, and run number"""

    char_expand    = u' \u25BE' # down-head triangle

    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.parent = parent
        self.dark_list_show_type = cp.dark_list_show_type
        self.list_of_show_dark   = cp.list_of_show_dark

        self.setGeometry(100, 50, 600, 30)
        self.setWindowTitle('Instrument Experiment Run')
        self.setFrame()
 

        self.labSelect = QtGui.QLabel('Show runs:')

        self.butSelect = QtGui.QPushButton( self.dark_list_show_type.value() + self.char_expand )
        self.butSelect.setMaximumWidth(90)
        
        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addWidget(self.labSelect)
        self.hbox.addWidget(self.butSelect)
        self.hbox.addStretch(1)
    
        self.setLayout(self.hbox)

        #self.connect( self.ediExp,     QtCore.SIGNAL('editingFinished ()'), self.processEdiExp )
        self.connect( self.butSelect,     QtCore.SIGNAL('clicked()'),          self.onButSelect  )

        self.showToolTips()
        self.setStyle()

        cp.guidarkcontrolbar = self

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        self.butSelect  .setToolTip('Select the type of files to list from the pop-up menu.')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setStyle(self):
        #self.setStyleSheet(cp.styleYellow)
        self.labSelect  .setStyleSheet (cp.styleLabel)
        self.setStyleButtons()
        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))
         

    def setStyleButtons(self):
        self.butSelect.setStyleSheet(cp.styleButton)

 
    def setParent(self,parent) :
        self.parent = parent


    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp
            del cp.guidarkcontrolbar# GUIDarkControlBar
            cp.guidarkcontrolbar = None
        except AttributeError:
            pass # silently ignore


    def processClose(self):
        #print 'Close button'
        self.close()


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def moveEvent(self, e):
        #print 'moveEvent' 
        pass
#        cp.posGUIMain = (self.pos().x(),self.pos().y())


    def onButSelect(self):
        #print 'onButSelect'

        item_selected = gu.selectFromListInPopupMenu(self.list_of_show_dark)
        if item_selected is None : return            # selection is cancelled

        type = item_selected

        self.setType(type)
        #self.setStyleButtons()


    def setType(self, txt='None'):
        #print 'setType', txt
        self.dark_list_show_type.setValue(txt)
        self.butSelect.setText(txt + self.char_expand)        
        #if txt == 'None' : self.list_of_run = None
        #self.setFile()
        if cp.guidarklist != None :
            cp.guidarklist.updateList()


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIDarkControlBar ()
    widget.show()
    app.exec_()

#-----------------------------
