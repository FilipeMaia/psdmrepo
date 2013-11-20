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
        self.dark_list_show_runs = cp.dark_list_show_runs
        self.list_of_show_runs   = cp.list_of_show_runs

        self.dark_list_show_dets = cp.dark_list_show_dets
        self.list_of_show_dets   = cp.list_of_show_dets

        self.setGeometry(100, 50, 600, 30)
        self.setWindowTitle('Instrument Experiment Run')
        self.setFrame()
 

        self.labRuns = QtGui.QLabel('Show runs:')
        self.labDets   = QtGui.QLabel('for detectors:')

        self.butRuns = QtGui.QPushButton( self.dark_list_show_runs.value() + self.char_expand )
        self.butRuns.setFixedWidth(90)
        self.butDets = QtGui.QPushButton( self.dark_list_show_dets.value() + self.char_expand )
        self.butDets.setFixedWidth(110)
        #self.butRuns.setMaximumWidth(90)

        
        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addWidget(self.labRuns)
        self.hbox.addWidget(self.butRuns)
        self.hbox.addWidget(self.labDets)
        self.hbox.addWidget(self.butDets)
        self.hbox.addStretch(1)
    
        self.setLayout(self.hbox)

        #self.connect( self.ediExp,     QtCore.SIGNAL('editingFinished ()'), self.processEdiExp )
        self.connect( self.butRuns,     QtCore.SIGNAL('clicked()'),          self.onButRuns  )
        self.connect( self.butDets,     QtCore.SIGNAL('clicked()'),          self.onButDets  )

        self.showToolTips()
        self.setStyle()

        cp.guidarkcontrolbar = self

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        self.butRuns  .setToolTip('Select the type of runs to list')
        self.butDets  .setToolTip('Select the type of dets to list')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setStyle(self):
        #self.setStyleSheet(cp.styleYellow)
        self.labRuns.setStyleSheet (cp.styleLabel)
        self.labDets.setStyleSheet (cp.styleLabel)
        self.setStyleButtons()
        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))


    def setStyleButtons(self):
        self.butRuns.setStyleSheet(cp.styleButton)
        self.butDets.setStyleSheet(cp.styleButton)

 
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


    def onButRuns(self):
        #print 'onButRuns'
        item_selected = gu.selectFromListInPopupMenu(self.list_of_show_runs)
        if item_selected is None : return # selection is cancelled
        self.setRuns(item_selected)
        #self.setStyleButtons()


    def onButDets(self):
        #print 'onButDets'
        item_selected = gu.selectFromListInPopupMenu(self.list_of_show_dets)
        if item_selected is None : return # selection is cancelled
        self.setDets(item_selected)
        #self.setStyleButtons()


    def setRuns(self, txt='None'):
        #print 'setType', txt
        self.dark_list_show_runs.setValue(txt)
        self.butRuns.setText(txt + self.char_expand)        
        #if txt == 'None' : self.list_of_run = None
        #self.setFile()
        if cp.guidarklist != None :
            cp.guidarklist.updateList()


    def setDets(self, txt='None'):
        #print 'setType', txt
        self.dark_list_show_dets.setValue(txt)
        self.butDets.setText(txt + self.char_expand)        
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
