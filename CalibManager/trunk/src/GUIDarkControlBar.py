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
from GUILogger              import *
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

        self.dark_list_run_min   = cp.dark_list_run_min 
        self.dark_list_run_max   = cp.dark_list_run_max 

        self.setGeometry(100, 50, 390, 30)
        self.setWindowTitle('Instrument Experiment Run')
        self.setFrame()
 

        self.labRuns = QtGui.QLabel('Show runs')
        self.labDets = QtGui.QLabel('  for detectors:')
        self.labTo   = QtGui.QLabel(':')

        self.ediFrom = QtGui.QLineEdit  ( str(self.dark_list_run_min.value()) )
        self.ediTo   = QtGui.QLineEdit  ( str(self.dark_list_run_max.value()) )

        self.ediFrom.setValidator(QtGui.QIntValidator(0,9999,self))
        self.ediTo  .setValidator(QtGui.QIntValidator(0,9999,self))

        self.butRuns = QtGui.QPushButton( self.dark_list_show_runs.value() + self.char_expand )
        self.butRuns.setFixedWidth(90)
        self.butDets = QtGui.QPushButton( self.dark_list_show_dets.value() + self.char_expand )
        self.butDets.setFixedWidth(110)
        #self.butRuns.setMaximumWidth(90)

        self.cbx_deploy_hotpix = QtGui.QCheckBox('Deploy hotpix mask')
        self.cbx_deploy_hotpix.setChecked( cp.dark_deploy_hotpix.value() )
        
        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addWidget(self.labRuns)
        self.hbox.addWidget(self.butRuns)
        self.hbox.addWidget(self.ediFrom)
        self.hbox.addWidget(self.labTo)
        self.hbox.addWidget(self.ediTo)
        self.hbox.addWidget(self.labDets)
        self.hbox.addWidget(self.butDets)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.cbx_deploy_hotpix)
    
        self.setLayout(self.hbox)

        #self.connect( self.ediExp,     QtCore.SIGNAL('editingFinished ()'), self.processEdiExp )
        self.connect( self.butRuns,     QtCore.SIGNAL('clicked()'),          self.onButRuns  )
        self.connect( self.butDets,     QtCore.SIGNAL('clicked()'),          self.onButDets  )
        self.connect( self.ediFrom,     QtCore.SIGNAL('editingFinished()'),  self.onEdiFrom )
        self.connect( self.ediTo,       QtCore.SIGNAL('editingFinished()'),  self.onEdiTo )
        self.connect( self.cbx_deploy_hotpix, QtCore.SIGNAL('stateChanged(int)'), self.on_cbx ) 

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
        self.ediFrom.setToolTip('Fir run in the list')
        self.ediTo.  setToolTip('Last run in the list')


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
        self.labTo  .setStyleSheet (cp.styleLabel)
        self.setStyleButtons()
        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))
        self.setMinimumSize(390,28)
        self.ediFrom.setFixedWidth(40)
        self.ediTo.  setFixedWidth(40)
 

    def setStyleButtons(self):
        self.butRuns.setStyleSheet(cp.styleButton)
        self.butDets.setStyleSheet(cp.styleButton)
        self.setButtonsVisible()


    def setButtonsVisible(self):
        show_run_range = self.dark_list_show_runs.value() == 'in range' # self.list_of_show_runs[0]
        self.ediFrom.setVisible(show_run_range)
        self.ediTo.  setVisible(show_run_range)
        self.labTo.  setVisible(show_run_range)

 
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
        #print 'self.rect():', str(self.rect())


    def moveEvent(self, e):
        #print 'moveEvent' 
        pass
#        cp.posGUIMain = (self.pos().x(),self.pos().y())


    def onEdiFrom(self):
        #logger.info('onEdiFrom', __name__ )
        str_run_min = str( self.ediFrom.displayText() )        
        msg = 'Set the run range from %s' % str_run_min
        logger.info(msg, __name__ )
        self.dark_list_run_min.setValue(int(str_run_min))


    def onEdiTo(self):
        #logger.info('onEdiFrom', __name__ )
        str_run_max = str( self.ediTo.displayText() )        
        msg = 'Set the run range to %s' % str_run_max
        logger.info(msg, __name__ )
        self.dark_list_run_max.setValue(int(str_run_max))


    def onButRuns(self):
        #print 'onButRuns'
        item_selected = gu.selectFromListInPopupMenu(self.list_of_show_runs)
        if item_selected is None : return # selection is cancelled
        self.setRuns(item_selected)
        #self.setStyleButtons()
        self.setButtonsVisible()


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


    def on_cbx(self):
        #if self.cbx.hasFocus() :
        par = cp.dark_deploy_hotpix
        cbx = self.cbx_deploy_hotpix

        par.setValue( cbx.isChecked() )
        msg = 'check box ' + cbx.text()  + ' is set to: ' + str( par.value())
        logger.info(msg, __name__ )


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIDarkControlBar ()
    widget.show()
    app.exec_()

#-----------------------------
