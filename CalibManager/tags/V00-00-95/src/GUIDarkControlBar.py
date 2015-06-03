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
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
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
from GUIRange               import *
#from FileNameManager        import fnm

#---------------------
#  Class definition --
#---------------------
class GUIDarkControlBar ( QtGui.QWidget ) :
    """GUI sets the configuration parameters for instrument, experiment, and run number"""

    char_expand    = cp.char_expand
    #char_expand    = u' \u25BC' # down-head triangle
    #char_expand    = '' # down-head triangle

    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.parent = parent
        self.dark_list_show_runs = cp.dark_list_show_runs
        self.list_of_show_runs   = cp.list_of_show_runs

        self.dark_list_show_dets = cp.dark_list_show_dets
        self.list_of_show_dets   = cp.list_of_show_dets

        self.dark_list_run_min   = cp.dark_list_run_min 
        self.dark_list_run_max   = cp.dark_list_run_max 

        self.setGeometry(100, 50, 590, 36)
        self.setWindowTitle('Dark Control Bar')
        self.setFrame()
 

        self.labRuns = QtGui.QLabel('Show runs')
        self.labDets = QtGui.QLabel('  for detectors:')

        self.guirange  = GUIRange(None,\
                                  str(self.dark_list_run_min.value()),\
                                  str(self.dark_list_run_max.value()),\
                                  txt_from='', txt_to=':')
 
        self.butRuns = QtGui.QPushButton( self.dark_list_show_runs.value() + self.char_expand )
        self.butRuns.setFixedWidth(90)
        self.butDets = QtGui.QPushButton( self.dark_list_show_dets.value() + self.char_expand )
        self.butDets.setFixedWidth(110)
        self.butUpdate = QtGui.QPushButton("Update list")
        #self.butRuns.setMaximumWidth(90)

        self.cbx_deploy_hotpix = QtGui.QCheckBox('Deploy hotpix mask')
        self.cbx_deploy_hotpix.setChecked( cp.dark_deploy_hotpix.value() )
        
        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addWidget(self.labRuns)
        self.hbox.addWidget(self.butRuns)
        self.hbox.addWidget(self.guirange)
        self.hbox.addWidget(self.labDets)
        self.hbox.addWidget(self.butDets)
        self.hbox.addWidget(self.butUpdate)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.cbx_deploy_hotpix)
    
        self.setLayout(self.hbox)

        #self.connect( self.ediExp,     QtCore.SIGNAL('editingFinished ()'), self.processEdiExp )
        self.connect( self.butRuns,     QtCore.SIGNAL('clicked()'),          self.onButRuns  )
        self.connect( self.butDets,     QtCore.SIGNAL('clicked()'),          self.onButDets  )
        self.connect( self.butUpdate,   QtCore.SIGNAL('clicked()'),          self.onButUpdate )
        self.connect( self.cbx_deploy_hotpix, QtCore.SIGNAL('stateChanged(int)'), self.on_cbx ) 

        self.connect( self.guirange, QtCore.SIGNAL('update(QString)'), self.updateRunRange )

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
        self.butUpdate.setToolTip('Update list of runs')


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
        self.setMinimumSize(390,28)
 

    def setStyleButtons(self):
        self.butRuns.setStyleSheet(cp.styleButton)
        self.butDets.setStyleSheet(cp.styleButton)
        self.butUpdate.setStyleSheet(cp.styleButton)
        self.setButtonsVisible()


    def setButtonsVisible(self):
        show_run_range = self.dark_list_show_runs.value() == 'in range' # self.list_of_show_runs[0]
        #self.guirange.setFieldsEnable(show_run_range)
        self.guirange.setVisible(show_run_range)

        # DO NOT SHOW CHECKBOX!
        self.cbx_deploy_hotpix.  setVisible(False)

 
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


    def updateRunRange(self, text):
        fields = str(text).split(':')
        msg = 'updateRunRange %s'%text
        logger.info(msg, __name__ )
        if fields[0] == 'from' :
            self.dark_list_run_min.setValue(int(fields[1]))
        if fields[0] == 'to' :
            self.dark_list_run_max.setValue(int(fields[1]))

        #self.updateListOfRunsForNumRuns()
        self.updateListOfRuns(self.butUpdate, 'Update list')


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


    def onButUpdate(self):
        self.updateListOfRuns(self.butUpdate, 'Update list')


    def setRuns(self, txt='None'):
        #print 'setType', txt
        self.dark_list_show_runs.setValue(txt)
        self.updateListOfRuns(self.butRuns, txt)


    def setDets(self, txt='None'):
        #print 'setType', txt
        self.dark_list_show_dets.setValue(txt)
        self.updateListOfRuns(self.butDets, txt)

    
    def updateListOfRuns(self, but, txt='None') :
        but.setText('WAIT...')        
        but.setStyleSheet(cp.styleButtonBad)

        if cp.guidarklist is not None :
            cp.guidarklist.updateList()

        txt_on_but = txt if txt == 'Update list' else txt + self.char_expand 
        but.setText(txt_on_but)        
        but.setStyleSheet(cp.styleButton)


    def updateListOfRunsForNumRuns(self) :
        """DEPRICATED"""
        logger.info('WAIT for list of runs update...', __name__ )        
        if cp.guidarklist is not None :
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
