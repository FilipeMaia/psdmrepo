#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIRun ...
#
#------------------------------------------------------------------------

"""GUI for run control."""

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
from Logger                 import logger
from FileNameManager        import fnm

#from GUIRunLeft   import *
#from GUIRunRight  import *

#---------------------
#  Class definition --
#---------------------
class GUIRun ( QtGui.QWidget ) :
    """GUI for run control"""

    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 500, 200)
        self.setWindowTitle('Run Control and Monitoring')
        self.setFrame()
 
        self.tit_title  = QtGui.QLabel('Run Control and Monitoring')
        self.tit_status = QtGui.QLabel('Status:')
        self.tit_path   = QtGui.QLabel('Data:')
        self.tit_flat   = QtGui.QLabel('Flat:')
        self.tit_blam   = QtGui.QLabel('Blam:')

        self.but_close  = QtGui.QPushButton('Close') 
        self.but_apply  = QtGui.QPushButton('Save') 

        self.edi_path = QtGui.QLineEdit( fnm.path_data_xtc() )        
        self.edi_path.setReadOnly( True )   

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.tit_status)
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.but_close)
        self.hboxB.addWidget(self.but_apply)

        self.grid = QtGui.QGridLayout()
        self.grid_row = 1
        self.grid.addWidget(self.tit_title,     self.grid_row,   0, 1, 6)          
        self.grid.addWidget(self.tit_path,      self.grid_row+1, 0)
        self.grid.addWidget(self.edi_path,      self.grid_row+1, 1, 1, 7)
        self.grid.addWidget(self.tit_flat,      self.grid_row+2, 0)
        self.grid.addWidget(self.tit_blam,      self.grid_row+3, 0)
        self.grid.addLayout(self.hboxB,         self.grid_row+5, 0, 1, 8)

        self.setLayout(self.grid)
        
        self.connect( self.but_close, QtCore.SIGNAL('clicked()'), self.onClose )
        self.connect( self.but_apply, QtCore.SIGNAL('clicked()'), self.onSave  )

        self.showToolTips()
        self.setStyle()
        self.setStatus()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        self           .setToolTip('This GUI is intended for run control and monitoring.')
        self.but_close .setToolTip('Close this window.')
        self.but_apply .setToolTip('Apply changes to configuration parameters.')
        #self.but_show  .setToolTip('Show ...')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setMinimumHeight(200)
        self.           setStyleSheet (cp.styleBkgd)
        self.tit_title .setStyleSheet (cp.styleTitleBold)
        self.tit_status.setStyleSheet (cp.styleTitle)
        self.but_close .setStyleSheet (cp.styleButton)
        self.but_apply .setStyleSheet (cp.styleButton) 
        #self.but_show  .setStyleSheet (cp.styleButton) 
        self.tit_path  .setStyleSheet   (cp.styleLabel)
        self.tit_flat  .setStyleSheet   (cp.styleLabel)
        self.tit_blam  .setStyleSheet   (cp.styleLabel)


        self.edi_path.setStyleSheet   (cp.styleEditInfo)
        self.edi_path.setAlignment    (QtCore.Qt.AlignRight)

        self.tit_title .setAlignment(QtCore.Qt.AlignCenter)
        #self.titTitle .setBold()

    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)

        #try    : cp.guisystemsettingsleft.close()
        #except : pass

        #try    : cp.guisystemsettingsright.close()
        #except : pass

        try    : del cp.guirun # GUIRun
        except : pass

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def onSave(self):
        fname = cp.fname_cp.value()
        logger.debug('onSave:', __name__)
        cp.saveParametersInFile( fname )

    def onShow(self):
        logger.debug('onShow - is not implemented yet...', __name__)

    def setStatus(self, status_index=0, msg=''):

        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0 : self.tit_status.setStyleSheet(cp.styleStatusGood)
        if status_index == 1 : self.tit_status.setStyleSheet(cp.styleStatusWarning)
        if status_index == 2 : self.tit_status.setStyleSheet(cp.styleStatusAlarm)

        self.tit_status.setText('Status: ' + list_of_states[status_index] + msg)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIRun ()
    widget.show()
    app.exec_()

#-----------------------------
