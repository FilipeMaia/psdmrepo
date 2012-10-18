#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIBatchInfo...
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

import ConfigParametersCorAna as cp

from GUIBatchInfoLeft   import *
from GUIBatchInfoRight  import *

#---------------------
#  Class definition --
#---------------------
class GUIBatchInfo ( QtGui.QWidget ) :
    """GUI Batch Info Left Panel"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Batch Info')
        self.setFrame()
 
        self.titTitle  = QtGui.QLabel('Batch Information')
        self.titStatus = QtGui.QLabel('Status: Ready!')
        self.butClose  = QtGui.QPushButton('Close') 
        self.butApply  = QtGui.QPushButton('Apply') 
        self.butShow   = QtGui.QPushButton('Show Image') 
        cp.confpars.guibatchinfoleft  = GUIBatchInfoLeft()
        cp.confpars.guibatchinforight = GUIBatchInfoRight()

        self.hboxM = QtGui.QHBoxLayout()
        self.hboxM.addWidget(cp.confpars.guibatchinfoleft)
        self.hboxM.addWidget(cp.confpars.guibatchinforight)

        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.titStatus)
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.butClose)
        self.hboxB.addWidget(self.butApply)
        self.hboxB.addWidget(self.butShow )

        self.vbox  = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.titTitle)
        self.vbox.addLayout(self.hboxM)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)
        
        self.connect( self.butClose, QtCore.SIGNAL('clicked()'), self.onClose )
        self.connect( self.butApply, QtCore.SIGNAL('clicked()'), self.onApply )
        self.connect( self.butShow , QtCore.SIGNAL('clicked()'), self.onShow  )

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        #msg_edi = 'WARNING: whatever you edit may be incorrect...\nIt is recommended to use the '
        #self.butInstr  .setToolTip('Select the instrument name from the pop-up menu.')
        pass

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        #self.          setStyleSheet (cp.confpars.styleYellow)
        self.titTitle .setStyleSheet (cp.confpars.styleTitle + 'font-size: 18pt; font-family: Courier; font-weight: bold')
        self.titStatus.setStyleSheet (cp.confpars.styleTitle)
        self.butClose .setStyleSheet (cp.confpars.styleGray)
        self.butApply .setStyleSheet (cp.confpars.styleGray) 
        self.butShow  .setStyleSheet (cp.confpars.styleGray) 

        self.titTitle .setAlignment(QtCore.Qt.AlignCenter)
        #self.titTitle .setBold()

    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp.confpars
            del cp.confpars.guibatchinfo # GUIBatchInfo
        except AttributeError:
            pass # silently ignore

    def onClose(self):
        #print 'Close button'
        self.close()

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #print 'moveEvent' 
        pass
#        cp.confpars.posGUIMain = (self.pos().x(),self.pos().y())


    def onApply(self):
        print 'Apply button is empty'

    def onShow(self):
        print 'Show button is empty'

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIBatchInfo ()
    widget.show()
    app.exec_()

#-----------------------------
