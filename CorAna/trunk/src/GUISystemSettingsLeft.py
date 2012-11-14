#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISystemSettingsLeft...
#
#------------------------------------------------------------------------

"""GUI sets system parameters"""

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
from ConfigParametersCorAna   import confpars as cp
from Logger                   import logger
from GUICCDSettings           import *
from GUICCDCorrectionSettings import *

#---------------------
#  Class definition --
#---------------------
class GUISystemSettingsLeft ( QtGui.QWidget ) :
    """GUI sets system parameters"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('System Settings Left')
        self.setFrame()

        self.tit_sys_ram_size = QtGui.QLabel('Available RAM [MB]:')
        self.edi_sys_ram_size = QtGui.QLineEdit( str( cp.sys_ram_size.value() ) )        

        cp.guiccdsettings            = GUICCDSettings()
        cp.guiccdcorrectionssettings = GUICCDCorrectionSettings()

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.tit_sys_ram_size)
        self.hbox.addWidget(self.edi_sys_ram_size)
        self.hbox.addStretch(1) 

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(cp.guiccdsettings)
        self.vbox.addWidget(cp.guiccdcorrectionssettings)
        self.vbox.addStretch(1) 
        self.vbox.addLayout(self.hbox)

        self.setLayout(self.vbox)

        self.connect(self.edi_sys_ram_size, QtCore.SIGNAL('editingFinished()'), self.onEdit )

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        msg = 'GUI sets system parameters.'
        self.tit_sys_ram_size.setToolTip(msg)

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setMinimumWidth(450)
        self.setStyleSheet(cp.styleBkgd)
        self.tit_sys_ram_size.setStyleSheet(cp.styleTitle)
        self.edi_sys_ram_size.setStyleSheet(cp.styleEdit)
        self.edi_sys_ram_size.setFixedWidth(80)
        self.edi_sys_ram_size.setAlignment(QtCore.Qt.AlignRight) 



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
        try    : del cp.guisystemsettingsleft # GUISystemSettingsLeft
        except : pass

        try    : cp.guiccdsettings.close()
        except : pass

        try    : cp.guiccdcorrectionssettings.close()
        except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def onShow(self):
        logger.debug('onShow - is not implemented yet...', __name__)

    def onApply(self):
        logger.info('onApply - is already applied...', __name__)

    def onEdit(self):

        if self.edi_sys_ram_size.isModified() :            
            self.edi = self.edi_sys_ram_size 
            self.par = cp.sys_ram_size
            self.tit = 'sys_ram_size'

        else : return # no-modification

        self.edi.setModified(False)
        self.par.setValue( self.edi.displayText() )        
        msg = 'onEdit - set value of ' + self.tit  + ': ' + str( self.par.value())
        logger.info(msg, __name__ )

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUISystemSettingsLeft ()
    widget.show()
    app.exec_()

#-----------------------------
