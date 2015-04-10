#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISetupInfoRight...
#
#------------------------------------------------------------------------

"""GUI Setup Info Right Panel"""

#------------------------------
#  Module's version from CVS --
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

from ConfigParametersCorAna import confpars as cp
from GUIKineticMode         import *
from GUINonKineticMode      import *
from GUISetupPars           import *
from GUISetupEnergyAngle    import *
from Logger                 import logger

#---------------------
#  Class definition --
#---------------------
class GUISetupInfoRight ( QtGui.QWidget ) :
    """GUI Setup Info Right Panel"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(20, 40, 400, 500)
        self.setWindowTitle('Setup Info Right Panel')
        self.setFrame()
        self.setMinimumWidth(400) 

        self.hboxW = QtGui.QHBoxLayout()
        self.makeTabBar()
        self.guiSelector()

        self.tit_camera_mode = QtGui.QLabel('CCD Mode:')
   
        cp.guisetuppars        = GUISetupPars()
        cp.guisetupenergyangle = GUISetupEnergyAngle()

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.tit_camera_mode)
        self.vbox.addWidget(self.tab_bar)
        self.vbox.addLayout(self.hboxW)
        #self.vbox.addWidget(cp.guikineticmode)
        self.vbox.addWidget(cp.guisetuppars)
        self.vbox.addWidget(cp.guisetupenergyangle)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)

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

    def makeTabBar(self,mode=None) :
        #if mode is not None : self.tab_bar.close()
        self.tab_bar = QtGui.QTabBar()

        self.list_of_kin_modes  = ['Non-Kinetics', 'Kinetics']

        self.ind_tab_nonkinetic = self.tab_bar.addTab( self.list_of_kin_modes[0] )
        self.ind_tab_kinetic    = self.tab_bar.addTab( self.list_of_kin_modes[1] )

        self.tab_bar.setTabTextColor(self.ind_tab_kinetic,   QtGui.QColor('blue'))
        self.tab_bar.setTabTextColor(self.ind_tab_nonkinetic,QtGui.QColor('blue'))
        self.tab_bar.setShape(QtGui.QTabBar.RoundedNorth)

        logger.info('makeTabBar - set mode: ' + cp.kin_mode.value(), __name__)

        self.tab_bar.setCurrentIndex(self.list_of_kin_modes.index(cp.kin_mode.value()))

        self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.onTabBar)


    def guiSelector(self):
        try :
            self.gui_win.close()
        except AttributeError :
            pass

        if cp.kin_mode.value() == self.list_of_kin_modes[0] :
            cp.guinonkineticmode = GUINonKineticMode() 
            self.gui_win = cp.guinonkineticmode

        if cp.kin_mode.value() == self.list_of_kin_modes[1] :
            cp.guikineticmode = GUIKineticMode()
            self.gui_win = cp.guikineticmode

        self.hboxW.addWidget(self.gui_win)

    def onTabBar(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        cp.kin_mode.setValue( tab_name )
        logger.info(' ---> selected tab: ' + str(tab_ind) + ' - setup mode: ' + tab_name, __name__)
        self.guiSelector()
 
    def setStyle(self):
        self.setMinimumHeight(500)
        self.setStyleSheet(cp.styleBkgd)
        self.tit_camera_mode.setStyleSheet(cp.styleTitle)
        
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
        try    : del cp.guisetupinforight # GUISetupInfoRight
        except : pass # silently ignore

        try    : cp.guikineticmode.close()
        except : pass

        try    : cp.guinonkineticmode.close()
        except : pass

        try    : cp.guisetuppars.close()
        except : pass

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUISetupInfoRight ()
    widget.show()
    app.exec_()

#-----------------------------
