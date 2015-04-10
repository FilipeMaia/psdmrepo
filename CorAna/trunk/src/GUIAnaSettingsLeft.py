#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIAnaSettingsLeft...
#
#------------------------------------------------------------------------

"""GUI sets parameters for analysis"""

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
from Logger                 import logger
from GUIAnaPartitions       import *
from GUIAnaSettingsOptions  import *

#---------------------
#  Class definition --
#---------------------
class GUIAnaSettingsLeft ( QtGui.QWidget ) :
    """GUI sets parameters for analysis"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(20, 40, 390, 30)
        self.setWindowTitle('Analysis Settings Left')
        self.setFrame()

        self.tit_ana_type = QtGui.QLabel('Select Analysis Type:')
        self.hboxW = QtGui.QHBoxLayout()
        self.hboxS = QtGui.QHBoxLayout()
        self.makeTabBar()
        self.guiSelector()
 
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.tit_ana_type)
        self.vbox.addWidget(self.tab_bar)
        self.vbox.addLayout(self.hboxW)
        self.vbox.addLayout(self.hboxS)
        self.vbox.addStretch(1) 
        self.setLayout(self.vbox)

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        msg = 'Use these tabs to switch between analysis modes.'
        self.tab_bar.setToolTip(msg)

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

        self.list_ana_types  = ['Static', 'Dynamic']

        self.ind_tab_static  = self.tab_bar.addTab( self.list_ana_types[0] )
        self.ind_tab_dynamic = self.tab_bar.addTab( self.list_ana_types[1] )

        self.tab_bar.setTabTextColor(self.ind_tab_static, QtGui.QColor('green'))
        self.tab_bar.setTabTextColor(self.ind_tab_dynamic,QtGui.QColor('blue'))
        self.tab_bar.setShape(QtGui.QTabBar.RoundedNorth)

        logger.info(' makeTabBar - set mode: ' + cp.ana_type.value(), __name__)

        self.tab_bar.setCurrentIndex(self.list_ana_types.index(cp.ana_type.value()))

        self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.onTabBar)


    def guiSelector(self):
        try    : self.gui_win.close()
        except : pass

        try    : cp.guianasettingsoptions.close()
        except : pass

        try    : del cp.guianasettingsoptions
        except : pass

        if   cp.ana_type.value() == self.list_ana_types[0] :
            pass

        elif cp.ana_type.value() == self.list_ana_types[1] :
            cp.guianasettingsoptions = GUIAnaSettingsOptions(self)
            cp.guianasettingsoptions.setMinimumWidth(380)
            self.hboxS.addWidget(cp.guianasettingsoptions)

        self.gui_win = GUIAnaPartitions(self)
        self.gui_win.setMinimumWidth(380)
        self.hboxW.addWidget(self.gui_win)


    def setStyle(self):
        self.setFixedWidth(390)
        self.setStyleSheet(cp.styleBkgd)
        self.tit_ana_type.setStyleSheet (cp.styleTitle)


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
        try    : del cp.guianasettingsleft # GUIAnaSettingsLeft
        except : pass # silently ignore

        try    : cp.guianasettingsoptions.close()
        except : pass

        try    : self.gui_win.close()
        except : pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def onShow(self):
        logger.debug('onShow - is not implemented yet...', __name__)

    def onApply(self):
        logger.debug('onApply - is already applied...', __name__)


#    def onAnaRadioGrp(self):
#        if self.rad_ana_stat.isChecked() : cp.ana_type.setValue(self.list_ana_types[0])
#        if self.rad_ana_dyna.isChecked() : cp.ana_type.setValue(self.list_ana_types[1])
#        logger.info('onAnaRadioGrp - set cp.ana_type = '+ cp.ana_type.value(), __name__)

    def onTabBar(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        cp.ana_type.setValue( tab_name )
        logger.info(' ---> selected tab: ' + str(tab_ind) + ' - setup geometry mode: ' + tab_name, __name__)
        self.guiSelector()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIAnaSettingsLeft ()
    widget.show()
    app.exec_()

#-----------------------------
