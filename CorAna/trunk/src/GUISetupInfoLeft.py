#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISetupInfoLeft...
#
#------------------------------------------------------------------------

"""GUI Setup Info Left Panel"""

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

from GUISetupSpecular    import *
from GUISetupBeamZero    import *
from GUISetupData        import *
from GUIImgSizePosition  import *
from Logger              import logger

#---------------------
#  Class definition --
#---------------------
class GUISetupInfoLeft ( QtGui.QWidget ) :
    """GUI Setup Info Left Panel"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(20, 40, 400, 500)
        self.setWindowTitle('Setup Info Left Panel')
        self.setFrame()
 
        self.titDistance   = QtGui.QLabel('Sample-Detector Distance (mm):')
        self.titSetupGeom  = QtGui.QLabel('Experiment Setup Geometry:')

        cp.guiimgsizeposition = GUIImgSizePosition()

        self.ediDistance   = QtGui.QLineEdit  ( str(cp.sample_det_dist.value()) )

        self.hboxD = QtGui.QHBoxLayout()
        self.hboxW = QtGui.QHBoxLayout()

        self.makeTabBar()
        self.guiSelector()

        self.hboxD.addWidget(self.titDistance)
        #self.hboxD.addStretch(1)     
        self.hboxD.addWidget(self.ediDistance)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.titSetupGeom)
        self.vbox.addWidget(self.tab_bar)
        self.vbox.addLayout(self.hboxD)
        self.vbox.addLayout(self.hboxW)
        self.vbox.addWidget(cp.guiimgsizeposition)
        self.vbox.addStretch(1)     
        self.setLayout(self.vbox)
        
        self.connect( self.ediDistance,  QtCore.SIGNAL('editingFinished()'),        self.onEdiDistance )

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def makeTabBar(self,mode=None) :
        #if mode is not None : self.tab_bar.close()
        self.tab_bar = QtGui.QTabBar()

        self.list_of_modes  = ['Beam Zero', 'Specular', 'Data']

        self.ind_tab_bar_beamzero = self.tab_bar.addTab( self.list_of_modes[0] )
        self.ind_tab_bar_specul   = self.tab_bar.addTab( self.list_of_modes[1] )
        self.ind_tab_bar_data     = self.tab_bar.addTab( self.list_of_modes[2] )

        self.tab_bar.setTabTextColor(self.ind_tab_bar_beamzero,QtGui.QColor('blue'))
        self.tab_bar.setTabTextColor(self.ind_tab_bar_specul,  QtGui.QColor('blue'))
        self.tab_bar.setTabTextColor(self.ind_tab_bar_data,    QtGui.QColor('blue'))
        self.tab_bar.setShape(QtGui.QTabBar.RoundedNorth)

        logger.info(' makeTabBar - set mode: ' + cp.exp_setup_geom.value(), __name__)

        #self.tab_bar.setTabEnabled(self.list_of_modes.index(cp.exp_setup_geom.value()),False)
        self.tab_bar.setCurrentIndex(self.list_of_modes.index(cp.exp_setup_geom.value()))

        self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.onTabBar)


    def guiSelector(self):
        try :
            self.gui_win.close()
        except AttributeError :
            pass

        if cp.exp_setup_geom.value() == self.list_of_modes[0] :
            cp.guisetupbeamzero = GUISetupBeamZero()
            self.gui_win = cp.guisetupbeamzero

        if cp.exp_setup_geom.value() == self.list_of_modes[1] :
            cp.guisetupspecular = GUISetupSpecular()
            self.gui_win = cp.guisetupspecular

        if cp.exp_setup_geom.value() == self.list_of_modes[2] :
            cp.guisetupdata = GUISetupData()
            self.gui_win = cp.guisetupdata

        self.hboxW.addWidget(self.gui_win)


    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        msg_edit = 'Edit field'
        self.ediDistance .setToolTip( msg_edit )

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setMinimumHeight(500)
        self.             setStyleSheet (cp.styleBkgd)
        self.titDistance .setStyleSheet (cp.styleTitle)
        self.titSetupGeom.setStyleSheet (cp.styleTitle)

        self.ediDistance .setStyleSheet(cp.styleEdit) 
        #self.boxSetupGeom.setStyleSheet(cp.styleBox)

        self.ediDistance .setAlignment(QtCore.Qt.AlignRight)
        width = 80
        self.ediDistance .setFixedWidth(width)
        #self.boxSetupGeom.setFixedWidth(120)

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

        try    : cp.guibeamzeropars.close()
        except : pass

        try    : cp.guisetupbeamzero.close()
        except : pass

        try    : cp.guiguisetupspecular.close()
        except : pass

        try    : cp.guisetupdata.close()
        except : pass

        try    : guiimgsizeposition.close()
        except : pass

        try    : del cp.guisetupinfoleft # GUISetupInfoLeft
        except : pass # silently ignore

    def onClose(self):
        logger.debug('onClose', __name__) 
        self.close()

    def onTabBar(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        cp.exp_setup_geom.setValue( tab_name )
        logger.info(' ---> selected tab: ' + str(tab_ind) + ' - setup geometry mode: ' + tab_name, __name__)
        self.guiSelector()

    def setPopupMenu(self):
        self.popupMenuMode = QtGui.QMenu()
        for mode in self.list_of_modes :
            self.popupMenuMode.addAction( mode )

    def onButSetupGeom(self):
        action_selected = self.popupMenuMode.exec_(QtGui.QCursor.pos())
        if action_selected is None : return
        self.mode_name = action_selected.text()
        cp.exp_setup_geom.setValue( self.mode_name )
        self.butSetupGeom.setText( self.mode_name + cp.char_expand )
        logger.info(' ---> selected setup geometry mode: ' + self.mode_name, __name__)
        self.guiSelector()

    def onEdiDistance(self):
        cp.sample_det_dist.setValue( float(self.ediDistance.displayText()) )
        logger.info('Set sample_det_dist = ' + str(cp.sample_det_dist.value()), __name__ )

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUISetupInfoLeft ()
    widget.show()
    app.exec_()

#-----------------------------
