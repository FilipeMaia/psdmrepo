#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIBatchInfoLeft...
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

from ConfigParametersCorAna import confpars as cp

from GUIBeamZeroPars     import *
from GUISpecularPars     import *
from GUITransmissionPars import *
from GUIImgSizePosition  import *
from Logger              import logger

#---------------------
#  Class definition --
#---------------------
class GUIBatchInfoLeft ( QtGui.QWidget ) :
    """GUI Batch Info Left Panel"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Batch Info Left Panel')
        self.setFrame()
 
        self.titDistance   = QtGui.QLabel('Sample-Detector Distance (mm):')
        self.titSetupGeom  = QtGui.QLabel('Experiment Setup Geometry:')
        self.titPhotonE    = QtGui.QLabel('X-Ray Photon Energy (keV):')
        self.titNomAngle   = QtGui.QLabel('Nominal Angle:')
        self.titRealAngle  = QtGui.QLabel('Real Angle:')

        cp.guiimgsizeposition = GUIImgSizePosition()
        cp.guibeamzeropars    = GUIBeamZeroPars()

        self.ediDistance   = QtGui.QLineEdit  ( str(cp.sample_det_dist.value()) )
        self.ediPhotonE    = QtGui.QLineEdit  ( str(cp.photon_energy.value()) ) 
        self.ediNomAngle   = QtGui.QLineEdit  ( str(cp.nominal_angle.value()) ) 
        self.ediRealAngle  = QtGui.QLineEdit  ( str(cp.real_angle.value()) ) 

        self.ediPhotonE  .setReadOnly( True ) 
        self.ediNomAngle .setReadOnly( True ) 
        self.ediRealAngle.setReadOnly( True ) 

        self.hboxD = QtGui.QHBoxLayout()
        self.hboxW = QtGui.QHBoxLayout()
        self.hboxE = QtGui.QHBoxLayout()
        self.hboxA = QtGui.QHBoxLayout()
        self.hboxR = QtGui.QHBoxLayout()

        self.makeTabBar()
        self.guiSelector()
        self.guiAnglePanel()

        self.hboxD.addWidget(self.titDistance)
        self.hboxD.addStretch(1)     
        self.hboxD.addWidget(self.ediDistance)

        self.hboxA.addWidget(self.titNomAngle)
        self.hboxA.addStretch(1)     
        self.hboxA.addWidget(self.ediNomAngle)

        self.hboxR.addWidget(self.titRealAngle)
        self.hboxR.addStretch(1)     
        self.hboxR.addWidget(self.ediRealAngle)

        self.hboxE.addWidget(self.titPhotonE)
        self.hboxE.addStretch(1)     
        self.hboxE.addWidget(self.ediPhotonE)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.titSetupGeom)
        self.vbox.addWidget(self.tab_bar)
        self.vbox.addLayout(self.hboxD)
        self.vbox.addLayout(self.hboxW)
        self.vbox.addWidget(cp.guibeamzeropars)
        self.vbox.addWidget(cp.guiimgsizeposition)
        self.vbox.addLayout(self.hboxE)
        self.vbox.addLayout(self.hboxA)
        self.vbox.addLayout(self.hboxR)
        self.setLayout(self.vbox)
        
        self.connect( self.ediDistance,  QtCore.SIGNAL('editingFinished()'),        self.onEdiDistance )

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def guiAnglePanel(self):
        pass
#        try :
#            #self.self.ediNomAngle.close()
#            #self.self.titNomAngle.close()
#            self.hboxA.removeWidget(self.ediNomAngle)
#            self.hboxA.removeWidget(self.titNomAngle)
#            self.hboxA.update()
#        except AttributeError :
#            pass

#        if cp.exp_setup_geom.value() == self.list_of_modes[0] :
#            self.hboxA.setEnabled(False)

#        if cp.exp_setup_geom.value() == self.list_of_modes[1] :
#            self.hboxA.setEnabled(True)


    def makeTabBar(self,mode=None) :
        #if mode != None : self.tab_bar.close()
        self.tab_bar = QtGui.QTabBar()

        self.list_of_modes  = ['Transmission', 'Specular']

        self.ind_tab_bar_transm = self.tab_bar.addTab( self.list_of_modes[0] )
        self.ind_tab_bar_specul = self.tab_bar.addTab( self.list_of_modes[1] )

        self.tab_bar.setTabTextColor(self.ind_tab_bar_transm,QtGui.QColor('green'))
        self.tab_bar.setTabTextColor(self.ind_tab_bar_specul,QtGui.QColor('blue'))
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
            cp.guitransmissionpars = GUITransmissionPars() # GUIBeamZeroPars()
            self.gui_win = cp.guitransmissionpars

        if cp.exp_setup_geom.value() == self.list_of_modes[1] :
            cp.guispecularpars = GUISpecularPars()
            self.gui_win = cp.guispecularpars

        self.hboxW.addWidget(self.gui_win)


    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        msg_edit = 'Edit field'
        msg_info = 'Information field'
        msg_sele = 'Selection field'
        
        #self.boxSetupGeom.setToolTip( msg_sele )
        self.ediDistance .setToolTip( msg_edit )
        self.ediPhotonE  .setToolTip( msg_info )
        self.ediNomAngle .setToolTip( msg_info )
        self.ediRealAngle.setToolTip( msg_info )

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.             setStyleSheet (cp.styleBkgd)
        self.titDistance .setStyleSheet (cp.styleTitle)
        self.titPhotonE  .setStyleSheet (cp.styleTitle)
        self.titNomAngle .setStyleSheet (cp.styleTitle)
        self.titRealAngle.setStyleSheet (cp.styleTitle)
        self.titSetupGeom.setStyleSheet (cp.styleTitle)

        self.ediDistance .setStyleSheet(cp.styleEdit) 
        self.ediPhotonE  .setStyleSheet(cp.styleEditInfo) 
        self.ediNomAngle .setStyleSheet(cp.styleEditInfo) 
        self.ediRealAngle.setStyleSheet(cp.styleEditInfo) 
        #self.boxSetupGeom.setStyleSheet(cp.styleBox)

        self.ediDistance .setAlignment(QtCore.Qt.AlignRight)
        self.ediPhotonE  .setAlignment(QtCore.Qt.AlignRight)
        self.ediNomAngle .setAlignment(QtCore.Qt.AlignRight)
        self.ediRealAngle.setAlignment(QtCore.Qt.AlignRight)

        width = 80
        self.ediDistance .setFixedWidth(width)
        self.ediPhotonE  .setFixedWidth(width)
        self.ediNomAngle .setFixedWidth(width)
        self.ediRealAngle.setFixedWidth(width)
        #self.boxSetupGeom.setFixedWidth(120)

    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        logger.debug('moveEvent', __name__) 
#        cp.posGUIMain = (self.pos().x(),self.pos().y())

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__) 

        try    : cp.guibeamzeropars.close()
        except : pass

        try    : cp.guitransmissionpars.close()
        except : pass

        try    : cp.guispecularpars.close()
        except : pass

        try    : guiimgsizeposition.close()
        except : pass

        try    : del cp.guibatchinfoleft # GUIBatchInfoLeft
        except : pass # silently ignore

    def onClose(self):
        logger.info('onClose', __name__) 
        self.close()

    def onTabBar(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        cp.exp_setup_geom.setValue( tab_name )
        logger.info(' ---> selected tab: ' + str(tab_ind) + ' - setup geometry mode: ' + tab_name, __name__)
        self.guiSelector()
        self.guiAnglePanel()

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
        self.guiAnglePanel()

    def onEdiDistance(self):
        cp.sample_det_dist.setValue( float(self.ediDistance.displayText()) )
        logger.info('Set sample_det_dist = ' + str(cp.sample_det_dist.value()), __name__ )

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIBatchInfoLeft ()
    widget.show()
    app.exec_()

#-----------------------------
