#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISetupEnergyAngle...
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
from Logger                 import logger

#---------------------
#  Class definition --
#---------------------
class GUISetupEnergyAngle ( QtGui.QWidget ) :
    """GUI Setup Info Left Panel"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Setup Info Left Panel')
        self.setFrame()
 
        self.titPhotonE    = QtGui.QLabel('X-Ray Photon Energy (keV):')
        self.titNomAngle   = QtGui.QLabel('Nominal Angle (deg):')
        self.titRealAngle  = QtGui.QLabel('Real Angle (deg):')
        self.titTiltAngle  = QtGui.QLabel('Tilt Angle (deg):')

        #print 'cp.real_angle.value():', cp.real_angle.value()

        self.ediPhotonE    = QtGui.QLineEdit  ( '%8.3f' % (cp.photon_energy.value()) ) 
        self.ediNomAngle   = QtGui.QLineEdit  ( '%8.3f' % (cp.nominal_angle.value()) ) 
        self.ediRealAngle  = QtGui.QLineEdit  ( '%8.3f' % (cp.real_angle.value()) ) 
        self.ediTiltAngle  = QtGui.QLineEdit  ( '%8.3f' % (cp.tilt_angle.value()) ) 

        self.ediPhotonE  .setReadOnly( True ) 
        self.ediNomAngle .setReadOnly( True ) 
        self.ediRealAngle.setReadOnly( True ) 
        self.ediTiltAngle.setReadOnly( True ) 

        self.hboxE = QtGui.QHBoxLayout()
        self.hboxA = QtGui.QHBoxLayout()
        self.hboxR = QtGui.QHBoxLayout()
        self.hboxT = QtGui.QHBoxLayout()

        self.hboxA.addWidget(self.titNomAngle)
        self.hboxA.addStretch(1)     
        self.hboxA.addWidget(self.ediNomAngle)

        self.hboxR.addWidget(self.titRealAngle)
        self.hboxR.addStretch(1)     
        self.hboxR.addWidget(self.ediRealAngle)

        self.hboxT.addWidget(self.titTiltAngle)
        self.hboxT.addStretch(1)     
        self.hboxT.addWidget(self.ediTiltAngle)

        self.hboxE.addWidget(self.titPhotonE)
        self.hboxE.addStretch(1)     
        self.hboxE.addWidget(self.ediPhotonE)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.hboxE)
        self.vbox.addLayout(self.hboxA)
        self.vbox.addLayout(self.hboxR)
        self.vbox.addLayout(self.hboxT)
        self.setLayout(self.vbox)
        
        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        msg_edit = 'Edit field'
        msg_info = 'Information field'
        msg_sele = 'Selection field'
        
        self.ediPhotonE  .setToolTip( msg_info )
        self.ediNomAngle .setToolTip( msg_info )
        self.ediRealAngle.setToolTip( msg_info )
        self.ediTiltAngle.setToolTip( msg_info )

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.             setStyleSheet (cp.styleBkgd)
        self.titPhotonE  .setStyleSheet (cp.styleTitle)
        self.titNomAngle .setStyleSheet (cp.styleTitle)
        self.titRealAngle.setStyleSheet (cp.styleTitle)
        self.titTiltAngle.setStyleSheet (cp.styleTitle)

        self.ediPhotonE  .setStyleSheet(cp.styleEditInfo) 
        self.ediNomAngle .setStyleSheet(cp.styleEditInfo) 
        self.ediRealAngle.setStyleSheet(cp.styleEditInfo) 
        self.ediTiltAngle.setStyleSheet(cp.styleEditInfo) 

        self.ediPhotonE  .setAlignment(QtCore.Qt.AlignRight)
        self.ediNomAngle .setAlignment(QtCore.Qt.AlignRight)
        self.ediRealAngle.setAlignment(QtCore.Qt.AlignRight)
        self.ediTiltAngle.setAlignment(QtCore.Qt.AlignRight)

        width = 80
        self.ediPhotonE  .setFixedWidth(width)
        self.ediNomAngle .setFixedWidth(width)
        self.ediRealAngle.setFixedWidth(width)
        self.ediTiltAngle.setFixedWidth(width)

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
        try    : del cp.guisetupenergyangle # GUISetupEnergyAngle
        except : pass # silently ignore

    def onClose(self):
        logger.debug('onClose', __name__) 
        self.close()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUISetupEnergyAngle ()
    widget.show()
    app.exec_()

#-----------------------------
