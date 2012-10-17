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

import ConfigParametersCorAna as cp

from GUIBeamZeroPars import *

from GUISpecularPars import *
#---------------------
#  Class definition --
#---------------------
class GUIBatchInfoLeft ( QtGui.QWidget ) :
    """GUI Batch Info Left Panel"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Batch Info Left Panel')
        self.setFrame()
 
        self.list_of_modes  = ['Transmission', 'Specular']
        self.char_expand    = u' \u25BE' # down-head triangle

        self.titDistance   = QtGui.QLabel('Sample-Detector Distance (mm):')
        self.titSetupGeom  = QtGui.QLabel('Experiment Setup Geometry:')
        self.titPhotonE    = QtGui.QLabel('X-Ray Photon Energy (keV):')
        self.titNomAngle   = QtGui.QLabel('Nominal Angle:')

        cp.confpars.guibeamzeropars = GUIBeamZeroPars()
        cp.confpars.guispecularpars = GUISpecularPars()

        self.ediDistance   = QtGui.QLineEdit  ( str(cp.confpars.sample_det_dist.value()) )
        self.ediPhotonE    = QtGui.QLineEdit  ( str(cp.confpars.photon_energy.value()) ) 
        self.ediNomAngle   = QtGui.QLineEdit  ( str(cp.confpars.nominal_angle.value()) ) 
        self.butSetupGeom  = QtGui.QPushButton( cp.confpars.exp_setup_geom.value() + self.char_expand  ) 

        self.ediNomAngle.setReadOnly( True ) 
        self.ediPhotonE .setReadOnly( True ) 

        self.popupMenuMode = QtGui.QMenu()
        for mode in self.list_of_modes :
            self.popupMenuMode.addAction( mode )

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.titSetupGeom,            0, 0, 1, 8)
        self.grid.addWidget(self.titDistance,             1, 0, 1, 8)
        self.grid.addWidget(cp.confpars.guibeamzeropars,  2, 0, 1,10)
        self.grid.addWidget(cp.confpars.guispecularpars,  3, 0, 1,10)
        self.grid.addWidget(self.titPhotonE,              5, 0, 1, 8)
        self.grid.addWidget(self.titNomAngle,             6, 0, 1, 8)


        self.grid.addWidget(self.butSetupGeom,            0, 9)
        self.grid.addWidget(self.ediDistance,             1, 9)
        self.grid.addWidget(self.ediPhotonE,              5, 9)
        self.grid.addWidget(self.ediNomAngle,             6, 9)

        self.setLayout(self.grid)

        self.connect( self.butSetupGeom, QtCore.SIGNAL('clicked()'),          self.onButSetupGeom )
        self.connect( self.ediDistance,  QtCore.SIGNAL('editingFinished ()'), self.onEdiDistance )

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
        #self.            setStyleSheet (cp.confpars.styleYellow)
        self.titDistance .setStyleSheet (cp.confpars.styleTitle)
        self.titPhotonE  .setStyleSheet (cp.confpars.styleTitle)
        self.titNomAngle .setStyleSheet (cp.confpars.styleTitle)
        self.titSetupGeom.setStyleSheet (cp.confpars.styleTitle)

        self.ediDistance .setStyleSheet(cp.confpars.styleEdit) 
        self.ediPhotonE  .setStyleSheet(cp.confpars.styleGreen) 
        self.ediNomAngle .setStyleSheet(cp.confpars.styleGreen) 

        self.ediDistance .setAlignment(QtCore.Qt.AlignRight)
        self.ediPhotonE  .setAlignment(QtCore.Qt.AlignRight)
        self.ediNomAngle .setAlignment(QtCore.Qt.AlignRight)

        width = 80
        self.ediDistance .setFixedWidth(width)
        self.ediPhotonE  .setFixedWidth(width)
        self.ediNomAngle .setFixedWidth(width)
        self.butSetupGeom.setFixedWidth(120)


    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp.confpars
            del cp.confpars.guibatchinfoleft # GUIBatchInfoLeft
        except AttributeError:
            pass # silently ignore

    def processClose(self):
        #print 'Close button'
        self.close()

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #print 'moveEvent' 
        pass
#        cp.confpars.posGUIMain = (self.pos().x(),self.pos().y())


    def onButSetupGeom(self):
        action_selected = self.popupMenuMode.exec_(QtGui.QCursor.pos())
        if action_selected is None : return
        self.mode_name = action_selected.text()
        cp.confpars.exp_setup_geom.setValue( self.mode_name )
        self.butSetupGeom.setText( self.mode_name + self.char_expand )
        print ' ---> selected setup geometry mode: ' + self.mode_name


    def onEdiDistance(self):
        cp.confpars.sample_det_dist.setValue( float(self.ediDistance.displayText()) )
        print 'Set sample_det_dist =', cp.confpars.sample_det_dist.value()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIBatchInfoLeft ()
    widget.show()
    app.exec_()

#-----------------------------
