#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUICCDCorrectionSettings...
#
#------------------------------------------------------------------------

"""GUI for CCD Correction Settings"""

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
from Logger import logger
from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------
class GUICCDCorrectionSettings ( QtGui.QWidget ) :
    """GUI for CCD Correction Settings"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('CCD Correction Settings')
        self.setFrame()

        self.tit_ccdcorr_sets         = QtGui.QLabel('CD Correction Settings:')
        self.cbx_ccdcorr_blemish      = QtGui.QCheckBox('Use blemish file', self)
        self.cbx_ccdcorr_flatfield    = QtGui.QCheckBox('Use flatfield file', self)
        self.cbx_ccdcorr_distortion   = QtGui.QCheckBox('Use distortion file', self)
        self.cbx_ccdcorr_parasitic    = QtGui.QCheckBox('Use parasitic file', self)

        self.cbx_ccdcorr_blemish   .setChecked( cp.ccdcorr_blemish     .value() )
        self.cbx_ccdcorr_flatfield .setChecked( cp.ccdcorr_flatfield   .value() )
        self.cbx_ccdcorr_distortion.setChecked( cp.ccdcorr_distortion  .value() )
        self.cbx_ccdcorr_parasitic .setChecked( cp.ccdcorr_parasitic   .value() )

        self.grid = QtGui.QGridLayout()

        self.grid_row = 0
        self.grid.addWidget(self.tit_ccdcorr_sets      ,  self.grid_row+1, 0, 1, 6)     
        self.grid.addWidget(self.cbx_ccdcorr_blemish   ,  self.grid_row+2, 1, 1, 4)     
        self.grid.addWidget(self.cbx_ccdcorr_flatfield ,  self.grid_row+3, 1, 1, 4)          
        self.grid.addWidget(self.cbx_ccdcorr_distortion,  self.grid_row+4, 1, 1, 4) 
        self.grid.addWidget(self.cbx_ccdcorr_parasitic ,  self.grid_row+5, 1, 1, 4) 

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.grid)
        self.vbox.addStretch(1)

        self.setLayout(self.vbox)

        self.connect(self.cbx_ccdcorr_blemish   , QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 
        self.connect(self.cbx_ccdcorr_flatfield , QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 
        self.connect(self.cbx_ccdcorr_distortion, QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 
        self.connect(self.cbx_ccdcorr_parasitic , QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit field'
        #self.rad_mask_none.setToolTip(msg_rad_mask)


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):
        self.                       setMinimumWidth(450)
        self.                       setStyleSheet (cp.styleBkgd)
        self.tit_ccdcorr_sets      .setStyleSheet (cp.styleTitle)
        self.cbx_ccdcorr_blemish   .setStyleSheet (cp.styleLabel)
        self.cbx_ccdcorr_flatfield .setStyleSheet (cp.styleLabel)
        self.cbx_ccdcorr_distortion.setStyleSheet (cp.styleLabel)
        self.cbx_ccdcorr_parasitic .setStyleSheet (cp.styleLabel)

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
        try    : del cp.guianasettingsright # GUICCDCorrectionSettings
        except : pass

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def on(self):
        logger.debug('on click - is not implemented yet', __name__)

    def onCBox(self):

        if self.cbx_ccdcorr_blemish     .hasFocus() :
            self.cbx = self.cbx_ccdcorr_blemish
            self.par = cp.ccdcorr_blemish
            self.tit = 'ccdcorr_blemish' 

        elif self.cbx_ccdcorr_flatfield .hasFocus() :
            self.cbx = self.cbx_ccdcorr_flatfield
            self.par = cp.ccdcorr_flatfield
            self.tit = 'ccdcorr_flatfield' 

        elif self.cbx_ccdcorr_distortion.hasFocus() :
            self.cbx = self.cbx_ccdcorr_distortion
            self.par = cp.ccdcorr_distortion
            self.tit = 'ccdcorr_distortion' 

        elif self.cbx_ccdcorr_parasitic .hasFocus() :
            self.cbx = self.cbx_ccdcorr_parasitic
            self.par = cp.ccdcorr_parasitic
            self.tit = 'ccdcorr_parasitic' 

        self.par.setValue( self.cbx.isChecked() )
        msg = 'onCBox - set status of ' + self.tit  + ': ' + str(self.par.value())
        logger.info(msg, __name__ )
    
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUICCDCorrectionSettings ()
    widget.show()
    app.exec_()

#-----------------------------
