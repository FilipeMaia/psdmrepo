#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUICCDSettings...
#
#------------------------------------------------------------------------

"""GUI sets parameters for analysis (right pannel)"""

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
class GUICCDSettings ( QtGui.QWidget ) :
    """GUI sets parameters for analysis (right panel)"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Analysis Settings Right')
        self.setFrame()

        self.tit_ccdset         = QtGui.QLabel('CCD Settings:')
        self.tit_ccdset_pixsize = QtGui.QLabel('Pixel size [mm]')
        self.tit_ccdset_adcsatu = QtGui.QLabel('ADC saturation level [ADU]')
        self.tit_ccdset_aduphot = QtGui.QLabel('ADU per photon')
        self.tit_ccdset_ccdeff  = QtGui.QLabel('CCD efficiency')
        self.tit_ccdset_ccddain = QtGui.QLabel('CCD gain')

        self.edi_ccdset_pixsize = QtGui.QLineEdit( str( cp.ccdset_pixsize.value() ) )        
        self.edi_ccdset_adcsatu = QtGui.QLineEdit( str( cp.ccdset_adcsatu.value() ) )        
        self.edi_ccdset_aduphot = QtGui.QLineEdit( str( cp.ccdset_aduphot.value() ) )        
        self.edi_ccdset_ccdeff  = QtGui.QLineEdit( str( cp.ccdset_ccdeff .value() ) )        
        self.edi_ccdset_ccddain = QtGui.QLineEdit( str( cp.ccdset_ccddain.value() ) )        

        self.grid = QtGui.QGridLayout()

        self.grid_row = 0
        self.grid.addWidget(self.tit_ccdset,               self.grid_row+1, 0, 1, 5)
        self.grid.addWidget(self.tit_ccdset_pixsize,       self.grid_row+2, 1, 1, 3)
        self.grid.addWidget(self.tit_ccdset_adcsatu,       self.grid_row+3, 1, 1, 3)
        self.grid.addWidget(self.tit_ccdset_aduphot,       self.grid_row+4, 1, 1, 3)
        self.grid.addWidget(self.tit_ccdset_ccdeff ,       self.grid_row+5, 1, 1, 3)
        self.grid.addWidget(self.tit_ccdset_ccddain,       self.grid_row+6, 1, 1, 3)

        self.grid.addWidget(self.edi_ccdset_pixsize,       self.grid_row+2, 4)
        self.grid.addWidget(self.edi_ccdset_adcsatu,       self.grid_row+3, 4)
        self.grid.addWidget(self.edi_ccdset_aduphot,       self.grid_row+4, 4)
        self.grid.addWidget(self.edi_ccdset_ccdeff ,       self.grid_row+5, 4)
        self.grid.addWidget(self.edi_ccdset_ccddain,       self.grid_row+6, 4)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.grid)
        self.vbox.addStretch(1)

        self.setLayout(self.vbox)

        self.connect(self.edi_ccdset_pixsize, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ccdset_adcsatu, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ccdset_aduphot, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ccdset_ccdeff , QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ccdset_ccddain, QtCore.SIGNAL('editingFinished()'), self.onEdit )

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit field'
        self.tit_ccdset.setToolTip('Set CCD parameters')
        self.edi_ccdset_pixsize.setToolTip(msg)
        self.edi_ccdset_adcsatu.setToolTip(msg)
        self.edi_ccdset_aduphot.setToolTip(msg)
        self.edi_ccdset_ccdeff .setToolTip(msg)
        self.edi_ccdset_ccddain.setToolTip(msg)


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):

        width = 60
        self.                   setMinimumWidth(450)
        self.                   setStyleSheet (cp.styleBkgd)
        self.tit_ccdset        .setStyleSheet (cp.styleTitle)
        self.tit_ccdset_pixsize.setStyleSheet (cp.styleLabel)
        self.tit_ccdset_adcsatu.setStyleSheet (cp.styleLabel)
        self.tit_ccdset_aduphot.setStyleSheet (cp.styleLabel)
        self.tit_ccdset_ccdeff .setStyleSheet (cp.styleLabel)
        self.tit_ccdset_ccddain.setStyleSheet (cp.styleLabel)

        self.edi_ccdset_pixsize.setStyleSheet(cp.styleEdit)
        self.edi_ccdset_adcsatu.setStyleSheet(cp.styleEdit)
        self.edi_ccdset_aduphot.setStyleSheet(cp.styleEdit)
        self.edi_ccdset_ccdeff .setStyleSheet(cp.styleEdit)
        self.edi_ccdset_ccddain.setStyleSheet(cp.styleEdit)

        self.edi_ccdset_pixsize.setFixedWidth(width)
        self.edi_ccdset_adcsatu.setFixedWidth(width)
        self.edi_ccdset_aduphot.setFixedWidth(width)
        self.edi_ccdset_ccdeff .setFixedWidth(width)
        self.edi_ccdset_ccddain.setFixedWidth(width)

        self.edi_ccdset_pixsize.setAlignment(QtCore.Qt.AlignRight) 
        self.edi_ccdset_adcsatu.setAlignment(QtCore.Qt.AlignRight) 
        self.edi_ccdset_aduphot.setAlignment(QtCore.Qt.AlignRight) 
        self.edi_ccdset_ccdeff .setAlignment(QtCore.Qt.AlignRight) 
        self.edi_ccdset_ccddain.setAlignment(QtCore.Qt.AlignRight) 

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
        try    : del cp.guiccdsettings # GUICCDSettings
        except : pass

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def on(self):
        logger.debug('on click - is not implemented yet', __name__)

    def onEdit(self):

        if self.edi_ccdset_pixsize.isModified() :            
            self.edi = self.edi_ccdset_pixsize 
            self.par = cp.ccdset_pixsize
            self.tit = 'ccdset_pixsize'

        elif self.edi_ccdset_adcsatu.isModified() :            
            self.edi = self.edi_ccdset_adcsatu
            self.par = cp.ccdset_adcsatu
            self.tit = 'ccdset_adcsatu'

        elif self.edi_ccdset_aduphot.isModified() :            
            self.edi = self.edi_ccdset_aduphot
            self.par = cp.ccdset_aduphot
            self.tit = 'ccdset_aduphot'

        elif self.edi_ccdset_ccdeff.isModified() :            
            self.edi = self.edi_ccdset_ccdeff
            self.par = cp.ccdset_ccdeff
            self.tit = 'ccdset_ccdeff'

        elif self.edi_ccdset_ccddain.isModified() :            
            self.edi = self.edi_ccdset_ccddain
            self.par = cp.ccdset_ccddain
            self.tit = 'ccdset_ccddain'

        else : return # no-modification

        self.edi.setModified(False)
        self.par.setValue( self.edi.displayText() )        
        msg = 'onEdit - set value of ' + self.tit  + ': ' + str( self.par.value())
        logger.info(msg, __name__ )

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUICCDSettings ()
    widget.show()
    app.exec_()

#-----------------------------
