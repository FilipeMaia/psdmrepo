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
        self.setGeometry(200, 400, 300, 250)
        self.setWindowTitle('Analysis Settings Right')
        self.setFrame()

        self.tit_ccdset         = QtGui.QLabel('CCD Settings:')
        self.tit_ccdset_pixsize = QtGui.QLabel('Pixel size [mm]')
        self.tit_ccdset_adcsatu = QtGui.QLabel('ADC saturation level [ADU]')
        self.tit_ccdset_aduphot = QtGui.QLabel('ADU per photon')
        self.tit_ccdset_ccdeff  = QtGui.QLabel('CCD efficiency')
        self.tit_ccdset_ccdgain = QtGui.QLabel('CCD gain')
        self.tit_mask_hot       = QtGui.QLabel('Hot pix mask thr. on RMS [ADU]:')

        self.edi_ccdset_pixsize = QtGui.QLineEdit( str( cp.ccdset_pixsize.value() ) )        
        self.edi_ccdset_adcsatu = QtGui.QLineEdit( str( cp.ccdset_adcsatu.value() ) )        
        self.edi_ccdset_aduphot = QtGui.QLineEdit( str( cp.ccdset_aduphot.value() ) )        
        self.edi_ccdset_ccdeff  = QtGui.QLineEdit( str( cp.ccdset_ccdeff .value() ) )        
        self.edi_ccdset_ccdgain = QtGui.QLineEdit( str( cp.ccdset_ccdgain.value() ) )        

        self.edi_mask_hot       = QtGui.QLineEdit( str( cp.mask_hot_thr.value() ) )        
        #self.cbx_mask_hot       = QtGui.QCheckBox('Use hot pix mask', self)
        self.cbx_mask_hot       = QtGui.QCheckBox('', self)
        self.cbx_mask_hot.setChecked( cp.mask_hot_is_used.value() )

        self.tit_orient = QtGui.QLabel('CCD orientation [deg]:')
        self.list_of_orient = ['0', '90', '180', '270'] 
        self.box_orient     = QtGui.QComboBox( self ) 
        self.box_orient.addItems(self.list_of_orient)
        self.box_orient.setCurrentIndex( self.list_of_orient.index(cp.ccd_orient.value()) )


        self.grid = QtGui.QGridLayout()

        self.grid_row = 0
        self.grid.addWidget(self.tit_ccdset,               self.grid_row+1, 0, 1, 5)
        self.grid.addWidget(self.tit_ccdset_pixsize,       self.grid_row+2, 1, 1, 3)
        self.grid.addWidget(self.tit_ccdset_adcsatu,       self.grid_row+3, 1, 1, 3)
        self.grid.addWidget(self.tit_ccdset_aduphot,       self.grid_row+4, 1, 1, 3)
        self.grid.addWidget(self.tit_ccdset_ccdeff ,       self.grid_row+5, 1, 1, 3)
        self.grid.addWidget(self.tit_ccdset_ccdgain,       self.grid_row+6, 1, 1, 3)

        self.grid.addWidget(self.edi_ccdset_pixsize,       self.grid_row+2, 4)
        self.grid.addWidget(self.edi_ccdset_adcsatu,       self.grid_row+3, 4)
        self.grid.addWidget(self.edi_ccdset_aduphot,       self.grid_row+4, 4)
        self.grid.addWidget(self.edi_ccdset_ccdeff ,       self.grid_row+5, 4)
        self.grid.addWidget(self.edi_ccdset_ccdgain,       self.grid_row+6, 4)

        self.grid.addWidget(self.tit_mask_hot,             self.grid_row+7, 1)
        self.grid.addWidget(self.edi_mask_hot,             self.grid_row+7, 4)
        self.grid.addWidget(self.cbx_mask_hot,             self.grid_row+7, 0)

        self.grid.addWidget(self.tit_orient,               self.grid_row+9, 1, 1, 3)
        self.grid.addWidget(self.box_orient,               self.grid_row+9, 4)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.grid)
        self.vbox.addStretch(1)

        self.setLayout(self.vbox)

        self.connect(self.edi_ccdset_pixsize, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ccdset_adcsatu, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ccdset_aduphot, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ccdset_ccdeff , QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ccdset_ccdgain, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_mask_hot,       QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.cbx_mask_hot,       QtCore.SIGNAL('stateChanged(int)'), self.on_cbx ) 
        self.connect(self.box_orient,         QtCore.SIGNAL('currentIndexChanged(int)'), self.on_box_orient )

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
        self.edi_ccdset_ccdgain.setToolTip(msg)
        self.tit_mask_hot.setToolTip('Hot pixel mask is generated \nat pedestals evaluation \nusing threshold on RMS '\
                                     + str(cp.mask_hot_thr.value()) + ' ADU')
        self.cbx_mask_hot.setToolTip('On/off hot pixel mask\nin the final analysis')
        self.edi_mask_hot.setToolTip('Threshold [ADU] on RMS for hot pixels')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):

        width = 60
        self.                   setMinimumWidth(300)
        self.                   setStyleSheet (cp.styleBkgd)
        self.tit_ccdset        .setStyleSheet (cp.styleTitle)
        self.tit_ccdset_pixsize.setStyleSheet (cp.styleLabel)
        self.tit_ccdset_adcsatu.setStyleSheet (cp.styleLabel)
        self.tit_ccdset_aduphot.setStyleSheet (cp.styleLabel)
        self.tit_ccdset_ccdeff .setStyleSheet (cp.styleLabel)
        self.tit_ccdset_ccdgain.setStyleSheet (cp.styleLabel)
        self.tit_mask_hot      .setStyleSheet (cp.styleLabel)
        self.cbx_mask_hot      .setStyleSheet (cp.styleLabel)

        self.edi_ccdset_pixsize.setStyleSheet(cp.styleEdit)
        self.edi_ccdset_adcsatu.setStyleSheet(cp.styleEdit)
        self.edi_ccdset_aduphot.setStyleSheet(cp.styleEdit)
        self.edi_ccdset_ccdeff .setStyleSheet(cp.styleEdit)
        self.edi_ccdset_ccdgain.setStyleSheet(cp.styleEdit)
        self.edi_mask_hot      .setStyleSheet(cp.styleEdit)
        
        self.edi_ccdset_pixsize.setFixedWidth(width)
        self.edi_ccdset_adcsatu.setFixedWidth(width)
        self.edi_ccdset_aduphot.setFixedWidth(width)
        self.edi_ccdset_ccdeff .setFixedWidth(width)
        self.edi_ccdset_ccdgain.setFixedWidth(width)
        self.edi_mask_hot      .setFixedWidth(width)
        
        self.edi_ccdset_pixsize.setAlignment(QtCore.Qt.AlignRight) 
        self.edi_ccdset_adcsatu.setAlignment(QtCore.Qt.AlignRight) 
        self.edi_ccdset_aduphot.setAlignment(QtCore.Qt.AlignRight) 
        self.edi_ccdset_ccdeff .setAlignment(QtCore.Qt.AlignRight) 
        self.edi_ccdset_ccdgain.setAlignment(QtCore.Qt.AlignRight) 
        self.edi_mask_hot      .setAlignment(QtCore.Qt.AlignRight) 
        
        self.tit_orient        .setStyleSheet(cp.styleLabel)
        self.tit_orient        .setAlignment (QtCore.Qt.AlignLeft)
        self.box_orient        .setStyleSheet(cp.styleButton)


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

        elif self.edi_ccdset_ccdgain.isModified() :            
            self.edi = self.edi_ccdset_ccdgain
            self.par = cp.ccdset_ccdgain
            self.tit = 'ccdset_ccdgain'

        elif self.edi_mask_hot.isModified() :            
            self.edi = self.edi_mask_hot
            self.par = cp.mask_hot_thr
            self.tit = 'mask_hot_thr'

        else : return # no-modification

        self.edi.setModified(False)
        self.par.setValue( self.edi.displayText() )        
        msg = 'onEdit - set value of ' + self.tit  + ': ' + str( self.par.value())
        logger.info(msg, __name__ )

    def on_cbx(self):
        #if self.cbx_dark.hasFocus() :
        par = cp.mask_hot_is_used
        par.setValue( self.cbx_mask_hot.isChecked() )
        msg = 'on_cbx - set status of parameter mask_hot_is_used: ' + str(par.value())
        logger.info(msg, __name__ )
        #self.setButtonState()

    def on_box_orient(self):
        orient_selected = self.box_orient.currentText()
        cp.ccd_orient.setValue( orient_selected ) 
        logger.info('on_box_orient - orient_selected: ' + orient_selected, __name__)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUICCDSettings ()
    widget.show()
    app.exec_()

#-----------------------------
