#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIAnaSettingsOptions...
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
class GUIAnaSettingsOptions ( QtGui.QWidget ) :
    """GUI sets options for analysis"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(20, 40, 370, 30)
        self.setWindowTitle('Analysis Options')
        self.setFrame()

        self.tit_ana_opts        = QtGui.QLabel('Dynamic Analysis Options:')
        self.tit_ana_opt1        = QtGui.QLabel('# of delays per multiple tau level:')
        self.tit_ana_opt2        = QtGui.QLabel('# of slice delays per multiple tau level:')
        self.edi_ana_opt1        = QtGui.QLineEdit( str( cp.ana_ndelays.value() ) )        
        self.edi_ana_opt2        = QtGui.QLineEdit( str( cp.ana_nslice_delays.value() ) )        
        self.edi_ana_opt3        = QtGui.QLineEdit( str( cp.ana_npix_to_smooth.value() ) )        
        self.cbx_ana_smooth_norm = QtGui.QCheckBox('use smoothed sym. norm., Npix min:', self)
        self.cbx_ana_two_corfuns = QtGui.QCheckBox('Two time correlation function control', self)
        self.cbx_ana_spec_stab   = QtGui.QCheckBox('Check speckle stability', self)
        self.cbx_ana_smooth_norm.setChecked( cp.ana_smooth_norm.value() )
        self.cbx_ana_two_corfuns.setChecked( cp.ana_two_corfuns.value() )
        self.cbx_ana_spec_stab  .setChecked( cp.ana_spec_stab.value() )

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.tit_ana_opts,             0, 0, 1, 8)
        self.grid.addWidget(self.tit_ana_opt1,             1, 1, 1, 8)
        self.grid.addWidget(self.edi_ana_opt1,             1, 8)
        self.grid.addWidget(self.tit_ana_opt2,             2, 1, 1, 7)
        self.grid.addWidget(self.edi_ana_opt2,             2, 8)
        self.grid.addWidget(self.cbx_ana_smooth_norm,      3, 1, 1, 7)
        self.grid.addWidget(self.edi_ana_opt3,             3, 8)
        self.grid.addWidget(self.cbx_ana_two_corfuns,      4, 1, 1, 8)
        self.grid.addWidget(self.cbx_ana_spec_stab,        5, 1, 1, 8)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.grid)
        self.vbox.addStretch(1)

        self.setLayout(self.vbox)

        self.connect(self.edi_ana_opt1, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ana_opt2, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ana_opt3, QtCore.SIGNAL('editingFinished()'), self.onEdit )

        self.connect(self.cbx_ana_smooth_norm, QtCore.SIGNAL('stateChanged(int)'), self.onCBox )
        self.connect(self.cbx_ana_two_corfuns, QtCore.SIGNAL('stateChanged(int)'), self.onCBox) 
        self.connect(self.cbx_ana_spec_stab  , QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit field'
        self.tit_ana_opts.setToolTip('Change analysis options.')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):

        width = 60
        self.                    setMinimumWidth(370)
        self.                    setStyleSheet (cp.styleBkgd)

        self.tit_ana_opts       .setStyleSheet (cp.styleTitle)
        self.tit_ana_opt1       .setStyleSheet (cp.styleLabel)
        self.tit_ana_opt2       .setStyleSheet (cp.styleLabel)
        self.cbx_ana_smooth_norm.setStyleSheet (cp.styleLabel)

        self.cbx_ana_two_corfuns.setStyleSheet (cp.styleLabel)
        self.cbx_ana_spec_stab  .setStyleSheet (cp.styleLabel)

        self.edi_ana_opt1       .setStyleSheet(cp.styleEdit)
        self.edi_ana_opt2       .setStyleSheet(cp.styleEdit)
        self.edi_ana_opt3       .setStyleSheet(cp.styleEdit) 

        self.edi_ana_opt1       .setMinimumWidth(width)
        self.edi_ana_opt2       .setMinimumWidth(width)
        self.edi_ana_opt3       .setMinimumWidth(width) 

        self.edi_ana_opt1       .setAlignment(QtCore.Qt.AlignRight)
        self.edi_ana_opt2       .setAlignment(QtCore.Qt.AlignRight)
        self.edi_ana_opt3       .setAlignment(QtCore.Qt.AlignRight) 

    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        #logger.debug('resizeEvent')
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent')
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent')
        try    : del cp.guianasettingsoptions # GUIAnaSettingsOptions
        except : pass

    def onClose(self):
        logger.debug('onClose')
        self.close()

    def on(self):
        logger.debug('on click - is not implemented yet')

    def onCBox(self):
        if  self.cbx_ana_smooth_norm.hasFocus() :
            self.cbx = self.cbx_ana_smooth_norm
            self.par = cp.ana_smooth_norm
            self.tit = 'ana_smooth_norm'
 
        elif self.cbx_ana_two_corfuns.hasFocus() :
            self.cbx = self.cbx_ana_two_corfuns
            self.par = cp.ana_two_corfuns
            self.tit = 'ana_two_corfuns' 

        elif self.cbx_ana_spec_stab  .hasFocus() :
            self.cbx = self.cbx_ana_spec_stab
            self.par = cp.ana_spec_stab
            self.tit = 'ana_spec_stab' 

        self.par.setValue( self.cbx.isChecked() )
        msg = 'onCBox - set status of ' + self.tit  + ': ' + str( self.par.value())
        logger.info(msg, __name__ )
    

    def onEdit(self):

        if self.edi_ana_opt1.isModified() :            
            self.edi = self.edi_ana_opt1
            self.par = cp.ana_ndelays
            self.tit = 'ana_ndelays'

        elif self.edi_ana_opt2.isModified() :            
            self.edi = self.edi_ana_opt2
            self.par = cp.ana_nslice_delays
            self.tit = 'ana_nslice_delays'

        elif self.edi_ana_opt3.isModified() :            
            self.edi = self.edi_ana_opt3
            self.par = cp.ana_npix_to_smooth
            self.tit = 'ana_npix_to_smooth'

        else : return # no-modification

        self.edi.setModified(False)
        self.par.setValue( self.edi.displayText() )        
        msg = 'onEdit - set value of ' + self.tit  + ': ' + str( self.par.value())
        logger.info(msg, __name__ )

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIAnaSettingsOptions()
    widget.show()
    app.exec_()

#-----------------------------
