#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIAnaSettingsRight...
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
class GUIAnaSettingsRight ( QtGui.QWidget ) :
    """GUI sets parameters for analysis (right panel)"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Analysis Settings Right')
        self.setFrame()

        self.tit_ana_opts        = QtGui.QLabel('Dynamic Analysis Options:')
        self.tit_ana_opt1        = QtGui.QLabel('# of delays per multiple tau level:')
        self.tit_ana_opt2        = QtGui.QLabel('# of slice delays per multiple tau level:')
        self.edi_ana_opt1        = QtGui.QLineEdit( str( cp.ana_ndelays.value() ) )        
        self.edi_ana_opt2        = QtGui.QLineEdit( str( cp.ana_nslice_delays.value() ) )        
        self.edi_ana_opt3        = QtGui.QLineEdit( str( cp.ana_npix_to_smooth.value() ) )        
        self.cbx_ana_smooth_norm = QtGui.QCheckBox('use smoothed symmetric normalization,  Npix min:', self)
        self.cbx_ana_two_corfuns = QtGui.QCheckBox('Two time correlation function control', self)
        self.cbx_ana_spec_stab   = QtGui.QCheckBox('Check speckle stability', self)
        self.cbx_ana_smooth_norm.setChecked( cp.ana_smooth_norm.value() )
        self.cbx_ana_two_corfuns.setChecked( cp.ana_two_corfuns.value() )
        self.cbx_ana_spec_stab  .setChecked( cp.ana_spec_stab.value() )

        self.tit_lld      = QtGui.QLabel('Low Level Discrimination (LLD):')
        self.edi_lld_adu  = QtGui.QLineEdit( str( cp.lld_adu.value() ) )        
        self.edi_lld_rms  = QtGui.QLineEdit( str( cp.lld_rms.value() ) )        
        self.rad_lld_none = QtGui.QRadioButton('no LLD')
        self.rad_lld_adu  = QtGui.QRadioButton('ADU threshold:')
        self.rad_lld_rms  = QtGui.QRadioButton('dark RMS threshold:')
        self.rad_lld_grp  = QtGui.QButtonGroup()
        self.rad_lld_grp.addButton(self.rad_lld_none)
        self.rad_lld_grp.addButton(self.rad_lld_adu )
        self.rad_lld_grp.addButton(self.rad_lld_rms )
        self.list_lld_types = ['NONE', 'ADU', 'RMS']
        if   cp.lld_type.value() == self.list_lld_types[1] : self.rad_lld_adu .setChecked(True)
        elif cp.lld_type.value() == self.list_lld_types[2] : self.rad_lld_rms .setChecked(True)
        else                                               : self.rad_lld_none.setChecked(True)

        self.tit_res_sets        = QtGui.QLabel('Saving settings:')
        self.cbx_res_ascii_out   = QtGui.QCheckBox('ASCII output', self)
        self.cbx_res_fit1        = QtGui.QCheckBox('Perform Fit1', self)
        self.cbx_res_fit2        = QtGui.QCheckBox('Perform Fit2', self)
        self.cbx_res_fit_cust    = QtGui.QCheckBox('Perform Custom Fit', self)
        self.cbx_res_png_out     = QtGui.QCheckBox('Create PNG Files', self)
        self.cbx_res_ascii_out.setChecked( cp.res_ascii_out.value() )
        self.cbx_res_fit1     .setChecked( cp.res_fit1     .value() )
        self.cbx_res_fit2     .setChecked( cp.res_fit2     .value() )
        self.cbx_res_fit_cust .setChecked( cp.res_fit_cust .value() )
        self.cbx_res_png_out  .setChecked( cp.res_png_out  .value() )

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.tit_ana_opts,             0, 0, 1, 8)
        self.grid.addWidget(self.tit_ana_opt1,             1, 1, 1, 8)
        self.grid.addWidget(self.edi_ana_opt1,             1, 9)
        self.grid.addWidget(self.tit_ana_opt2,             2, 1, 1, 8)
        self.grid.addWidget(self.edi_ana_opt2,             2, 9)
        self.grid.addWidget(self.cbx_ana_smooth_norm,      3, 1, 1, 8)
        self.grid.addWidget(self.edi_ana_opt3,             3, 9)
        self.grid.addWidget(self.cbx_ana_two_corfuns,      4, 1, 1, 7)
        self.grid.addWidget(self.cbx_ana_spec_stab,        5, 1, 1, 7)

        self.grid.addWidget(self.tit_lld,                  6, 0, 1, 8)
        self.grid.addWidget(self.rad_lld_none,             7, 1, 1, 3)
        self.grid.addWidget(self.rad_lld_adu,              8, 1, 1, 3)
        self.grid.addWidget(self.edi_lld_adu,              8, 4)
        self.grid.addWidget(self.rad_lld_rms,              9, 1, 1, 3)
        self.grid.addWidget(self.edi_lld_rms,              9, 4)

        self.grid.addWidget(self.tit_res_sets,            10, 0, 1, 8)     
        self.grid.addWidget(self.cbx_res_fit1,            11, 1, 1, 4)     
        self.grid.addWidget(self.cbx_res_fit2,            12, 1, 1, 4)          
        self.grid.addWidget(self.cbx_res_fit_cust,        13, 1, 1, 4) 
        self.grid.addWidget(self.cbx_res_ascii_out,       11, 6, 1, 3)
        self.grid.addWidget(self.cbx_res_png_out,         12, 6, 1, 3) 

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.grid)
        self.vbox.addStretch(1)

        self.setLayout(self.vbox)

        self.connect(self.rad_lld_none, QtCore.SIGNAL('clicked()'), self.onRadioLLD )
        self.connect(self.rad_lld_adu,  QtCore.SIGNAL('clicked()'), self.onRadioLLD )
        self.connect(self.rad_lld_rms,  QtCore.SIGNAL('clicked()'), self.onRadioLLD )

        self.connect(self.edi_ana_opt1, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ana_opt2, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_ana_opt3, QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_lld_adu , QtCore.SIGNAL('editingFinished()'), self.onEdit )
        self.connect(self.edi_lld_rms , QtCore.SIGNAL('editingFinished()'), self.onEdit )

        self.connect(self.cbx_ana_smooth_norm   , QtCore.SIGNAL('stateChanged(int)'), self.onCBox )
        self.connect(self.cbx_ana_two_corfuns   , QtCore.SIGNAL('stateChanged(int)'), self.onCBox) 
        self.connect(self.cbx_ana_spec_stab     , QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 

        self.connect(self.cbx_res_ascii_out     , QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 
        self.connect(self.cbx_res_fit1          , QtCore.SIGNAL('stateChanged(int)'), self.onCBox )
        self.connect(self.cbx_res_fit2          , QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 
        self.connect(self.cbx_res_fit_cust      , QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 
        self.connect(self.cbx_res_png_out       , QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit field'
        self.tit_ana_opts.setToolTip('This section allows to monitor/modify\nthe beam zero parameters\nin transmission mode')
        #self.edi_kin_top_row    .setToolTip( msg )
        #self.edi_kin_slice_first.setToolTip( msg )
        #self.edi_kin_slice_last .setToolTip( msg )

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):

        width = 60
        self.                    setMinimumWidth(450)
        self.                    setStyleSheet (cp.styleBkgd)

        self.tit_ana_opts       .setStyleSheet (cp.styleTitle)
        self.tit_ana_opt1       .setStyleSheet (cp.styleLabel)
        self.tit_ana_opt2       .setStyleSheet (cp.styleLabel)
        self.cbx_ana_smooth_norm.setStyleSheet (cp.styleLabel)

        self.tit_res_sets       .setStyleSheet (cp.styleTitle)     
        self.cbx_res_ascii_out  .setStyleSheet (cp.styleLabel)
        self.cbx_res_fit1       .setStyleSheet (cp.styleLabel)
        self.cbx_res_fit2       .setStyleSheet (cp.styleLabel)
        self.cbx_res_fit_cust   .setStyleSheet (cp.styleLabel)
        self.cbx_res_png_out    .setStyleSheet (cp.styleLabel)
        self.cbx_ana_two_corfuns.setStyleSheet (cp.styleLabel)
        self.cbx_ana_spec_stab  .setStyleSheet (cp.styleLabel)

        self.edi_ana_opt1       .setStyleSheet(cp.styleEdit)
        self.edi_ana_opt2       .setStyleSheet(cp.styleEdit)
        self.edi_ana_opt3       .setStyleSheet(cp.styleEdit) 
        self.edi_lld_adu        .setStyleSheet(cp.styleEdit) 
        self.edi_lld_rms        .setStyleSheet(cp.styleEdit) 

        self.edi_ana_opt1       .setFixedWidth(width)
        self.edi_ana_opt2       .setFixedWidth(width)
        self.edi_ana_opt3       .setFixedWidth(width) 
        self.edi_lld_adu        .setFixedWidth(width)
        self.edi_lld_rms        .setFixedWidth(width)

        self.edi_ana_opt1       .setAlignment(QtCore.Qt.AlignRight)
        self.edi_ana_opt2       .setAlignment(QtCore.Qt.AlignRight)
        self.edi_ana_opt3       .setAlignment(QtCore.Qt.AlignRight) 
        self.edi_lld_adu        .setAlignment(QtCore.Qt.AlignRight) 
        self.edi_lld_rms        .setAlignment(QtCore.Qt.AlignRight) 

        self.tit_lld            .setStyleSheet (cp.styleTitle)
        self.rad_lld_none       .setStyleSheet (cp.styleLabel)
        self.rad_lld_adu        .setStyleSheet (cp.styleLabel)
        self.rad_lld_rms        .setStyleSheet (cp.styleLabel)

#        self.box_kin_mode       .setStyleSheet(cp.styleBox) 


    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        logger.debug('resizeEvent')
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        logger.debug('moveEvent')
#        cp.posGUIMain = (self.pos().x(),self.pos().y())

    def closeEvent(self, event):
        logger.debug('closeEvent')
        try: # try to delete self object in the cp
            del cp.guianasettingsright # GUIAnaSettingsRight
        except AttributeError:
            pass # silently ignore

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

        elif self.cbx_res_ascii_out  .hasFocus() :
            self.cbx = self.cbx_res_ascii_out
            self.par = cp.res_ascii_out
            self.tit = 'res_ascii_out' 

        elif self.cbx_res_fit1       .hasFocus() :
            self.cbx = self.cbx_res_fit1
            self.par = cp.res_fit1
            self.tit = 'res_fit1' 

        elif self.cbx_res_fit2       .hasFocus() :
            self.cbx = self.cbx_res_fit2
            self.par = cp.res_fit2
            self.tit = 'res_fit2' 

        elif self.cbx_res_fit_cust   .hasFocus() :
            self.cbx = self.cbx_res_fit_cust
            self.par = cp.res_fit_cust
            self.tit = 'res_fit_cust' 

        elif self.cbx_res_png_out    .hasFocus() :
            self.cbx = self.cbx_res_png_out
            self.par = cp.res_png_out
            self.tit = 'res_png_out' 

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

        elif self.edi_lld_adu.isModified() :            
            self.edi = self.edi_lld_adu 
            self.par = cp.lld_adu
            self.tit = 'lld_adu'

        elif self.edi_lld_rms.isModified() :            
            self.edi = self.edi_lld_rms
            self.par = cp.lld_rms
            self.tit = 'lld_rms'

        else : return # no-modification

        self.edi.setModified(False)
        self.par.setValue( self.edi.displayText() )        
        msg = 'onEdit - set value of ' + self.tit  + ': ' + str( self.par.value())
        logger.info(msg, __name__ )


    def onRadioLLD(self): 
        if self.rad_lld_none.isChecked() : cp.lld_type.setValue( self.list_lld_types[0] )
        if self.rad_lld_adu .isChecked() : cp.lld_type.setValue( self.list_lld_types[1] )
        if self.rad_lld_rms .isChecked() : cp.lld_type.setValue( self.list_lld_types[2] )
        logger.info('onRadioLLD - selected Low Level Discrimination type: ' + cp.lld_type.value(), __name__ )

#-----------------------------

#    def on_edi_kin_slice_last(self):
#        cp.kin_slice_last.setValue( float(self.edi_kin_slice_last.displayText()) )
#        logger.info('Set kin_slice_last =' + str(cp.kin_slice_last.value()) )#

#    def on_box_kin_mode(self):
#        self.mode_name = self.box_kin_mode.currentText()
#        cp.kin_mode.setValue( self.mode_name )
#        logger.info(' ---> selected kinematic mode: ' + self.mode_name )

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIAnaSettingsRight ()
    widget.show()
    app.exec_()

#-----------------------------
