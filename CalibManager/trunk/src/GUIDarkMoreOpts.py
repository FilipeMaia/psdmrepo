#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDarkMoreOpts ...
#
#------------------------------------------------------------------------

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

from ConfigParametersForApp import cp
from Logger                 import logger
import GlobalUtils          as     gu
from FileNameManager        import fnm
from BatchJobPedestals      import bjpeds
from GUIFileBrowser         import *
from PlotImgSpe             import *

#---------------------
#  Class definition --
#---------------------
class GUIDarkMoreOpts ( QtGui.QWidget ) :
#class GUIDarkMoreOpts ( QtGui.QGroupBox ) :
    """GUI sets the source dark run number, validity range, and starts calibration of pedestals"""

    char_expand    = u' \u25BE' # down-head triangle

    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)
        #QtGui.QGroupBox.__init__(self, 'More', parent)

        self.setGeometry(100, 100, 600, 70)
        self.setWindowTitle('GUI Dark Run Go')
        #try : self.setWindowIcon(cp.icon_help)
        #except : pass
        self.setFrame()

        #self.lab_run  = QtGui.QLabel('Dark run')

        self.cbx_dark_more = QtGui.QCheckBox('More options')
        self.cbx_dark_more.setChecked( cp.dark_more_opts.value() )
 
        self.but_stop = QtGui.QPushButton( 'Stop' )
        self.but_fbro = QtGui.QPushButton( 'File browser' )
        self.but_plot = QtGui.QPushButton( 'Plot' )
        self.but_depl = QtGui.QPushButton( 'Deploy' )

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.but_stop)
        self.hbox.addWidget(self.but_fbro)
        self.hbox.addWidget(self.but_plot)
        self.hbox.addWidget(self.but_depl)
        self.hbox.addStretch(1)     

        #self.cbx_dark_more.move(50,0)
        #self.hbox.move(50,30)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.cbx_dark_more)
        self.vbox.addLayout(self.hbox)
        self.vbox.addStretch(1)     
        self.setLayout(self.vbox)

        self.connect(self.cbx_dark_more  , QtCore.SIGNAL('stateChanged(int)'), self.on_cbx ) 
        self.connect( self.but_stop, QtCore.SIGNAL('clicked()'), self.on_but_stop )
        self.connect( self.but_fbro, QtCore.SIGNAL('clicked()'), self.on_but_fbro )
        self.connect( self.but_plot, QtCore.SIGNAL('clicked()'), self.on_but_plot )
        self.connect( self.but_depl, QtCore.SIGNAL('clicked()'), self.on_but_depl )
   
        self.showToolTips()

        self.setStyle()

        cp.guidarkmoreopts = self


    def showToolTips(self):
        pass
        #self.but_run .setToolTip('Select the run for calibration.')
        #self.but_go  .setToolTip('Begin data processing for calibration.')
        #self.but_stop.setToolTip('Emergency stop data processing.')
        #self.edi_from.setToolTip('Type in the run number \nas a lower limit of the validity range.')
        #self.edi_to  .setToolTip('Type in the run number or "end"\nas an upper limit of the validity range.')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(1)
        self.frame.setMidLineWidth(0)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def setFieldsEnabled(self, is_enabled=True):

        logger.info('Set fields enabled: %s' %  is_enabled, __name__)

        #self.but_run  .setEnabled(is_enabled)
        #self.but_go   .setEnabled(is_enabled)
        #self.but_stop .setEnabled(is_enabled)

        self.setStyle()


    def setStyle(self):
        #self.setMinimumSize(600,70)
        #self.setMinimumSize(600,70)
        self.setFixedHeight(90)
        self.setStyleSheet (cp.styleBkgd)

        width = 100

        self.but_stop.setFixedWidth(width)
        self.but_fbro.setFixedWidth(width)
        self.but_plot.setFixedWidth(width)
        self.but_depl.setFixedWidth(width)

        #self.lab_from.setStyleSheet(cp.styleLabel)
        #self.lab_to  .setStyleSheet(cp.styleLabel)
        #self.lab_run .setStyleSheet(cp.styleLabel)

        #self.edi_from .setAlignment (QtCore.Qt.AlignRight)
        #self.edi_to   .setAlignment (QtCore.Qt.AlignRight)

        #if self.str_run_number.value() == 'None' and self.but_run.isEnabled() :
        #    self.but_run.setStyleSheet(cp.styleButtonBad)
        #else :
        #    self.but_run.setStyleSheet(cp.styleDefault)

        #if self.edi_from.isReadOnly() : self.edi_from.setStyleSheet (cp.styleEditInfo)
        #else                          : self.edi_from.setStyleSheet (cp.styleEdit)

        #if self.edi_to.isReadOnly()   : self.edi_to.setStyleSheet (cp.styleEditInfo)
        #else                          : self.edi_to.setStyleSheet (cp.styleEdit)

        #self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))
        #self.setContentsMargins (QtCore.QMargins(10,10,10,10))
        #self.setContentsMargins (QtCore.QMargins(0,5,0,0))

        self.but_stop.setVisible( self.cbx_dark_more.isChecked() )
        self.but_fbro.setVisible( self.cbx_dark_more.isChecked() )
        self.but_plot.setVisible( self.cbx_dark_more.isChecked() )
        self.but_depl.setVisible( self.cbx_dark_more.isChecked() )

        self.frame.setVisible( self.cbx_dark_more.isChecked() )


    def setParent(self,parent) :
        self.parent = parent


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())
        #self.box_txt.setGeometry(self.contentsRect())

        
    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        #self.saveLogTotalInFile() # It will be saved at closing of GUIMain

        #try    : cp.guimain.butLogger.setStyleSheet(cp.styleButtonBad)
        #except : pass

        #self.box_txt.close()

        #try    : del cp.gui... # GUIDarkMoreOpts
        #except : pass

        try    : cp.guifilebrowser.close()
        except : pass

        try    : cp.plotimgspe.close()
        except : pass


    def onClose(self):
        logger.info('onClose', __name__)
        self.close()


    def on_but_stop(self):
        logger.info('on_but_stop', __name__)
        bjpeds.stop_auto_processing()


    def get_list_of_files_peds(self) :
        list_of_fnames = fnm.get_list_of_files_peds() \
             + cp.blsp.get_list_of_files_for_all_sources(fnm.path_peds_ave()) \
             + cp.blsp.get_list_of_files_for_all_sources(fnm.path_peds_rms())
        list_of_fnames.append(fnm.path_hotpix_mask())
        return list_of_fnames

        
    def on_but_fbro(self):
        logger.info('on_but_fbro', __name__)
        try    :
            cp.guifilebrowser.close()
            #self.but_fbro.setStyleSheet(cp.styleButtonBad)
        except :
            #self.but_fbro.setStyleSheet(cp.styleButtonGood)
            
            cp.guifilebrowser = GUIFileBrowser(None, self.get_list_of_files_peds(), fnm.path_peds_scan_psana_cfg())
            cp.guifilebrowser.move(self.pos().__add__(QtCore.QPoint(880,40))) # open window with offset w.r.t. parent
            cp.guifilebrowser.show()


    def on_but_plot(self):
        logger.info('on_but_plot', __name__)
        try :
            cp.plotimgspe.close()
            try    : del cp.plotimgspe
            except : pass
        except :
            arr = gu.get_array_from_file( fnm.path_peds_ave() )
            if arr == None : return
            #print arr.shape,'\n', arr
            cp.plotimgspe = PlotImgSpe(None, arr, ifname=fnm.path_peds_ave(), ofname=fnm.path_peds_aver_plot())
            #cp.plotimgspe.setParent(self)
            cp.plotimgspe.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.plotimgspe.show()
            #but.setStyleSheet(cp.styleButtonGood)

    def on_but_depl(self):
        logger.info('on_but_depl', __name__)

   
    def setStatusMessage(self):
        if cp.guistatus is None : return
        msg = 'New status msg from GUIDarkMoreOpts'
        cp.guistatus.setStatusMessage(msg)

    def on_cbx(self):
        #if self.cbx.hasFocus() :
        par = cp.dark_more_opts
        cbx = self.cbx_dark_more
        tit = cbx.text()

        par.setValue( cbx.isChecked() )
        msg = 'check box ' + tit  + ' is set to: ' + str( par.value())
        logger.info(msg, __name__ )

        self.setStyle()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    w = GUIDarkMoreOpts()
    w.setFieldsEnabled(True)
    w.show()
    app.exec_()

#-----------------------------
