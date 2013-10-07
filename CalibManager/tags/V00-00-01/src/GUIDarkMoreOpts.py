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
from GUIFileBrowser         import *
from PlotImgSpe             import *
import CSPAD2x2Image        as     cspad2x2img
import CSPADImage           as     cspadimg

#---------------------
#  Class definition --
#---------------------
class GUIDarkMoreOpts ( QtGui.QWidget ) :
#class GUIDarkMoreOpts ( QtGui.QGroupBox ) :
    """GUI sets the source dark run number, validity range, and starts calibration of pedestals"""

    char_expand    = u' \u25BE' # down-head triangle
    dict_status = {True  : 'Created:', 
                   False : 'N/A     ' }

               
    def __init__ ( self, parent=None, run_number='0000' ) :

        QtGui.QWidget.__init__(self, parent)
        #QtGui.QGroupBox.__init__(self, 'More', parent)

        self.parent     = parent
        self.run_number = run_number
        self.det_name   = cp.det_name
        self.calib_dir  = cp.calib_dir

        self.setGeometry(100, 100, 600, 50)
        self.setWindowTitle('GUI Dark Run Go')
        #try : self.setWindowIcon(cp.icon_help)
        #except : pass
        self.setFrame()

        #self.lab_run  = QtGui.QLabel('Dark run')

        self.cbx_dark_more = QtGui.QCheckBox('More options')
        self.cbx_dark_more.setChecked( cp.dark_more_opts.value() )
 
        self.but_srcs = QtGui.QPushButton( 'Sources' )
        self.but_flst = QtGui.QPushButton( 'O/Files' )
        self.but_fbro = QtGui.QPushButton( 'Show files' )
        self.but_plot = QtGui.QPushButton( 'Plot' )
        self.but_show = QtGui.QPushButton( 'Show cmd' )

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.cbx_dark_more)
        self.hbox.addWidget(self.but_srcs)
        self.hbox.addWidget(self.but_flst)
        self.hbox.addWidget(self.but_fbro)
        self.hbox.addWidget(self.but_plot)
        self.hbox.addWidget(self.but_show)
        self.hbox.addStretch(1)     

        #self.cbx_dark_more.move(50,0)
        #self.hbox.move(50,30)

        self.vbox = QtGui.QVBoxLayout()
        #self.vbox.addWidget(self.cbx_dark_more)
        self.vbox.addLayout(self.hbox)
        self.vbox.addStretch(1)     
        self.setLayout(self.vbox)

        self.connect(self.cbx_dark_more  , QtCore.SIGNAL('stateChanged(int)'), self.on_cbx ) 
        self.connect( self.but_srcs, QtCore.SIGNAL('clicked()'), self.on_but_srcs )
        self.connect( self.but_flst, QtCore.SIGNAL('clicked()'), self.on_but_flst )
        self.connect( self.but_fbro, QtCore.SIGNAL('clicked()'), self.on_but_fbro )
        self.connect( self.but_plot, QtCore.SIGNAL('clicked()'), self.on_but_plot )
        self.connect( self.but_show, QtCore.SIGNAL('clicked()'), self.on_but_show )
   
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
        self.frame.setVisible(False)


    def setFieldsEnabled(self, is_enabled=True):

        logger.info('Set fields enabled: %s' %  is_enabled, __name__)

        #self.but_run  .setEnabled(is_enabled)
        #self.but_go   .setEnabled(is_enabled)
        #self.but_stop .setEnabled(is_enabled)

        self.setStyle()


    def setStyle(self):
        #self.setMinimumSize(600,70)
        #self.setMinimumSize(600,70)
        self.setFixedHeight(40)
        self.setStyleSheet (cp.styleBkgd)
        self.cbx_dark_more.setFixedHeight(30)
        
        width = 80

        self.but_srcs.setFixedWidth(width)
        self.but_flst.setFixedWidth(width)
        self.but_fbro.setFixedWidth(width)
        self.but_plot.setFixedWidth(width)
        self.but_show.setFixedWidth(width)

        #self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))
        self.setContentsMargins (QtCore.QMargins(-5,-5,-5,-5))
        #self.setContentsMargins (QtCore.QMargins(10,10,10,10))
        #self.setContentsMargins (QtCore.QMargins(0,5,0,0))

        self.but_srcs.setVisible( self.cbx_dark_more.isChecked() )
        self.but_flst.setVisible( self.cbx_dark_more.isChecked() )
        self.but_fbro.setVisible( self.cbx_dark_more.isChecked() )
        self.but_plot.setVisible( self.cbx_dark_more.isChecked() )
        self.but_show.setVisible( self.cbx_dark_more.isChecked() )


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



    def on_but_srcs(self) :
        self.exportLocalPars()

        #cp.blsp.parse_batch_log_peds_scan()
        cp.blsp.print_list_of_types_and_sources()



    def on_but_flst(self):
        self.exportLocalPars()

        logger.info('on_but_flst', __name__)
        list_of_files = self.get_list_of_files_peds()
        msg = 'File status for run %s:\n' % self.run_number
        for fname in list_of_files :

            exists     = os.path.exists(fname)
            msg += '%s  %s' % (fname.ljust(55), self.dict_status[exists].ljust(5))

            if exists :
                ctime_sec  = os.path.getctime(fname)
                ctime_str  = gu.get_local_time_str(ctime_sec, fmt='%Y-%m-%d %H:%M:%S')
                size_byte  = os.path.getsize(fname)
                file_owner = gu.get_path_owner(fname)
                #file_mode  = gu.get_path_mode(fname)
                msg += '  %s  %12d(Byte) %s\n' % (ctime_str, size_byte, file_owner)
            else :
                msg += '\n'
        logger.info(msg, __name__ )



    def get_list_of_files_peds(self) :
        list_of_fnames = fnm.get_list_of_files_peds() \
             + cp.blsp.get_list_of_files_for_all_sources(fnm.path_peds_ave()) \
             + cp.blsp.get_list_of_files_for_all_sources(fnm.path_peds_rms())
        list_of_fnames.append(fnm.path_hotpix_mask())
        return list_of_fnames



    def on_but_fbro(self):
        self.exportLocalPars()

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
        self.exportLocalPars()

        logger.debug('on_but_plot', __name__)
        try :
            cp.plotimgspe.close()
            try    : del cp.plotimgspe
            except : pass
        except :

            self.arr     = None
            self.img_arr = None

            msg = 'Plot image for %s' % self.det_name.value()
            logger.info(msg, __name__)

            list_of_fnames = cp.blsp.get_list_of_files_for_all_sources(fnm.path_peds_ave()) \
                           + cp.blsp.get_list_of_files_for_all_sources(fnm.path_peds_rms())

            #print 'list_of_fnames = ', list_of_fnames

            if list_of_fnames != [] :

                fname = list_of_fnames[0]
                if len(list_of_fnames) > 1 :
                    fname = gu.selectFromListInPopupMenu(list_of_fnames)

                msg = 'Selected file to plot: %s' % fname
                logger.info(msg, __name__)

                self.arr = gu.get_array_from_file( fname )
                #print self.arr.shape,'\n', self.arr.shape

            if self.det_name.value() == cp.list_of_dets[0] : # CSAPD, shape = (5920,388) 
                self.arr.shape = (32*185,388) 
                self.img_arr = cspadimg.get_cspad_raw_data_array_image(self.arr)

            elif self.det_name.value() == cp.list_of_dets[1] : # CSAPD2x2
                self.arr.shape = (185,388,2) 
                self.img_arr = cspad2x2img.get_cspad2x2_non_corrected_image_for_raw_data_array(self.arr)

            elif self.det_name.value() == cp.list_of_dets[2] : # Camera
                pass

            elif self.det_name.value() == cp.list_of_dets[3] : # Princeton
                self.img_arr = self.arr

            elif self.det_name.value() == cp.list_of_dets[4] : # pnCCD
                pass


            if self.img_arr == None :
                msg = 'self.img_arr == None'
                return
            #print arr.shape,'\n', arr.shape
            cp.plotimgspe = PlotImgSpe(None, self.img_arr, ofname=fnm.path_peds_aver_plot())
            #cp.plotimgspe = PlotImgSpe(None, self.img_arr, ifname=fnm.path_peds_ave(), ofname=fnm.path_peds_aver_plot())
            #cp.plotimgspe.setParent(self)
            cp.plotimgspe.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.plotimgspe.show()
            #but.setStyleSheet(cp.styleButtonGood)


    def get_gui_run(self):
        return self.parent.parent.gui_run


    def exportLocalPars(self):
        """run appropriate method from GUIDarkListItemRun.py"""
        self.get_gui_run().exportLocalPars()


    def on_but_show(self):
        """Prints the list of commands for deployment of calibration file(s)"""
        list_of_deploy_commands = self.get_gui_run().get_list_of_deploy_commands()
        msg = 'Deploy command(s):'
        for cmd in list_of_deploy_commands :
            msg += '\n' + cmd
        logger.info(msg, __name__)


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
