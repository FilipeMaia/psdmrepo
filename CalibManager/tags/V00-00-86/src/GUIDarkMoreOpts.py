#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDarkMoreOpts ...
#
#------------------------------------------------------------------------

#------------------------------
#  Module's version from SVN --
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

from   ConfigParametersForApp import cp
from   Logger                 import logger
import GlobalUtils            as     gu
from   FileNameManager        import fnm
from   GUIFileBrowser         import *
from   PlotImgSpe             import *
import FileDeployer           as     fdmets

#import PyCSPadImage.CSPAD2x2Image as     cspad2x2img
#import PyCSPadImage.CSPADImage    as     cspadimg

#---------------------
#  Class definition --
#---------------------
class GUIDarkMoreOpts ( QtGui.QWidget ) :
#class GUIDarkMoreOpts ( QtGui.QGroupBox ) :
    """GUI sets the source dark run number, validity range, and starts calibration of pedestals"""

    char_expand    = cp.char_expand
    #char_expand    = u' \u25BC' # down-head triangle
    #char_expand    = ' V' # down-head triangle
    dict_status = {True  : 'Created:', 
                   False : 'N/A     '}
               
    def __init__ ( self, parent=None, run_number='0000' ) :

        QtGui.QWidget.__init__(self, parent)
        #QtGui.QGroupBox.__init__(self, 'More', parent)

        self.parent     = parent
        self.run_number = run_number
        self.instr_name = cp.instr_name
        self.exp_name   = cp.exp_name
        self.det_name   = cp.det_name
        self.calib_dir  = cp.calib_dir

        self.setGeometry(100, 100, 600, 35)
        self.setWindowTitle('GUI Dark Run Go')
        #try : self.setWindowIcon(cp.icon_help)
        #except : pass
        self.setFrame()

        #self.lab_run  = QtGui.QLabel('Dark run')

        self.cbx_dark_more = QtGui.QCheckBox('More options')
        self.cbx_dark_more.setChecked( cp.dark_more_opts.value() )
 
 
        self.lab_show = QtGui.QLabel('Show:')
        self.but_srcs = QtGui.QPushButton( 'Sources DB' )
        self.but_sxtc = QtGui.QPushButton( 'Sources XTC' )
        self.but_flst = QtGui.QPushButton( 'O/Files' )
        self.but_fxtc = QtGui.QPushButton( 'xtc Files' )
        self.but_view = QtGui.QPushButton( 'View' )
        self.but_plot = QtGui.QPushButton( 'Plot' )
        self.but_show = QtGui.QPushButton( 'Show cmd' )

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.lab_show)
        self.hbox.addWidget(self.but_fxtc)
        self.hbox.addWidget(self.but_srcs)
        self.hbox.addWidget(self.but_sxtc)
        self.hbox.addWidget(self.but_flst)
        self.hbox.addWidget(self.but_show)
        self.hbox.addSpacing(50) 
        self.hbox.addWidget(self.cbx_dark_more)
        self.hbox.addWidget(self.but_view)
        self.hbox.addWidget(self.but_plot)
        self.hbox.addStretch(1)     

        #self.wbox = QtGui.QWidget()
        #self.wbox.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))
        #self.wbox.setFixedWidth(650)
        #self.wbox.setLayout(self.hbox)
        
        #self.obox = QtGui.QHBoxLayout()
        #self.obox.addWidget(self.hbox)
        #self.obox.addStretch(1)     

        #self.cbx_dark_more.move(50,0)
        #self.hbox.move(50,30)

        self.vbox = QtGui.QVBoxLayout()
        #self.vbox.addWidget(self.wbox)
        self.vbox.addLayout(self.hbox)
        self.vbox.addStretch(1)     
        self.setLayout(self.vbox)

        self.connect( self.cbx_dark_more  , QtCore.SIGNAL('stateChanged(int)'), self.on_cbx ) 
        self.connect( self.but_srcs, QtCore.SIGNAL('clicked()'), self.on_but_srcs )
        self.connect( self.but_sxtc, QtCore.SIGNAL('clicked()'), self.on_but_sxtc )
        self.connect( self.but_flst, QtCore.SIGNAL('clicked()'), self.on_but_flst )
        self.connect( self.but_fxtc, QtCore.SIGNAL('clicked()'), self.on_but_fxtc )
        self.connect( self.but_view, QtCore.SIGNAL('clicked()'), self.on_but_view )
        self.connect( self.but_plot, QtCore.SIGNAL('clicked()'), self.on_but_plot )
        self.connect( self.but_show, QtCore.SIGNAL('clicked()'), self.on_but_show )
   
        self.showToolTips()

        self.setStyle()

        cp.guidarkmoreopts = self


    def showToolTips(self):
        pass
        self.but_srcs.setToolTip('Show data sources from DB')
        self.but_sxtc.setToolTip('Show data types and sources from xtc file scan')
        self.but_flst.setToolTip('Show status list of output files')
        self.but_fxtc.setToolTip('Show input xtc files for run %s' % self.run_number)
        self.but_view.setToolTip('Start text file browser in separate window')
        self.but_plot.setToolTip('Start image plot browser in separate window')
        self.but_show.setToolTip('Show commands for file deployment')
        self.cbx_dark_more.setToolTip('Add more buttons with other options')
        #self..setToolTip('')

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
        self.setFixedHeight(35)
        self.setStyleSheet (cp.styleBkgd)
        #self.cbx_dark_more.setFixedHeight(30)
        self.lab_show.setStyleSheet(cp.styleLabel)

        width = 80

        self.but_fxtc.setFixedWidth(60)
        self.but_srcs.setFixedWidth(90)
        self.but_sxtc.setFixedWidth(90)
        self.but_flst.setFixedWidth(60)
        self.but_show.setFixedWidth(80)
        self.but_view.setFixedWidth(90)
        self.but_plot.setFixedWidth(60)

        #self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))
        self.setContentsMargins (QtCore.QMargins(-5,-5,-5,-5))
        #self.setContentsMargins (QtCore.QMargins(10,10,10,10))
        #self.setContentsMargins (QtCore.QMargins(0,5,0,0))

        #self.but_srcs.setVisible( self.cbx_dark_more.isChecked() )
        #self.but_sxtc.setVisible( self.cbx_dark_more.isChecked() )
        #self.but_flst.setVisible( self.cbx_dark_more.isChecked() )
        #self.but_fxtc.setVisible( self.cbx_dark_more.isChecked() )
        #self.but_show.setVisible( self.cbx_dark_more.isChecked() )
        self.but_view.setVisible( self.cbx_dark_more.isChecked() )
        self.but_plot.setVisible( self.cbx_dark_more.isChecked() )

        self.cbx_dark_more.setVisible( False )



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
        logger.debug('onClose', __name__)
        self.close()


    def on_but_srcs(self) :
        """Print sources from RegDB
        """
        self.exportLocalPars()
        txt = '\n' + 50*'-' + '\nSources from DB:\n' \
            + cp.blsp.txt_of_sources_in_run()
        logger.info(txt, __name__)


    def on_but_sxtc(self) :
        """Print sources from XTC scan log
        """
        self.exportLocalPars()
        txt = '\n' + 50*'-' + '\nData Types and Sources from xtc scan of the\n' \
            + cp.blsp.txt_list_of_types_and_sources()
        
        logger.info(txt, __name__)


    def on_but_flst(self):
        self.exportLocalPars()

        logger.debug('on_but_flst', __name__)
        msg = '\n' + 50*'-' + '\nFile status in %s for run %s:\n' % (fnm.path_dir_work(), self.run_number)
        list_of_files = self.get_list_of_files_peds()
        for fname in list_of_files :

            exists     = os.path.exists(fname)
            msg += '%s  %s' % (os.path.basename(fname).ljust(75), self.dict_status[exists].ljust(5))

            if exists :
                ctime_sec  = os.path.getctime(fname)
                ctime_str  = gu.get_local_time_str(ctime_sec, fmt='%Y-%m-%d %H:%M:%S')
                size_byte  = os.path.getsize(fname)
                file_owner = gu.get_path_owner(fname)
                file_mode  = gu.get_path_mode(fname)
                msg += '  %s  %12d(Byte) %s  %s\n' % (ctime_str, size_byte, file_owner, str(oct(file_mode)) )
            else :
                msg += '\n'
        logger.info(msg, __name__ )


    def on_but_fxtc(self):
        self.exportLocalPars()

        logger.debug('on_but_fxtc', __name__)
        #list_of_files = self.get_list_of_files_peds()
        dir_xtc = fnm.path_to_xtc_dir()
        list_of_files = gu.get_list_of_files_in_dir_for_part_fname(dir_xtc, pattern='-r'+self.run_number)
        msg = '\n' + 50*'-' + '\nXTC files in %s for run %s:\n' % (dir_xtc, self.run_number)
        for fname in list_of_files :

            exists     = os.path.exists(fname)
            msg += '%s  %s' % (os.path.basename(fname).ljust(22), self.dict_status[exists].ljust(5))

            if exists :
                ctime_sec  = os.path.getctime(fname)
                ctime_str  = gu.get_local_time_str(ctime_sec, fmt='%Y-%m-%d %H:%M:%S')
                size_byte  = os.path.getsize(fname)
                file_owner = gu.get_path_owner(fname)
                file_mode  = gu.get_path_mode(fname)
                msg += '  %s  %12d(Byte)  %s  %s\n' % (ctime_str, size_byte, file_owner, str(oct(file_mode)) )
            else :
                msg += '\n'
        logger.info(msg, __name__ )


    def get_list_of_files_peds(self) :
        return fnm.get_list_of_files_peds() \
             + self.get_list_of_files_peds_for_plot()


    def get_list_of_files_peds_for_plot(self) :
        lst_of_srcs = cp.blsp.list_of_sources_for_selected_detectors()
        return gu.get_list_of_files_for_list_of_insets(fnm.path_peds_ave(),    lst_of_srcs) \
             + gu.get_list_of_files_for_list_of_insets(fnm.path_peds_rms(),    lst_of_srcs) \
             + gu.get_list_of_files_for_list_of_insets(fnm.path_hotpix_mask(), lst_of_srcs)


    def on_but_view(self):
        self.exportLocalPars()

        logger.debug('on_but_view', __name__)
        try    :
            cp.guifilebrowser.close()
            #self.but_view.setStyleSheet(cp.styleButtonBad)
        except :
            #self.but_view.setStyleSheet(cp.styleButtonGood)
            
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
            list_of_fnames = self.get_list_of_files_peds_for_plot()
            #print 'list_of_fnames = ', list_of_fnames

            if list_of_fnames == [] :
                logger.warning('List of files is empty... There is nothing to plot...', __name__)
                return

            fname = list_of_fnames[0]
            if len(list_of_fnames) > 1 :
                fname = gu.selectFromListInPopupMenu(list_of_fnames)

            if fname is None or fname == '' : return

            msg = 'Selected file to plot: %s' % fname
            logger.info(msg, __name__)
            #print msg

            #self.img_arr = self.get_image_array_from_file(fname)
            #if self.img_arr is None :
            #    return

            #print arr.shape,'\n', arr.shape
            tit = 'Plot for %s' % os.path.basename(fname)
            
            cp.plotimgspe = PlotImgSpe(None, ifname=fname, ofname=fnm.path_peds_ave_plot(), title=tit, is_expanded=False)
            #cp.plotimgspe = PlotImgSpe(None, self.img_arr, ofname=fnm.path_peds_ave_plot(), title=tit)
            #cp.plotimgspe = PlotImgSpe(None, self.img_arr, ifname=fnm.path_peds_ave(), ofname=fnm.path_peds_ave_plot())
            #cp.plotimgspe.setParent(self)
            cp.plotimgspe.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.plotimgspe.show()
            #but.setStyleSheet(cp.styleButtonGood)




#    def get_image_array_from_file(self, fname) :
#        """DEPRICATED: Recognizes detector by file name and returns image array or None
#        """
#
#        arr     = None
#        img_arr = None
#
#        arr = gu.get_array_from_file( fname )
#        if arr is None :
#            logger.warning('Array is not retreaved from file...', __name__)
#            return None
#
#        #print 'arr:\n', arr
#        msg = 'arr.shape: %s' % str(arr.shape)
#        logger.info(msg, __name__)
# 
#        fname_lower = fname.lower()
#
#        #print 'fname_lower =', fname_lower
#        #print 'list_of_dets_lower =', cp.list_of_dets_lower
#
#        #self.list_of_dets   = ['CSPAD', 'CSPAD2x2', 'Princeton', 'pnCCD', 'Tm6740', 'Opal2000', 'Opal4000', 'Acqiris']
#
#        if cp.list_of_dets_lower[0]+'.' in fname_lower : # CSAPD, shape = (5920,388) 
#            arr.shape = (32*185,388) 
#            img_arr = cspadimg.get_cspad_raw_data_array_image(arr)
#
#        elif cp.list_of_dets_lower[1] in fname_lower : # CSAPD2x2
#            arr.shape = (185,388,2) 
#            img_arr = cspad2x2img.get_cspad2x2_non_corrected_image_for_raw_data_array(arr)
#
#        elif cp.list_of_dets_lower[2] in fname_lower : # Princeton
#            img_arr = arr
#
#        elif cp.list_of_dets_lower[3] in fname_lower : # pnCCD
#            img_arr = arr
#
#        elif cp.list_of_dets_lower[4] in fname_lower : # Camera
#            img_arr = arr
#
#        elif cp.list_of_dets_lower[5] in fname_lower : # Camera
#            img_arr = arr
#
#        elif cp.list_of_dets_lower[6] in fname_lower : # Camera
#            img_arr = arr
#
#        elif cp.list_of_dets_lower[7] in fname_lower : # Acqiris
#            img_arr = arr
#
#        else :
#            msg = 'Detector is not recognized in the file name: %s' %  fname
#            logger.warning(msg, __name__)
#            return None
#
#        return img_arr



    def get_gui_run(self):
        return self.parent.parent.gui_run


    def exportLocalPars(self):
        """run appropriate method from GUIDarkListItemRun.py"""
        self.get_gui_run().exportLocalPars()


    def on_but_show(self):
        """Prints the list of commands for deployment of calibration file(s)"""
        #str_run_number = '%04d' % self.run_number
        list_of_deploy_commands, list_of_sources = \
          fdmets.get_list_of_deploy_commands_and_sources_dark(self.run_number, self.get_gui_run().strRunRange())
        msg = '\n' + 50*'-' + '\nTentative deployment commands:\n' + '\n'.join(list_of_deploy_commands)
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
        logger.debug(msg, __name__ )

        self.setStyle()


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    w = GUIDarkMoreOpts()
    w.setFieldsEnabled(True)
    w.show()
    app.exec_()

#-----------------------------
