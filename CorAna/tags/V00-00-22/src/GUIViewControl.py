#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIViewControl...
#
#------------------------------------------------------------------------

"""GUI View Control"""

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
from ViewResults            import *
from PlotImgSpe             import *
from PlotG2                 import *
from PlotGraph              import *
from RecordsFromFiles       import rff

#---------------------
#  Class definition --
#---------------------
class GUIViewControl ( QtGui.QWidget ) :
    """GUI View Control"""

    def __init__ ( self, parent=None, fname=None, title='' ) :
        #super(GUIViewControl, self).__init__()
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 150)
        self.setWindowTitle('Load Results')
        self.setFrame()

        self.setFileName(fname)

        self.vr  = ViewResults(cp.res_fname.value())
        self.list_of_tau = self.vr.get_list_of_tau_from_file(fnm.path_cora_merge_tau())
        #print 'self.list_of_tau =', self.list_of_tau
        self.g_ind   = 0
        self.g_title = title
        self.tau_ind = 0
        self.g_show_tau = False

        self.initCorArray()        

        self.makeLayout()        

        self.showToolTips()
        self.setStyle()
        self.onSlider()

        self.setButtonState()

        #self.overlay = Overlay(self,'Load Results')
                
    #-------------------
    #  Public methods --
    #-------------------

    def makeLayout(self):

        self.tit               = QtGui.QLabel('View Control')
        self.tit_geom          = QtGui.QLabel('Geom. maps:')
        self.tit_part          = QtGui.QLabel('Partitions:')
        self.tit_data          = QtGui.QLabel(u'Raw data (\u03C4):')  # tau = u"\u03C4"
        self.tit_calc          = QtGui.QLabel(u'Calc.(\u03C4):')
        self.tit_mask          = QtGui.QLabel('Masks:')
        self.tit_save          = QtGui.QLabel('Save:')
        self.tit_plot          = QtGui.QLabel('Plots:')
        self.edi               = QtGui.QLineEdit( os.path.basename(cp.res_fname.value()) )        
        self.but               = QtGui.QPushButton('File')
        self.but_close         = QtGui.QPushButton('Close')
        self.but_Ip            = QtGui.QPushButton('<Ip>')
        self.but_If            = QtGui.QPushButton('<If>')
        self.but_I2            = QtGui.QPushButton('<Ip x If>')
        self.but_g2raw         = QtGui.QPushButton('g2 raw')
        self.but_X             = QtGui.QPushButton('X')
        self.but_Y             = QtGui.QPushButton('Y')
        self.but_R             = QtGui.QPushButton('R')
        self.but_P             = QtGui.QPushButton('Phi')
        self.but_Q             = QtGui.QPushButton('q')
        self.but_P_st          = QtGui.QPushButton('Phi stat')
        self.but_Q_st          = QtGui.QPushButton('q stat')
        self.but_QP_st         = QtGui.QPushButton('q-Phi stat')
        self.but_P_dy          = QtGui.QPushButton('Phi dyna')
        self.but_Q_dy          = QtGui.QPushButton('q dyna')
        self.but_QP_dy         = QtGui.QPushButton('q-Phi dyna')
        self.but_1oIp          = QtGui.QPushButton('1/<Ip> stat')
        self.but_1oIf          = QtGui.QPushButton('1/<If> stat')
        self.but_g2map         = QtGui.QPushButton('g2 map')
        self.but_g2dy          = QtGui.QPushButton('g2 dyna')
        self.but_g2tau         = QtGui.QPushButton('g2 vs tau')
        self.but_g2tau_gr      = QtGui.QPushButton('g2(tau)')
        self.but_mask_img_lims = QtGui.QPushButton('Image lims')
        self.but_mask_blemish  = QtGui.QPushButton('Blemish')
        self.but_mask_hotpix   = QtGui.QPushButton('Hot pixels')
        self.but_mask_satpix   = QtGui.QPushButton('Satt. pixs')
        self.but_mask_roi      = QtGui.QPushButton('ROI')
        self.but_mask_total    = QtGui.QPushButton('Total')
        self.but_print_res     = QtGui.QPushButton('g2(<q>,tau)')
        self.but_data_ave      = QtGui.QPushButton('<Img data>')
        self.but_intens_t      = QtGui.QPushButton('I(q-st,t)-2D')
        self.but_intens_gr1    = QtGui.QPushButton('I(q-st)-gr')
        self.but_intens_gr     = QtGui.QPushButton('I(q-st,t)-gr')

        self.cbx_more = QtGui.QCheckBox('Show more', self)
        self.cbx_more.setChecked( cp.vc_cbx_show_more.value() )



        self.sli = QtGui.QSlider(QtCore.Qt.Horizontal, self)        
        self.sli.setValue(0)
        self.sli.setRange(0, self.list_of_tau.shape[0]-1)
        self.sli.setTickInterval(1)
        self.edi_tau = QtGui.QLineEdit('tau(ind)')

        self.grid = QtGui.QGridLayout()
        self.grid_row = 1
        self.grid.addWidget(self.tit,              self.grid_row,   0, 1, 5)
        self.grid.addWidget(self.cbx_more,         self.grid_row,   6, 1, 3)

        self.grid.addWidget(self.but,              self.grid_row+1, 0)
        self.grid.addWidget(self.edi,              self.grid_row+1, 1, 1, 9)
        self.grid.addWidget(self.edi_tau,          self.grid_row+2, 0)
        self.grid.addWidget(self.sli,              self.grid_row+2, 1, 1, 9)

        self.grid.addWidget(self.tit_data,         self.grid_row+3, 0)
        self.grid.addWidget(self.but_Ip,           self.grid_row+3, 1)
        self.grid.addWidget(self.but_If,           self.grid_row+3, 2)
        self.grid.addWidget(self.but_I2,           self.grid_row+3, 3)
        self.grid.addWidget(self.but_g2raw,        self.grid_row+3, 4)
        self.grid.addWidget(self.but_close,        self.grid_row+3, 6)

        self.grid.addWidget(self.tit_geom,         self.grid_row+4, 0)
        self.grid.addWidget(self.but_P,            self.grid_row+4, 1)
        self.grid.addWidget(self.but_Q,            self.grid_row+4, 2)
        self.grid.addWidget(self.but_X,            self.grid_row+4, 3)
        self.grid.addWidget(self.but_Y,            self.grid_row+4, 4)
        self.grid.addWidget(self.but_R,            self.grid_row+4, 5)

        self.grid.addWidget(self.tit_part,         self.grid_row+5, 0)
        self.grid.addWidget(self.but_P_st,         self.grid_row+5, 1)
        self.grid.addWidget(self.but_Q_st,         self.grid_row+5, 2)
        self.grid.addWidget(self.but_QP_st,        self.grid_row+5, 3)
        self.grid.addWidget(self.but_P_dy,         self.grid_row+5, 4)
        self.grid.addWidget(self.but_Q_dy,         self.grid_row+5, 5)
        self.grid.addWidget(self.but_QP_dy,        self.grid_row+5, 6)

        self.grid.addWidget(self.tit_mask,         self.grid_row+6, 0)
        self.grid.addWidget(self.but_mask_total,   self.grid_row+6, 1)
        self.grid.addWidget(self.but_mask_img_lims,self.grid_row+6, 2)
        self.grid.addWidget(self.but_mask_blemish, self.grid_row+6, 3)
        self.grid.addWidget(self.but_mask_hotpix,  self.grid_row+6, 4)
        self.grid.addWidget(self.but_mask_satpix,  self.grid_row+6, 5)
        self.grid.addWidget(self.but_mask_roi,     self.grid_row+6, 6)

        self.grid.addWidget(self.tit_calc,         self.grid_row+7, 0)
        self.grid.addWidget(self.but_g2tau,        self.grid_row+7, 1)
        self.grid.addWidget(self.but_g2tau_gr,     self.grid_row+7, 2)
        self.grid.addWidget(self.but_g2map,        self.grid_row+7, 3)
        self.grid.addWidget(self.but_g2dy,         self.grid_row+7, 4)
        self.grid.addWidget(self.but_1oIp,         self.grid_row+7, 5)
        self.grid.addWidget(self.but_1oIf,         self.grid_row+7, 6)

        self.grid.addWidget(self.tit_save,         self.grid_row+8, 0)
        self.grid.addWidget(self.but_print_res,    self.grid_row+8, 1)

        self.grid.addWidget(self.tit_plot,         self.grid_row+9, 0)
        self.grid.addWidget(self.but_data_ave,     self.grid_row+9, 1)
        self.grid.addWidget(self.but_intens_gr1,   self.grid_row+9, 2)
        self.grid.addWidget(self.but_intens_gr,    self.grid_row+9, 3)
        self.grid.addWidget(self.but_intens_t,     self.grid_row+9, 4)

        self.grid_row += 8

        #self.connect(self.edi,         QtCore.SIGNAL('editingFinished()'),        self.onEdit )
        self.connect(self.but,              QtCore.SIGNAL('clicked()'),         self.onBut     )
        self.connect(self.but_close,        QtCore.SIGNAL('clicked()'),         self.onButClose)
        self.connect(self.but_Ip,           QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_If,           QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_I2,           QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_g2raw,        QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_X,            QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_Y,            QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_R,            QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_P,            QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_Q,            QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_P_st,         QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_Q_st,         QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_QP_st,        QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_P_dy,         QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_Q_dy,         QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_QP_dy,        QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_1oIp,         QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_1oIf,         QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_g2map,        QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_g2dy ,        QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_g2tau,        QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_g2tau_gr,     QtCore.SIGNAL('clicked()'),         self.on_but_g2tau_gr )
        self.connect(self.but_mask_img_lims,QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_mask_blemish, QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_mask_hotpix,  QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_mask_satpix,  QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_mask_roi,     QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_mask_total,   QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.sli,              QtCore.SIGNAL('valueChanged(int)'), self.onSlider  )
        self.connect(self.sli,              QtCore.SIGNAL('sliderReleased()'),  self.onSliderReleased )
        self.connect(self.cbx_more,         QtCore.SIGNAL('stateChanged(int)'), self.on_cbx ) 
        self.connect(self.but_print_res,    QtCore.SIGNAL('clicked()'),         self.onSave ) 
        self.connect(self.but_data_ave,     QtCore.SIGNAL('clicked()'),         self.onButView ) 
        self.connect(self.but_intens_t,     QtCore.SIGNAL('clicked()'),         self.onButView ) 
        self.connect(self.but_intens_gr1,   QtCore.SIGNAL('clicked()'),         self.onButIntens ) 
        self.connect(self.but_intens_gr,    QtCore.SIGNAL('clicked()'),         self.onButIntens ) 
  
        self.setLayout(self.grid)


    def setFileName(self, fname=None) :
        if fname is None : pass
        else : cp.res_fname.setValue(fname)


    def showToolTips(self):
        msg = 'Use this GUI to see results'
        self.setToolTip(msg)
        self.edi.setToolTip('Click on "File"\nto select the file')
        self.but.setToolTip('Click on this button\nand select the file')
        self.cbx_more.setToolTip('Show more or less buttons')
 

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def setStyle(self):
        self.setMinimumWidth(400)
        self.setStyleSheet(cp.styleBkgd)

        width = 80
        self.but.setFixedWidth(width)
        self.tit.setStyleSheet(cp.styleTitle)
        self.edi.setStyleSheet(cp.styleEditInfo) # cp.styleEditInfo
        self.but.setStyleSheet(cp.styleButton) 
        self.edi.setAlignment (QtCore.Qt.AlignLeft)
        self.edi.setReadOnly  (True)  
        self.edi_tau.setFixedWidth(width)
        self.edi_tau.setStyleSheet(cp.styleEditInfo) # cp.styleEditInfo
        self.edi_tau.setReadOnly  (True)
        self.edi_tau.setAlignment (QtCore.Qt.AlignCenter)

        self.tit_mask.setStyleSheet(cp.styleLabel)
        self.tit_geom.setStyleSheet(cp.styleLabel)
        self.tit_part.setStyleSheet(cp.styleLabel)
        self.tit_data.setStyleSheet(cp.styleLabel)
        self.tit_calc.setStyleSheet(cp.styleLabel)
        self.tit_save.setStyleSheet(cp.styleLabel)
        self.tit_plot.setStyleSheet(cp.styleLabel)

        self.tit_mask.setAlignment (QtCore.Qt.AlignCenter)
        self.tit_geom.setAlignment (QtCore.Qt.AlignCenter)
        self.tit_part.setAlignment (QtCore.Qt.AlignCenter)
        self.tit_data.setAlignment (QtCore.Qt.AlignCenter)
        self.tit_calc.setAlignment (QtCore.Qt.AlignCenter)
        self.tit_save.setAlignment (QtCore.Qt.AlignCenter)
        self.tit_plot.setAlignment (QtCore.Qt.AlignCenter)
        
        self.but_close        .setStyleSheet(cp.styleButtonBad)
        self.but_Ip           .setStyleSheet(cp.styleButton)
        self.but_If           .setStyleSheet(cp.styleButton)
        self.but_I2           .setStyleSheet(cp.styleButton)
        self.but_g2raw        .setStyleSheet(cp.styleButton)
        self.but_X            .setStyleSheet(cp.styleButton)
        self.but_Y            .setStyleSheet(cp.styleButton)
        self.but_R            .setStyleSheet(cp.styleButton)
        self.but_P            .setStyleSheet(cp.styleButton)
        self.but_Q            .setStyleSheet(cp.styleButton)
        self.but_P_st         .setStyleSheet(cp.styleButton)
        self.but_Q_st         .setStyleSheet(cp.styleButton)
        self.but_QP_st        .setStyleSheet(cp.styleButton)
        self.but_P_dy         .setStyleSheet(cp.styleButton)
        self.but_Q_dy         .setStyleSheet(cp.styleButton)
        self.but_QP_dy        .setStyleSheet(cp.styleButton)
        self.but_1oIp         .setStyleSheet(cp.styleButton)
        self.but_1oIf         .setStyleSheet(cp.styleButton)
        self.but_g2map        .setStyleSheet(cp.styleButton)
        self.but_g2dy         .setStyleSheet(cp.styleButton)
        self.but_g2tau        .setStyleSheet(cp.styleButton)
        self.but_g2tau_gr     .setStyleSheet(cp.styleButton)
        self.but_mask_img_lims.setStyleSheet(cp.styleButton)
        self.but_mask_blemish .setStyleSheet(cp.styleButton)
        self.but_mask_hotpix  .setStyleSheet(cp.styleButton)
        self.but_mask_satpix  .setStyleSheet(cp.styleButton)
        self.but_mask_roi     .setStyleSheet(cp.styleButton)
        self.but_mask_total   .setStyleSheet(cp.styleButton)
        self.but_print_res    .setStyleSheet(cp.styleButton)
        self.but_data_ave     .setStyleSheet(cp.styleButton)
        self.but_intens_t     .setStyleSheet(cp.styleButton)
        self.but_intens_gr    .setStyleSheet(cp.styleButton)
        self.but_intens_gr1   .setStyleSheet(cp.styleButton)

        self.cbx_more         .setStyleSheet (cp.styleLabel)
 

    def resizeEvent(self, e):
        self.frame.setGeometry(self.rect())
        e.accept()


    def moveEvent(self, e):
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        try    : del cp.guiviewcontrol # GUIViewControl
        except : pass # silently ignore


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def onEdit(self):
        logger.debug('onEdit', __name__)
        edi = self.edi
        par = cp.res_fname
        par.setValue( str(edi.displayText()) )
        logger.info('Set parameter = ' + str( par.value()), __name__ )

        
    def onBut(self):
        logger.debug('onBut - xelect file', __name__)
        but = self.but
        edi = self.edi
        par = cp.res_fname
        #dir = cp.dir_results.value() # is not used

        path0 = par.value()
        path  = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',path0) )
        dname, fname = os.path.split(path)

        if dname == '' or fname == '' :
            logger.warning('Input directiry name or file name is empty... keep file name unchanged...', __name__)
            return

        edi.setText (os.path.basename(path))
        par.setValue(path)
        logger.info('selected the file name: ' + str(par.value()), __name__ )
        self.initCorArray()        


    def initCorArray(self) :
        #print 'initCorArray().set_file_name():'
        self.vr.set_file_name()
        

    def setImgArray(self):
        t0 = gu.get_time_sec()
        if   self.g_ind == 0  : self.arr2d = self.vr.get_Ip_for_itau(self.tau_ind)
        elif self.g_ind == 1  : self.arr2d = self.vr.get_If_for_itau(self.tau_ind)
        elif self.g_ind == 2  : self.arr2d = self.vr.get_I2_for_itau(self.tau_ind)
        elif self.g_ind == 3  : self.arr2d = self.vr.get_g2_raw_for_itau(self.tau_ind)
        elif self.g_ind == 4  : self.arr2d = self.vr.get_x_map()
        elif self.g_ind == 5  : self.arr2d = self.vr.get_y_map()
        elif self.g_ind == 6  : self.arr2d = self.vr.get_r_map()
        elif self.g_ind == 7  : self.arr2d = self.vr.get_phi_map()
        elif self.g_ind == 8  : self.arr2d = self.vr.get_q_map()
        elif self.g_ind == 9  : self.arr2d = self.vr.get_phi_map_for_stat_bins()
        elif self.g_ind == 10 : self.arr2d = self.vr.get_and_save_map_for_stat_q_bins() # self.vr.get_q_map_for_stat_bins()
        elif self.g_ind == 11 : self.arr2d = self.vr.get_phi_map_for_dyna_bins()
        elif self.g_ind == 12 : self.arr2d = self.vr.get_q_map_for_dyna_bins()
        elif self.g_ind == 13 : self.arr2d = self.vr.get_q_phi_map_for_stat_bins()
        elif self.g_ind == 14 : self.arr2d = self.vr.get_q_phi_map_for_dyna_bins()
        elif self.g_ind == 15 : self.arr2d = self.vr.get_1oIp_map_for_stat_bins_itau(self.tau_ind)
        elif self.g_ind == 16 : self.arr2d = self.vr.get_1oIf_map_for_stat_bins_itau(self.tau_ind)
        elif self.g_ind == 17 : self.arr2d = self.vr.get_g2_map_for_itau(self.tau_ind)
        elif self.g_ind == 18 : self.arr2d = self.vr.get_g2_map_for_dyna_bins_itau(self.tau_ind)
        elif self.g_ind == 19 : self.arr2d = self.vr.get_g2_vs_itau_arr()
        elif self.g_ind == 20 : self.arr2d =     rff.get_data_ave_array()
        #elif self.g_ind == 21 : self.arr2d = self.vr.get_intens_stat_q_bins_vs_itau_arr()
        elif self.g_ind == 22 : self.arr2d =     rff.get_intens_stat_q_bins_arr()
        elif self.g_ind == 30 : self.arr2d = self.vr.get_mask_image_limits()
        elif self.g_ind == 31 : self.arr2d = self.vr.get_mask_blemish()
        elif self.g_ind == 32 : self.arr2d = self.vr.get_mask_hotpix()
        elif self.g_ind == 33 : self.arr2d = self.vr.get_mask_satpix()
        elif self.g_ind == 34 : self.arr2d = self.vr.get_mask_roi()
        elif self.g_ind == 35 : self.arr2d = self.vr.get_mask_total()


        else :
            logger.warning('Request for non-implemented plot ...', __name__)

        if self.arr2d is None :
            logger.warning('Requested array is not found (None).'
                           '\nDrawing command is terminated.'
                           '\nCheck if the file is available.', __name__)
            return False

        else :
            msg = 'Consumed time to get map: %11.6f sec' % (gu.get_time_sec()-t0)
            logger.info(msg, __name__)
            logger.info('arr2d.shape: ' + str(self.arr2d.shape) , __name__)
            #print 'arr2d:\n', self.arr2d 
            return True


    def selectedOption(self, ind, title, show_tau=False):
        logger.info(title + 'is selected', __name__)
        self.g_ind      = ind
        self.g_title    = title
        self.g_show_tau = show_tau


    def getTitle(self):      
        if self.g_show_tau : return self.g_title + self.stringTau()
        else               : return self.g_title


    def onButView(self):
        logger.info('onButView', __name__)
        if   self.but_Ip           .hasFocus() : self.selectedOption(  0, '<Ip> map, ',   show_tau=True)
        elif self.but_If           .hasFocus() : self.selectedOption(  1, '<If> map, ',   show_tau=True)
        elif self.but_I2           .hasFocus() : self.selectedOption(  2, '<Ip x If>, ',  show_tau=True)
        elif self.but_g2raw        .hasFocus() : self.selectedOption(  3, 'g2 raw map, ', show_tau=True)
        elif self.but_X            .hasFocus() : self.selectedOption(  4, 'X map')
        elif self.but_Y            .hasFocus() : self.selectedOption(  5, 'Y map')
        elif self.but_R            .hasFocus() : self.selectedOption(  6, 'R map')
        elif self.but_P            .hasFocus() : self.selectedOption(  7, 'Phi map')
        elif self.but_Q            .hasFocus() : self.selectedOption(  8, 'q map' )
        elif self.but_P_st         .hasFocus() : self.selectedOption(  9, 'Phi map for static bins')
        elif self.but_Q_st         .hasFocus() : self.selectedOption( 10, 'q map for static bins')
        elif self.but_P_dy         .hasFocus() : self.selectedOption( 11, 'Phi map for dynamic bins')
        elif self.but_Q_dy         .hasFocus() : self.selectedOption( 12, 'q map for dynamic bins')
        elif self.but_QP_st        .hasFocus() : self.selectedOption( 13, 'q-Phi map for static bins')
        elif self.but_QP_dy        .hasFocus() : self.selectedOption( 14, 'q-Phi map for dynamic bins')
        elif self.but_1oIp         .hasFocus() : self.selectedOption( 15, '1/<Ip> norm. map for static bins, ', show_tau=True)
        elif self.but_1oIf         .hasFocus() : self.selectedOption( 16, '1/<If> norm. map for static bins, ', show_tau=True)
        elif self.but_g2map        .hasFocus() : self.selectedOption( 17, 'g2 map, ',                           show_tau=True)
        elif self.but_g2dy         .hasFocus() : self.selectedOption( 18, 'g2 map for dynamic bins, ',          show_tau=True)
        elif self.but_g2tau        .hasFocus() : self.selectedOption( 19, 'g2 vs itau')
        #elif self.but_g2tau_gr     .hasFocus() : self.selectedOption( 20, 'g2(tau) for dynamic bins')
        #elif self.but_intens_gr    .hasFocus() : self.selectedOption( 21, 'I(itau, q-static)')
        elif self.but_data_ave     .hasFocus() : self.selectedOption( 20, '<Img-data>')
        elif self.but_intens_t     .hasFocus() : self.selectedOption( 22, 'I(t, q-static)')

        elif self.but_mask_img_lims.hasFocus() : self.selectedOption( 30, 'Mask image limits')
        elif self.but_mask_blemish .hasFocus() : self.selectedOption( 31, 'Mask blemish')
        elif self.but_mask_hotpix  .hasFocus() : self.selectedOption( 32, 'Mask hot pixels')
        elif self.but_mask_satpix  .hasFocus() : self.selectedOption( 33, 'Mask satturated pixels')
        elif self.but_mask_roi     .hasFocus() : self.selectedOption( 34, 'Mask for good regions')
        elif self.but_mask_total   .hasFocus() : self.selectedOption( 35, 'Mask total')
        else :
            logger.warning('Request for non-implemented button ...', __name__)

        self.drawPlot()
  

    def drawPlot(self):
        try :
            self.redrawPlotResetLimits()
            #print 'Redraw plot'
        except :
            #self.setImgArray()
            if self.but_g2tau.hasFocus() \
            or self.but_intens_t.hasFocus():
                cp.plotimgspe_g = PlotImgSpe(None, self.arr2d, title=self.getTitle(), orient=90, y_is_flip=True)
            else :
                cp.plotimgspe_g = PlotImgSpe(None, self.arr2d, title=self.getTitle(), \
                                             orient=int(cp.ccd_orient.value()), y_is_flip=cp.y_is_flip.value()) 
            cp.plotimgspe_g.move(self.parentWidget().parentWidget().pos().__add__(QtCore.QPoint(850,20)))
            cp.plotimgspe_g.show()


    def redrawPlotResetLimits(self):
        if not self.setImgArray() : return
        if self.but_g2tau.hasFocus() \
        or self.but_intens_t.hasFocus() :
            cp.plotimgspe_g.set_image_array_new(self.arr2d, self.getTitle(), orient=90, y_is_flip=True)
        else :
            cp.plotimgspe_g.set_image_array_new(self.arr2d, self.getTitle(), orient=int(cp.ccd_orient.value()), y_is_flip=cp.y_is_flip.value())


    def redrawPlot(self):
        if not self.setImgArray() : return
        cp.plotimgspe_g.set_image_array(self.arr2d, self.getTitle())


    def onSave(self):
        logger.debug('onSave', __name__)
        if self.but_print_res   .hasFocus() : self.vr.print_table_of_results()

            
    def onButClose(self):
        logger.info('onButClose', __name__)
        try    : cp.plotimgspe_g.close()
        except : pass
        try    : del cp.plotimgspe_g
        except : pass
        cp.plotimgspe_g = None

        try    : cp.plot_g2.close()
        except : pass
        try    : del cp.plot_g2
        except : pass



    def on_but_g2tau_gr(self):  
        #self.arr2d = self.vr.get_random_img()
        #self.list_of_tau
        q_average_for_dyna = self.vr.get_q_average_for_dyna_bins_trim_overflow()
        self.arr2d = self.vr.get_g2_vs_itau_arr()
        arr_tau = self.list_of_tau

        #print 'arr2d.shape', self.arr2d.shape
        #print 'q_ave.shape', q_average_for_dyna.shape
        #print 'len(self.list_of_tau)', len(self.list_of_tau)

        self.arr_g2 = [self.arr2d, self.list_of_tau, q_average_for_dyna]

        try :
            cp.plot_g2.close()
            try    : del cp.plot_g2
            except : pass
        except :
            cp.plot_g2 = PlotG2(None, self.arr_g2, ofname='./fig_g2.png', title='G2 TITLE IS NOT IMPLEMENTED') 
            cp.plot_g2.move(self.parentWidget().parentWidget().pos().__add__(QtCore.QPoint(870,20)))
            cp.plot_g2.show()

            #cp.plotg2_is_on = False
        

    def onButIntens(self):

        try :
            cp.plot_gr.close()
            try    : del cp.plot_gr
            except : pass
        except :

            arr_x  = rff.get_q_ave_for_stat_q_bins()
            if arr_x is None :
                logger.warning('Requested array q_ave_for_stat_q_bins is not available.'
                               '\nDrawing command is terminated.'
                               '\nCheck if the file is available.', __name__)
                return

            if self.but_intens_gr.hasFocus() : 

                arrsy  = rff.get_intens_stat_q_bins_arr()
                if arrsy is None :
                    logger.warning('Requested array q_ave_for_stat_q_bins is not available.'
                                   '\nDrawing command is terminated.'
                                   '\nCheck if the file is available.', __name__)
                    return

                arr_n  = rff.get_time_records_arr()[:,1] # take the time of event component
                arrays = (arrsy, arr_x, arr_n)
                ofname = './fig_intensity_for_static_q_t-intervals.png'
                title  = 'Intensity for static q bins in time intervals'
                
                #print 'arr_n :\n',     arr_n
                #print 'arr_n.shape :', arr_n.shape
                #print 'arr_x :\n',     arr_x
                #print 'arr_x.shape :', arr_x.shape
                #print 'arrsy :\n',     arrsy
                #print 'arrsy.shape :', arrsy.shape
                
                
            elif self.but_intens_gr1.hasFocus() :  

                arr_y  = rff.get_intens_stat_q_bins()
                arrays = (arr_y, arr_x, None)
                ofname = './fig_intensity_for_static_q_ave.png'
                title  = 'Intensity for static q bins averaged'

            labs=(r'$q_{static}$', r'$<I>(q)$')
            cp.plot_gr = PlotGraph(None, arrays, ofname, title, axlabs=labs)                 
            cp.plot_gr.move(self.parentWidget().parentWidget().pos().__add__(QtCore.QPoint(870,20)))
            cp.plot_gr.show()



    def stringTau(self):
        return 'tau(%d)=%d' % (self.tau_ind, self.tau_val) 


    def onSlider(self):
        self.tau_ind = self.sli.value()
        self.tau_val = self.list_of_tau[self.tau_ind]        
        value_str = u"\u03C4" + str( '(' + str(self.tau_ind) + ')=' + str(self.tau_val) )
        #logger.info('onSlider: value = ' + value_str , __name__)
        self.edi_tau.setText(value_str)


    def onSliderReleased(self):
        #print 'onSliderReleased'
        self.tau_ind = self.sli.value()
        self.tau_val = self.list_of_tau[self.tau_ind]
        logger.info('onSliderReleased: ' + self.stringTau() , __name__)

        if cp.plotimgspe_g is not None :
            self.redrawPlot()            

    def on_cbx(self):
        #if self.cbx_dark.hasFocus() :
        par = cp.vc_cbx_show_more
        par.setValue( self.cbx_more.isChecked() )
        msg = 'on_cbx - set status of parameter vc_cbx_show_more: ' + str(par.value())
        logger.info(msg, __name__ )
        self.setButtonState()



    def setButtonState(self):

        is_visible = self.cbx_more.isChecked()

        self.tit_data.setVisible(is_visible)
        #self.tit_geom.setVisible(is_visible)
        #self.tit_part.setVisible(is_visible)
        #self.tit_calc.setVisible(is_visible)
        #self.tit_mask.setVisible(is_visible)

        self.but_Ip   .setVisible(is_visible)
        self.but_If   .setVisible(is_visible)
        self.but_I2   .setVisible(is_visible)
        self.but_g2raw.setVisible(is_visible)

        self.but_X            .setVisible(is_visible)
        self.but_Y            .setVisible(is_visible)
        self.but_R            .setVisible(is_visible)
        #self.but_P            .setVisible(is_visible)
        #self.but_Q            .setVisible(is_visible)

        #self.but_P_st         .setVisible(is_visible)
        #self.but_Q_st         .setVisible(is_visible)
        #self.but_QP_st        .setVisible(is_visible)
        #self.but_P_dy         .setVisible(is_visible)
        #self.but_Q_dy         .setVisible(is_visible)
        #self.but_QP_dy        .setVisible(is_visible)

        self.but_1oIp         .setVisible(is_visible)
        self.but_1oIf         .setVisible(is_visible)
        self.but_g2map        .setVisible(is_visible)
        self.but_g2dy         .setVisible(is_visible)
        #self.but_g2tau        .setVisible(is_visible)
        #self.but_g2tau_gr     .setVisible(is_visible)

        self.but_mask_img_lims.setVisible(is_visible)
        self.but_mask_blemish .setVisible(is_visible)
        self.but_mask_hotpix  .setVisible(is_visible)
        self.but_mask_satpix  .setVisible(is_visible)
        self.but_mask_roi     .setVisible(is_visible)
        #self.but_mask_total   .setVisible(is_visible)
 
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIViewControl ()
    widget.show()
    app.exec_()

#-----------------------------
