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

from ConfigParametersCorAna import confpars as cp
from Logger                 import logger
from ViewResults            import *

#---------------------
#  Class definition --
#---------------------
class GUIViewControl ( QtGui.QWidget ) :
    """GUI View Control"""

    def __init__ ( self, parent=None, fname=None ) :
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
        self.tau_ind = 0

        self.initCorArray()        
        
        self.tit       = QtGui.QLabel('View Control')
        self.edi       = QtGui.QLineEdit( os.path.basename(cp.res_fname.value()) )        
        self.but       = QtGui.QPushButton('File')
        self.but_close = QtGui.QPushButton('Close')
        self.but_Ip    = QtGui.QPushButton('<Ip>')
        self.but_If    = QtGui.QPushButton('<If>')
        self.but_I2    = QtGui.QPushButton('<Ip x If>')
        self.but_G2    = QtGui.QPushButton('100 x G2')
   
        self.but_X     = QtGui.QPushButton('X')
        self.but_Y     = QtGui.QPushButton('Y')
        self.but_R     = QtGui.QPushButton('R')
        self.but_P     = QtGui.QPushButton('Phi')
        self.but_Q     = QtGui.QPushButton('Q')

        self.but_P_st  = QtGui.QPushButton('Phi stat')
        self.but_Q_st  = QtGui.QPushButton('Q stat')
        self.but_QP_st = QtGui.QPushButton('Q-Phi stat')

        self.but_P_dy  = QtGui.QPushButton('Phi dyna')
        self.but_Q_dy  = QtGui.QPushButton('Q dyna')
        self.but_QP_dy = QtGui.QPushButton('Q-Phi stat')

        self.but_1oIp  = QtGui.QPushButton('1/<Ip>')
        self.but_1oIf  = QtGui.QPushButton('1/<If>')

        self.sli = QtGui.QSlider(QtCore.Qt.Horizontal, self)        
        self.sli.setValue(0)
        self.sli.setRange(0, self.list_of_tau.shape[0]-1)
        self.sli.setTickInterval(1)
        self.edi_tau = QtGui.QLineEdit('tau(ind)')


        self.grid = QtGui.QGridLayout()
        self.grid_row = 1
        self.grid.addWidget(self.tit,      self.grid_row,   0, 1, 9)
        self.grid.addWidget(self.but,      self.grid_row+1, 0)
        self.grid.addWidget(self.edi,      self.grid_row+1, 1, 1, 9)
        self.grid.addWidget(self.edi_tau,  self.grid_row+2, 0)
        self.grid.addWidget(self.sli,      self.grid_row+2, 1, 1, 9)
        self.grid.addWidget(self.but_Ip,   self.grid_row+3, 0)
        self.grid.addWidget(self.but_If,   self.grid_row+3, 1)
        self.grid.addWidget(self.but_I2,   self.grid_row+3, 2)
        self.grid.addWidget(self.but_G2,   self.grid_row+3, 3)
        self.grid.addWidget(self.but_close,self.grid_row+3, 4)
        self.grid.addWidget(self.but_X,    self.grid_row+4, 0)
        self.grid.addWidget(self.but_Y,    self.grid_row+4, 1)
        self.grid.addWidget(self.but_R,    self.grid_row+4, 2)
        self.grid.addWidget(self.but_P,    self.grid_row+4, 3)
        self.grid.addWidget(self.but_Q,    self.grid_row+4, 4)
        self.grid.addWidget(self.but_P_st, self.grid_row+5, 0)
        self.grid.addWidget(self.but_Q_st, self.grid_row+5, 1)
        self.grid.addWidget(self.but_QP_st,self.grid_row+5, 2)
        self.grid.addWidget(self.but_P_dy, self.grid_row+5, 3)
        self.grid.addWidget(self.but_Q_dy, self.grid_row+5, 4)
        self.grid.addWidget(self.but_QP_dy,self.grid_row+5, 5)
        self.grid.addWidget(self.but_1oIp, self.grid_row+6, 0)
        self.grid.addWidget(self.but_1oIf, self.grid_row+6, 1)

        self.grid_row += 4

        #self.connect(self.edi, QtCore.SIGNAL('editingFinished()'),        self.onEdit )
        self.connect(self.but,      QtCore.SIGNAL('clicked()'),         self.onBut     )
        self.connect(self.but_close,QtCore.SIGNAL('clicked()'),         self.onButClose)
        self.connect(self.but_Ip,   QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_If,   QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_I2,   QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_G2,   QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_X,    QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_Y,    QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_R,    QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_P,    QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_Q,    QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_P_st, QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_Q_st, QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_QP_st,QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_P_dy, QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_Q_dy, QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_QP_dy,QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_1oIp, QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.but_1oIf, QtCore.SIGNAL('clicked()'),         self.onButView )
        self.connect(self.sli,      QtCore.SIGNAL('valueChanged(int)'), self.onSlider  )
        self.connect(self.sli,      QtCore.SIGNAL('sliderReleased()'),  self.onSliderReleased )
 
        self.setLayout(self.grid)

        self.showToolTips()
        self.setStyle()
        self.onSlider()

        #self.overlay = Overlay(self,'Load Results')
                
    #-------------------
    #  Public methods --
    #-------------------

    def setFileName(self, fname=None) :
        if fname == None : pass
        else : cp.res_fname.setValue(fname)


    def showToolTips(self):
        msg = 'Use this GUI to set partitions.'
        self.setToolTip(msg)
        self.edi.setToolTip('Click on "File"\nto select the file')
        self.but.setToolTip('Click on this button\nand select the file')


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

        self.but_close.setStyleSheet(cp.styleButtonBad)
        self.but_Ip   .setStyleSheet(cp.styleButton)
        self.but_If   .setStyleSheet(cp.styleButton)
        self.but_I2   .setStyleSheet(cp.styleButton)
        self.but_G2   .setStyleSheet(cp.styleButton)

        self.but_X    .setStyleSheet(cp.styleButton)
        self.but_Y    .setStyleSheet(cp.styleButton)
        self.but_R    .setStyleSheet(cp.styleButton)
        self.but_P    .setStyleSheet(cp.styleButton)
        self.but_Q    .setStyleSheet(cp.styleButton)
        self.but_P_st .setStyleSheet(cp.styleButton)
        self.but_Q_st .setStyleSheet(cp.styleButton)
        self.but_QP_st.setStyleSheet(cp.styleButton)
        self.but_P_dy .setStyleSheet(cp.styleButton)
        self.but_Q_dy .setStyleSheet(cp.styleButton)
        self.but_QP_dy.setStyleSheet(cp.styleButton)
        self.but_1oIp .setStyleSheet(cp.styleButton)
        self.but_1oIf .setStyleSheet(cp.styleButton)


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
        self.vr.set_file_name()


    def setImgArray(self):

        t0 = gu.get_time_sec()

        if self.g_ind == -1 :
            self.arr2d = self.vr.get_g2_for_itau(self.tau_ind)

        elif self.g_ind == 0 :
            self.arr2d = self.vr.get_Ip_for_itau(self.tau_ind)

        elif self.g_ind == 1 :
            self.arr2d = self.vr.get_If_for_itau(self.tau_ind)

        elif self.g_ind == 2 :
            self.arr2d = self.vr.get_I2_for_itau(self.tau_ind)

        elif self.g_ind == 3 :
            self.arr2d, y_map = self.vr.get_xy_maps()

        elif self.g_ind == 4 :
            x_map, self.arr2d = self.vr.get_xy_maps()

        elif self.g_ind == 5 :
            self.arr2d = self.vr.get_r_map()

        elif self.g_ind == 6 :
            self.arr2d = self.vr.get_phi_map()

        elif self.g_ind == 7 :
            self.arr2d = self.vr.get_q_map()

        elif self.g_ind == 8 :
            self.arr2d = self.vr.get_phi_map_for_stat_bins()

        elif self.g_ind == 9 :
            self.arr2d = self.vr.get_q_map_for_stat_bins()

        elif self.g_ind == 10 :
            self.arr2d = self.vr.get_phi_map_for_dyna_bins()

        elif self.g_ind == 11 :
            self.arr2d = self.vr.get_q_map_for_dyna_bins()

        elif self.g_ind == 12 :
            self.arr2d = self.vr.get_q_phi_map_for_stat_bins()

        elif self.g_ind == 13 :
            self.arr2d = self.vr.get_q_phi_map_for_dyna_bins()

        elif self.g_ind == 14 :
            self.arr2d = self.vr.get_1oIp_map_for_stat_bins_itau(self.tau_ind)

        elif self.g_ind == 15 :
            self.arr2d = self.vr.get_1oIf_map_for_stat_bins_itau(self.tau_ind)
    
        else :
            logger.warning('Request for non-implemented plot ...', __name__)

        print 'Get map consumed time: ', gu.get_time_sec()-t0 # < 0.04sec for 1300x1340 img  
        #print 'arr2d:\n', self.arr2d 
        print 'arr2d.shape =', self.arr2d.shape 


    def onButView(self):
        logger.info('onButView', __name__)

        if self.but_G2.hasFocus() :
            logger.info('G2 is selected', __name__)
            self.g_ind = -1

        if self.but_Ip.hasFocus() :
            logger.info('<Ip> is selected', __name__)
            self.g_ind = 0

        if self.but_If.hasFocus() :
            logger.info('<If> is selected', __name__)
            self.g_ind = 1

        if self.but_I2.hasFocus() :
            logger.info('<Ip x If> is selected', __name__)
            self.g_ind = 2

        if self.but_X.hasFocus() :
            logger.info('X is selected', __name__)
            self.g_ind = 3

        if self.but_Y.hasFocus() :
            logger.info('Y is selected', __name__)
            self.g_ind = 4

        if self.but_R.hasFocus() :
            logger.info('R is selected', __name__)
            self.g_ind = 5

        if self.but_P.hasFocus() :
            logger.info('Phi is selected', __name__)
            self.g_ind = 6

        if self.but_Q.hasFocus() :
            logger.info('Q is selected', __name__)
            self.g_ind = 7

        if self.but_P_st.hasFocus() :
            logger.info('Phi static is selected', __name__)
            self.g_ind = 8

        if self.but_Q_st.hasFocus() :
            logger.info('Q static is selected', __name__)
            self.g_ind = 9

        if self.but_P_dy.hasFocus() :
            logger.info('Phi dynamic is selected', __name__)
            self.g_ind = 10

        if self.but_Q_dy.hasFocus() :
            logger.info('Q dynamic is selected', __name__)
            self.g_ind = 11

        if self.but_QP_st.hasFocus() :
            logger.info('Q-Phi static is selected', __name__)
            self.g_ind = 12

        if self.but_QP_dy.hasFocus() :
            logger.info('Q-Phi dynamic is selected', __name__)
            self.g_ind = 13

        if self.but_1oIp.hasFocus() :
            logger.info('1/<Ip> is selected', __name__)
            self.g_ind = 14

        if self.but_1oIf.hasFocus() :
            logger.info('1/<If> is selected', __name__)
            self.g_ind = 15


        self.drawPlot()
  

    def drawPlot(self):
        try :
            self.redrawPlotResetLimits()
        except :
            #self.setImgArray()
            cp.plotimgspe_g = PlotImgSpe(None,self.arr2d) 
            cp.plotimgspe_g.move(self.parentWidget().parentWidget().pos().__add__(QtCore.QPoint(850,20)))
            cp.plotimgspe_g.show()


    def redrawPlotResetLimits(self):
        self.setImgArray()
        cp.plotimgspe_g.set_image_array_new(self.arr2d)


    def redrawPlot(self):
        self.setImgArray()
        cp.plotimgspe_g.set_image_array(self.arr2d)


    def onButClose(self):
        logger.info('onButClose', __name__)
        try    : cp.plotimgspe_g.close()
        except : pass
        try    : del cp.plotimgspe_g
        except : pass


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
        value_str = str( 'tau(' + str(self.tau_ind) + ')=' + str(self.tau_val) )
        logger.info('onSliderReleased: ' + value_str , __name__)
        self.redrawPlot()
        

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIViewControl ()
    widget.show()
    app.exec_()

#-----------------------------
