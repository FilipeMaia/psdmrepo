#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDarkList ...
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
from FileNameManager        import fnm
from GUIDarkListItem        import *
import GlobalUtils          as     gu
import RegDBUtils           as     ru
from BatchLogScanParser     import blsp

#---------------------
#  Class definition --
#---------------------
#class GUIDarkList ( QtGui.QGroupBox ) :
class GUIDarkList ( QtGui.QWidget ) :
    """GUI for the list of widgers"""

    def __init__ ( self, parent=None ) :

        self.parent = parent
        self.dark_list_show_runs   = cp.dark_list_show_runs
        self.dark_list_show_dets   = cp.dark_list_show_dets
        self.instr_name            = cp.instr_name
        self.exp_name              = cp.exp_name
        self.det_name              = cp.det_name
        self.str_run_number        = cp.str_run_number
        self.list_of_det_pars      = cp.list_of_det_pars
        self.list_of_dets_selected = cp.list_of_dets_selected

        #self.calib_dir      = cp.calib_dir
        #self.det_name       = cp.det_name

        #QtGui.QGroupBox.__init__(self, 'Runs', parent)
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(100, 100, 730, 400)
        self.setWindowTitle('List of dark runs')
        #self.setTitle('My status')
        #try : self.setWindowIcon(cp.icon_help)
        #except : pass
        self.setFrame()

        self.list = QtGui.QListWidget(parent=self)

        self.widg_width = 500

        self.size     = QtCore.QSize(self.widg_width,35)
        self.size_ext = QtCore.QSize(self.widg_width,200)

        self.updateList()

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.list)
        self.setLayout(vbox)

        self.list.itemClicked.connect(self.onItemClick)
        self.list.itemDoubleClicked.connect(self.onItemDoubleClick)

        #self.connect(self.list.horizontalHeader(),
        #             QtCore.SIGNAL('sectionClicked (int)'),
        #             self.random_function)
 
        self.showToolTips()
        self.setStyle()

        cp.guidarklist = self

    #-------------------
    #  Public methods --
    #-------------------

    def updateList(self):

        self.list.clear()

        if self.instr_name.value() == self.instr_name.value_def() : return
        if self.exp_name  .value() == self.exp_name  .value_def() : return

        # Get run records from RegDB
        self.dict_run_recs = ru.calibration_runs (self.instr_name.value(), self.exp_name.value())
        #print 'self.dict_run_recs = ', self.dict_run_recs

        self.list_of_records = []
        self.list_of_run_strs_in_dir = fnm.get_list_of_xtc_runs()   # ['0001', '0202', '0203',...]
        self.list_of_run_nums_in_dir = gu.list_of_int_from_list_of_str(self.list_of_run_strs_in_dir) # [1, 202, 203, 204,...]
        self.list_of_run_nums_in_regdb = ru.list_of_runnums(self.instr_name.value(), self.exp_name.value())

        #print 'list_of_run_nums_in_dir:\n',   self.list_of_run_nums_in_dir
        #print 'list_of_run_nums_in_regdb:\n', self.list_of_run_nums_in_regdb
        
        if self.list_of_run_nums_in_regdb == [] : self.list_of_runs = self.list_of_run_nums_in_dir
        else                                    : self.list_of_runs = self.list_of_run_nums_in_regdb

        for run_num in self.list_of_runs :

            str_run_num = '%04d' % run_num
            self.str_run_number.setValue(str_run_num)

            if not self.isSelectedRun            (run_num) : continue
            if not self.hasSelectedDetectorsInRun(run_num) : continue
        
            if not run_num in self.list_of_run_nums_in_dir :
                self.comment = 'NOT FOUND xtc file!'
                #self.type    = 'N/A'


            widg = GUIDarkListItem ( self, str_run_num, self.type, self.comment)
            item = QtGui.QListWidgetItem('', self.list)
            #self.list.addItem(item)
            #item.setFlags (  QtCore.Qt.ItemIsEnabled ) #| QtCore.Qt.ItemIsSelectable  | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsTristate)
            #print 'item.flags(): %o' % item.flags()
            #item.setCheckState(0)
            item.setFlags (  QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable  | QtCore.Qt.ItemIsUserCheckable )
            #item.setFlags ( QtCore.Qt.ItemIsEnabled )

            item.setSizeHint(self.size)
            self.list.setItemWidget(item, widg)
            #self.list.setItemSelected(item, True)

            record = str_run_num, item, widg
            self.list_of_records.append(record)



    def isSelectedRun(self, run_num, type_to_select = 'dark') :

        # Unpack RegDB info
        if self.dict_run_recs != {} :
            run_rec = self.dict_run_recs[run_num]
            list_of_calibs = run_rec['calibrations']
            self.comment   = run_rec['comment']
            self.is_found_type = type_to_select in list_of_calibs
            #print run_num, is_found_type, list_of_calibs, self.comment
        else :
            self.is_found_type = False
            self.comment   = ''

        if self.is_found_type :
            self.type = type_to_select
            return True

        if self.dark_list_show_runs.value() == self.dark_list_show_runs.value_def() : # 'all' - For all runs
            self.type = ''
            return True

        return False



    def det_is_in_list_of_sources(self, det_name, list_of_srcs) :

        pattern = det_name.lower() + '.'
        for src in list_of_srcs :
            if src.lower().find(pattern) != -1 :
                return True
        return False



    def hasSelectedDetectorsInRun(self, run_num) :

        if self.dark_list_show_dets.value() == self.dark_list_show_dets.value_def() : # 'all' - For all detectors
            return True

        if self.det_name.value() == '' : # If detector(s) were not selected
            logger.warning('Detector is not selected !!!', __name__)
            return False

        list_of_srcs = blsp.get_list_of_sources()

        # For all detectors in run
        for det in self.list_of_dets_selected() :
            if not self.det_is_in_list_of_sources(det, list_of_srcs) : return False
        return True

        # For any detector in run
        #for det in self.list_of_dets_selected() :
        #    if self.det_is_in_list_of_sources(det, list_of_srcs) : return True
        #return False



    def setFieldsEnabled(self, is_enabled=False):
        for (run, item, widg) in self.list_of_records :
            #print '  run:', run          
            widg.setFieldsEnabled(is_enabled)


    def getRunAndItemForWidget(self, widg_active):
        for (run, item, widg) in self.list_of_records :
            if widg == widg_active :
                return run, item
        return None, None


    def showToolTips(self):
        #self           .setToolTip('This GUI is intended for run control and monitoring.')
        #self.but_close .setToolTip('Close this window.')
        pass

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(1)
        self.frame.setMidLineWidth(0)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setStyle(self):
        self.setMinimumWidth(self.widg_width)
        #self.setFixedHeight(400)
        self.setMinimumHeight(200)
        #self.setBaseSize(600, 400)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        #self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.           setStyleSheet (cp.styleBkgd)

        #self.list.adjustSize()
        #print 'self.list.size():',  self.list.size()
        #self.setMinimumSize(self.list.size())

        #self.tit_status.setStyleSheet (cp.styleTitle)
        #self.tit_status.setStyleSheet (cp.styleDefault)
        #self.tit_status.setStyleSheet (cp.styleTitleInFrame)
        #self.lab_txt   .setReadOnly   (True)
        #self.lab_txt   .setStyleSheet (cp.styleWhiteFixed) 
        #self.lab_txt   .setStyleSheet (cp.styleBkgd)
        
        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))
        #self.setContentsMargins (QtCore.QMargins(1,10,1,1))


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())
        #self.lab_txt.setGeometry(self.contentsRect())
        
    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        #self.saveLogTotalInFile() # It will be saved at closing of GUIMain

        #try    : cp.guimain.butLogger.setStyleSheet(cp.styleButtonBad)
        #except : pass

        #self.lab_txt.close()

        #try    : del cp.guidarklist # GUIDarkList
        #except : pass

        for (run, item, widg) in self.list_of_records :
            widg.close()


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

        
    def onItemExpand(self, widg):
        run, item = self.getRunAndItemForWidget(widg)
        logger.info('Expand widget for run %s' % run, __name__)
        self.size_ext = QtCore.QSize(self.widg_width, widg.getHeight())
        item.setSizeHint(self.size_ext)


    def onItemShrink(self, widg):
        logger.info('onItemExpand', __name__)
        run, item = self.getRunAndItemForWidget(widg)
        logger.info('Shrink widget for run %s' % run, __name__)
        item.setSizeHint(self.size)


    def onItemClick(self, item):
        logger.info('onItemClick - do nothing...', __name__)
        #print 'onItemClick' # , isChecked: ', str(item.checkState())

        widg = self.list.itemWidget(item)

        if item.sizeHint() == self.size_ext :
            pass
            #widg.onClickShrink()
            #print 'widg.size:', widg.size()
            #item.setSizeHint(self.size)
        else :
            pass
            #widg.onClickExpand()
            #print 'widg.size:', widg.size()

            #self.size_ext = QtCore.QSize(self.widg_width, widg.getHeight())
            #item.setSizeHint(self.size_ext)


    def onItemDoubleClick(self, item):
        logger.info('onItemDoubleClick', __name__)
        #print 'onItemDoubleClick' #, isChecked: ', str(item.checkState())

    
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    w = GUIDarkList()
    #w.setStatusMessage('Test of GUIDarkList...')
    #w.statusOfDir('./')
    w.show()
    app.exec_()

#-----------------------------
