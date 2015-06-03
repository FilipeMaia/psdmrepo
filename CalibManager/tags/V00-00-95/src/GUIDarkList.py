#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDarkList ...
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

from ConfigParametersForApp import cp
from Logger                 import logger
from FileNameManager        import fnm
from GUIDarkListItem        import *
import GlobalUtils          as     gu
import RegDBUtils           as     ru
from BatchLogScanParser     import blsp

from time import time

#---------------------
#  Class definition --
#---------------------
#class GUIDarkList ( QtGui.QGroupBox ) :
class GUIDarkList ( QtGui.QWidget ) :
    """GUI for the list of widgers"""

    def __init__ ( self, parent=None ) :

        self.parent = parent
        self.dark_list_run_min     = cp.dark_list_run_min 
        self.dark_list_run_max     = cp.dark_list_run_max 
        self.dark_list_show_runs   = cp.dark_list_show_runs
        self.dark_list_show_dets   = cp.dark_list_show_dets
        self.list_of_show_dets     = cp.list_of_show_dets
        self.instr_name            = cp.instr_name
        self.exp_name              = cp.exp_name
        self.det_name              = cp.det_name
        self.str_run_number        = cp.str_run_number
        self.list_of_det_pars      = cp.list_of_det_pars
        self.list_of_dets_selected = cp.list_of_dets_selected
        self.dict_guidarklistitem  = cp.dict_guidarklistitem
        self.list_of_visible_records = []

        self.click_counter = 0
        self.update_counter = 0

        #self.calib_dir      = cp.calib_dir
        #self.det_name       = cp.det_name

        #QtGui.QGroupBox.__init__(self, 'Runs', parent)
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(100, 100, 800, 300)
        self.setWindowTitle('List of dark runs')
        #self.setTitle('My status')
        #try : self.setWindowIcon(cp.icon_help)
        #except : pass
        self.setFrame()

        #self.list = QtGui.QListWidget(parent=self)
        # Use singleton object
        if cp.dark_list is None : self.list = cp.dark_list = QtGui.QListWidget()
        else                    : self.list = cp.dark_list

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

        self.connectToThreadWorker()

        cp.guidarklist = self

    #-------------------
    #  Public methods --
    #-------------------

    def connectToThreadWorker(self):
        try : self.connect   ( cp.thread_check_new_xtc_files, QtCore.SIGNAL('update(QString)'), self.signalReciever )
        except : logger.warning('connectToThreadWorker is failed', __name__)


    def disconnectFromThreadWorker(self):
        try : self.disconnect( cp.thread_check_new_xtc_files, QtCore.SIGNAL('update(QString)'), self.signalReciever )
        except : pass


    def signalReciever(self, text):
        msg = 'Signal received: new xtc file is available, msg: %s' % text
        #print msg 
        logger.info(msg, __name__)
        self.updateList()
        self.scrollDown()


    def scrollDown(self):
        #self.list.moveCursor(QtGui.QTextCursor.End)
        #self.list.repaint()
        last_run_num = self.list_of_runs[-1]
        item, widg = self.dict_guidarklistitem[last_run_num]
        self.list.scrollToItem (item) #, hint=QAbstractItemView.EnsureVisible)


    def updateList(self, clearList=False) :

        self.update_counter += 1

        self.t0_sec = time()

        if clearList :
            self.removeItemWidgets()            
            self.list.clear()
            self.dict_guidarklistitem.clear()
            #msg = 'Consumed time to clean list (sec) = %7.3f' % (time()-self.t0_sec)        
            #print msg

        self.setItemsHidden()

        self.list_of_visible_records = []

        if self.instr_name.value() == self.instr_name.value_def() : return
        if self.exp_name  .value() == self.exp_name  .value_def() : return

        str_exp_name = self.exp_name.value()
        if 'tut' in str_exp_name :
            msg = 'Tutorial directory %s can not be used to generate the list of runs,'\
                  ' because their xtc files are not registered in the DB.' % str_exp_name
            logger.warning(msg, __name__)
            return

        msg = 'Begin to update the list of runs. It is slow procedure that takes ~0.1s/run, stay calm and wait.'
        logger.info(msg, __name__)
        #print msg
        # Get run records from RegDB
        self.dict_run_recs = ru.calibration_runs (self.instr_name.value(), self.exp_name.value())
        #print 'self.dict_run_recs = ', self.dict_run_recs

        self.list_of_run_strs_in_dir = fnm.get_list_of_xtc_runs()   # ['0001', '0202', '0203',...]
        self.list_of_run_nums_in_dir = gu.list_of_int_from_list_of_str(self.list_of_run_strs_in_dir) # [1, 202, 203, 204,...]
        self.list_of_run_nums_in_regdb = ru.list_of_runnums(self.instr_name.value(), self.exp_name.value())

        #print 'list_of_run_nums_in_dir  :', self.list_of_run_nums_in_dir
        #print 'list_of_run_nums_in_regdb:', self.list_of_run_nums_in_regdb
        #print '\nA. Consumed time (sec) =', time()-self.t0_sec
        #print 'Begin to construct the list of items for %s' % self.exp_name.value()

        
        if self.list_of_run_nums_in_regdb == [] : self.list_of_runs = self.list_of_run_nums_in_dir
        else                                    : self.list_of_runs = self.list_of_run_nums_in_regdb


        for run_num in self.list_of_runs :

            str_run_num = '%04d' % run_num
            self.str_run_number.setValue(str_run_num)

            if not self.isSelectedRun            (run_num) :
                #print 'not self.isSelectedRun', run_num 
                continue

            if not self.hasSelectedDetectorsInRun(run_num) :
                #print 'not self.hasSelectedDetectorsInRun', run_num 
                continue

            if not run_num in self.list_of_run_nums_in_dir :
                self.comment = 'xtc file IS NOT on disk!'
                self.xtc_in_dir = False
            else :
                self.xtc_in_dir = True
                #self.type    = 'N/A'

            item, widg = self.create_or_use_guidarklistitem(run_num)

            self.list.setItemHidden (item, False)

            record = run_num, item, widg
            self.list_of_visible_records.append(record)


        self.list.sortItems(QtCore.Qt.AscendingOrder)

        msg = 'Consumed time to generate list of files (sec) = %7.3f' % (time()-self.t0_sec)        
        logger.info(msg, __name__)



    def create_or_use_guidarklistitem(self, run_num) :
        """Creates QListWidgetItem and GUIDarkListItem objects for the 1st time and add them to self.list or use existing from the dictionary
        """
        if run_num in self.dict_guidarklistitem.keys() :
            #print 'Use existing GUIDarkListItem object for run %d' % run_num
            item, widg = self.dict_guidarklistitem[run_num]
            widg.updateButtons(self.type, self.comment, self.xtc_in_dir)
            return item, widg
        else :
            #print 'Create new GUIDarkListItem object for run %d' % run_num
            str_run_num = '%04d'%run_num
            item = QtGui.QListWidgetItem(str_run_num, self.list)
            widg = GUIDarkListItem(self, str_run_num, self.type, self.comment, self.xtc_in_dir)  
            self.dict_guidarklistitem[run_num] = [item, widg]
            item.setTextColor(QtGui.QColor(0, 0, 0, alpha=0)) # set item text invisible. All pars in the range [0,255]
            item.setFlags ( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable  | QtCore.Qt.ItemIsUserCheckable )
            item.setSizeHint(widg.size())
            self.list.setItemWidget(item, widg)
            return item, widg



    def setItemsHidden(self) :        
        for run, (item, widg) in self.dict_guidarklistitem.iteritems() :
            self.list.setItemHidden (item, True)
            #print 'Hide item for run %d' % run
 

    def removeItemWidgets(self) :     
        for run, (item, widg) in self.dict_guidarklistitem.iteritems() :
            self.list.removeItemWidget(item)
            widg.close()


    def isSelectedRun(self, run_num, type_to_select = 'dark') :
        # Unpack RegDB info
        if self.dict_run_recs != {} : ### and run_num in self.dict_run_recs.keys() : ###!!!!!!!!!!!!!!!!!!!
            run_rec = self.dict_run_recs[run_num]
            list_of_calibs = run_rec['calibrations']
            self.comment   = run_rec['comment']
            self.is_found_type = type_to_select in list_of_calibs
            if self.is_found_type : self.type = type_to_select
            else                  : self.type = ''
        else :
            self.is_found_type = False
            self.comment = ''
            self.type = ''

        if self.dark_list_show_runs.value() == 'in range' : # self.list_of_show_runs[0]
            if   run_num > self.dark_list_run_max.value() : return False
            elif run_num < self.dark_list_run_min.value() : return False
            else                                          : return True

        #if self.dark_list_show_runs.value() == 'dark' : # self.list_of_show_runs[1]
        if self.is_found_type :
            return True

        if self.dark_list_show_runs.value() == 'all' : # self.list_of_show_runs[2]
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

        if self.dark_list_show_dets.value() == self.list_of_show_dets[1] : # 'selected any' - For any of selected detectors in run
            for det in self.list_of_dets_selected() :
                if self.det_is_in_list_of_sources(det, list_of_srcs) : return True
            return False

        if self.dark_list_show_dets.value() == self.list_of_show_dets[2] : # 'selected all' - For all selected detectors in run
            for det in self.list_of_dets_selected() :
                if not self.det_is_in_list_of_sources(det, list_of_srcs) : return False
            return True

        return True


    def setFieldsEnabled(self, is_enabled=False):
        #for run, (item, widg) in self.dict_guidarklistitem.iteritems() :
        for (run, item, widg) in self.list_of_visible_records :
            #print '  run:', run          
            widg.setFieldsEnabled(is_enabled)


    def getRunAndItemForWidget(self, widg_active):
        #for run, (item, widg) in self.dict_guidarklistitem.iteritems() :
        for (run, item, widg) in self.list_of_visible_records :
            if widg == widg_active :
                return '%04d'%run, item
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
        self.setMinimumSize(760,80)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.setStyleSheet (cp.styleBkgd)
        self.setContentsMargins (QtCore.QMargins(-9,-9,-9,-9))


        #self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        #self.list.adjustSize()
        #print 'self.list.size():',  self.list.size()
        #self.setMinimumSize(self.list.size())

        #self.tit_status.setStyleSheet (cp.styleTitle)
        #self.tit_status.setStyleSheet (cp.styleDefault)
        #self.tit_status.setStyleSheet (cp.styleTitleInFrame)
        #self.lab_txt   .setReadOnly   (True)
        #self.lab_txt   .setStyleSheet (cp.styleWhiteFixed) 
        #self.lab_txt   .setStyleSheet (cp.styleBkgd)
        

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())
        #self.lab_txt.setGeometry(self.contentsRect())
        #print 'self.rect():', str(self.rect())

        
    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)

        self.disconnectFromThreadWorker()

        #self.saveLogTotalInFile() # It will be saved at closing of GUIMain

        #try    : cp.guimain.butLogger.setStyleSheet(cp.styleButtonBad)
        #except : pass

        #self.lab_txt.close()

        #try    : del cp.guidarklist # GUIDarkList
        #except : pass

        # DO NOT CLOSE THESE WIDGETS!
        #for run, (item, widg) in self.dict_guidarklistitem.iteritems() :
        #    widg.close()


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

        
    def onItemExpand(self, widg):
        run, item = self.getRunAndItemForWidget(widg)
        logger.debug('Expand widget for run %s' % run, __name__)
        item.setSizeHint(widg.size())


    def onItemShrink(self, widg):
        run, item = self.getRunAndItemForWidget(widg)
        logger.debug('Shrink widget for run %s' % run, __name__)
        item.setSizeHint(widg.size())


    def onItemClick(self, item):
        #print 'onItemClick' # , isChecked: ', str(item.checkState())

        if item.isSelected(): item.setSelected(False)
        #else               : item.setSelected(True)

        self.list_of_run_strs_in_dir = fnm.get_list_of_xtc_runs()   # ['0001', '0202', '0203',...]
        self.list_of_run_nums_in_dir = gu.list_of_int_from_list_of_str(self.list_of_run_strs_in_dir) # [1, 202, 203, 204,...]

        widg = self.list.itemWidget(item)
        run_num = widg.getRunNum()

        run_rec = self.dict_run_recs[run_num]
        self.comment   = run_rec['comment']
        list_of_calibs = run_rec['calibrations']
        self.type = ', '.join(list_of_calibs)

        if not run_num in self.list_of_run_nums_in_dir :
            self.comment = 'xtc file IS NOT on disk!'
            self.xtc_in_dir = False
        else :
            self.xtc_in_dir = True

        widg.updateButtons(self.type, self.comment, self.xtc_in_dir)

        msg = 'onItemClick - update button status for run %s with calib_type: %s, comment: %s, xtc_in_dir %s' \
              % (widg.str_run_num, self.type, self.comment, self.xtc_in_dir) 
        logger.info(msg, __name__)


    def onItemDoubleClick(self, item):
        logger.debug('onItemDoubleClick', __name__)
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
