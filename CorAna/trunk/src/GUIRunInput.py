#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIRunInput ...
#
#------------------------------------------------------------------------

"""GUI for run input information"""

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
from FileNameManager        import fnm
from BatchJobCorAna         import bjcora

#---------------------
#  Class definition --
#---------------------
class GUIRunInput ( QtGui.QWidget ) :
    """GUI for run input information"""

    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(100, 10, 740, 350)
        self.setWindowTitle('Run input information')
        self.setFrame()
 
        self.tit_title  = QtGui.QLabel('Input information')
        self.tit_status = QtGui.QLabel('Status:')

        self.makeTableInfo()
        self.makeTable()

        self.hboxT = QtGui.QHBoxLayout()
        self.hboxT.addWidget(self.tit_title)

        self.hboxS = QtGui.QHBoxLayout()
        self.hboxS.addWidget(self.tit_status)
        self.hboxS.addStretch(1)     

        self.hboxN = QtGui.QHBoxLayout()
        self.hboxN.addStretch(1)     
        self.hboxN.addWidget(self.table)
        self.hboxN.addStretch(1)     

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.hboxT)
        self.vbox.addLayout(self.hboxN)
        #self.vbox.addWidget(self.table)
        self.vbox.addWidget(self.table_info)
        #self.vbox.addStretch(1)     
        self.vbox.addLayout(self.hboxS)
        self.setLayout(self.vbox)
        
        self.showToolTips()
        self.setStyle()
        self.setStatus()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        self           .setToolTip('This GUI is intended for run control and monitoring.')
        #self.but_close .setToolTip('Close this window.')
        #self.but_apply .setToolTip('Apply changes to configuration parameters.')
        #self.but_show  .setToolTip('Show ...')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setStyle(self):
        self.setMinimumSize(740,350)
        self.           setStyleSheet (cp.styleBkgd)
        self.tit_title .setStyleSheet (cp.styleTitleBold)
        self.tit_title .setAlignment  (QtCore.Qt.AlignCenter)
        self.tit_status.setAlignment  (QtCore.Qt.AlignLeft)
        self.tit_status.setStyleSheet (cp.styleTitle)


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

        #try    : cp.guisystemsettingsleft.close()
        #except : pass

        #try    : cp.guisystemsettingsright.close()
        #except : pass

        try    : del cp.guiruninput # GUIRunInput
        except : pass



#    def onEdiNParts(self) :
#        s = str(self.edi_bat_nparts.text()) 
#        logger.info('onEdiNParts: ' + s, __name__)
#        cp.bat_img_nparts.setValue(s)
#        self.setTableItems()
#        bjcora.init_list_for_proc()


    def makeTableInfo(self) :

        self.table_info = QtGui.QTableWidget(4,5,self)
        self.table_info.setHorizontalHeaderLabels(['File name', 'Start', 'End', 'Total', u'\u0394t(sec):'])
        self.table_info.setVerticalHeaderLabels  (['Data:', 'Dark:', 'Flat:', 'Blem:'])

        self.table_info.horizontalHeader().setDefaultSectionSize(60)
        self.table_info.horizontalHeader().resizeSection(0,300)
        self.table_info.horizontalHeader().resizeSection(4,150)

        self.item_data_file  = QtGui.QTableWidgetItem('Data')        
        self.item_dark_file  = QtGui.QTableWidgetItem('Dark')        
        self.item_flat_file  = QtGui.QTableWidgetItem('Flat')        
        self.item_blem_file  = QtGui.QTableWidgetItem('Blem')        

        self.item_data_start = QtGui.QTableWidgetItem('Start')        
        self.item_data_end   = QtGui.QTableWidgetItem('End'  )        
        self.item_data_total = QtGui.QTableWidgetItem('Total')        
        self.item_data_time  = QtGui.QTableWidgetItem('Time' )        

        self.setTableInfoItems()

        self.list_of_items1 = [self.item_data_file,  self.item_dark_file, self.item_flat_file,  self.item_blem_file]
        self.list_of_items2 = [self.item_data_start, self.item_data_end,  self.item_data_total, self.item_data_time]
        self.list_of_items  = self.list_of_items1 + self.list_of_items2

        for i, item in enumerate(self.list_of_items1) :
            self.table_info.setItem(i, 0, item)
            item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        for i, item in enumerate(self.list_of_items2) :
            self.table_info.setItem(0, i+1, item)
            item.setTextAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        item_flags = QtCore.Qt.ItemFlags(QtCore.Qt.NoItemFlags)

        for item in self.list_of_items :
            item.setBackgroundColor (cp.colorEditInfo)
            item.setFlags(item_flags)

        self.table_info.setFixedSize(self.table_info.horizontalHeader().length()+55,self.table_info.verticalHeader().length()+30)
        #self.table_info.setFixedSize(685,150)
        #self.table_info.resize(1,1)

        #self.table_info.horizontalHeader().setStretchLastSection(True)
        #self.table_info.verticalHeader().setStretchLastSection(True)



    def setTableInfoItems(self) :

        self.item_data_file.setText( os.path.basename(fnm.path_data_xtc_cond()) )        

        if cp.bat_dark_is_used.value()  : self.item_dark_file.setText( os.path.basename(fnm.path_pedestals_ave()) )
        else                            : self.item_dark_file.setText( 'is not used' )

        if cp.ccdcorr_flatfield.value() : self.item_flat_file.setText( os.path.basename(fnm.path_flat()) )
        else                            : self.item_flat_file.setText( 'is not used' )

        if cp.ccdcorr_blemish.value()   : self.item_blem_file.setText( os.path.basename(fnm.path_blem()) )
        else                            : self.item_blem_file.setText( 'is not used' )

        self.item_data_start.setText ( str( cp.bat_data_start.value() ) )        
        self.item_data_end  .setText ( str( cp.bat_data_end  .value() ) )        
        self.item_data_total.setText ( str( cp.bat_data_total.value() ) )        
        self.item_data_time .setText ( str( cp.bat_data_dt_ave.value() ) + u'\u00B1'
                                     + str( cp.bat_data_dt_rms.value() ) )        
 


    def makeTable(self) :
        self.table = QtGui.QTableWidget(1,7,self)
        self.table.setHorizontalHeaderLabels(['#Parts', '#Rows', '#Cols', 'Img size', 'Part size', 'Rest', 'Control'])
        #self.table.setVerticalHeaderLabels(['Set:'])
        self.table.verticalHeader().hide()

        self.table.horizontalHeader().setDefaultSectionSize(60)
        self.table.horizontalHeader().resizeSection(3,80)
        self.table.horizontalHeader().resizeSection(4,80)
        self.table.horizontalHeader().resizeSection(6,70)

        #self.edi_bat_nparts = QtGui.QLineEdit       (str(cp.bat_img_nparts.value()))        
        #self.item_nparts    = QtGui.QTableWidget(self.edi_bat_nparts)

        self.item_nparts    = QtGui.QTableWidgetItem(str(cp.bat_img_nparts.value()))
        self.item_rows      = QtGui.QTableWidgetItem(str(cp.bat_img_rows.value()))
        self.item_cols      = QtGui.QTableWidgetItem(str(cp.bat_img_cols.value()))
        self.item_size      = QtGui.QTableWidgetItem(str(cp.bat_img_size.value()))
        self.item_part_size = QtGui.QTableWidgetItem('y')
        self.item_rest_size = QtGui.QTableWidgetItem('z')
        self.item_control   = QtGui.QTableWidgetItem('lock')

        item_flags = QtCore.Qt.ItemFlags(QtCore.Qt.NoItemFlags)
        self.item_rows     .setFlags(item_flags)
        self.item_cols     .setFlags(item_flags)
        self.item_size     .setFlags(item_flags)
        self.item_part_size.setFlags(item_flags)
        self.item_rest_size.setFlags(item_flags)

        self.item_control  .setCheckState(QtCore.Qt.Checked) # Unchecked, PartiallyChecked, Checked
        self.item_control  .setFlags(QtCore.Qt.ItemFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled))

        self.setTableItems()

        self.table.setItem(0, 0, self.item_nparts)
        self.table.setItem(0, 1, self.item_rows)
        self.table.setItem(0, 2, self.item_cols)
        self.table.setItem(0, 3, self.item_size)
        self.table.setItem(0, 4, self.item_part_size)
        self.table.setItem(0, 5, self.item_rest_size)
        self.table.setItem(0, 6, self.item_control)
        #self.table.setCellWidget(0, 3, self.edi_bat_nparts)
        

        #self.table.setFixedSize(445,60)
        self.table.setFixedSize(self.table.horizontalHeader().length() + 4,
                                self.table.verticalHeader()  .length() + 28)

        #self.table.horizontalHeader().setStretchLastSection(True)
        #self.table.verticalHeader().setStretchLastSection(True)

        self.table.itemChanged.connect(self.onItemChanged)
        #self.connect( self.edi_bat_nparts, QtCore.SIGNAL('editingFinished()'), self.onEdiNParts)



    def setTableItems(self) :

        #self.item_rows      .setText(str(cp.bat_img_rows.value()))
        #self.item_cols      .setText(str(cp.bat_img_cols.value()))
        #self.item_size      .setText(str(cp.bat_img_size.value()))
        #self.edi_bat_nparts .setText(str(cp.bat_img_nparts.value()))        
        #self.item_nparts   .setText(str(cp.bat_img_nparts.value())) 

        img_size  = cp.bat_img_size.value()
        nparts    = cp.bat_img_nparts.value()
        part_size = int(img_size/nparts)
        rest_size = img_size%nparts 
        self.item_part_size .setText(str(part_size))
        self.item_rest_size .setText(str(rest_size))

        self.item_rows     .setBackgroundColor (cp.colorEditInfo)
        self.item_cols     .setBackgroundColor (cp.colorEditInfo)
        self.item_size     .setBackgroundColor (cp.colorEditInfo)
        self.item_part_size.setBackgroundColor (cp.colorEditInfo)
        self.item_nparts   .setBackgroundColor (cp.colorEdit)
        #self.item_control  .setBackgroundColor (cp.colorEdit)

        self.setEditableField()

        if rest_size == 0 :
            self.item_rest_size.setBackgroundColor (cp.colorEditInfo)
        else :
            self.item_rest_size.setBackgroundColor (cp.colorEditBad)
 

    def setEditableField(self) :
        if self.item_control.checkState() == QtCore.Qt.Checked :
            self.item_nparts.setBackgroundColor (cp.colorEditInfo)
            self.item_nparts.setFlags(QtCore.Qt.ItemFlags(QtCore.Qt.NoItemFlags))
        else :
            self.item_nparts.setBackgroundColor (cp.colorEdit)
            self.item_nparts.setFlags(QtCore.Qt.ItemFlags(QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled))
 

    def onItemChanged(self, item):
        logger.debug('onItemChanged', __name__)

        self.table.blockSignals(True)

        if item == self.item_control :
            s = str(item.text()) 
            logger.info('Changed item: ' + s + ' state: ' + str(item.checkState()), __name__)
            #self.setTableItems()
            self.setEditableField()


        elif item == self.item_nparts :
            s = str(item.text()) 
            logger.info('Changed item: ' + s, __name__)
            cp.bat_img_nparts.setValue(s)
            self.setTableItems()
            bjcora.init_list_for_proc()
            bjcora.remove_files_cora_all()

        self.table.blockSignals(False)



    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def onSave(self):
        fname = cp.fname_cp.value()
        logger.debug('onSave:', __name__)
        cp.saveParametersInFile( fname )


    def onShow(self):
        logger.debug('onShow - is not implemented yet...', __name__)


    def setStatus(self, status_index=0, msg=''):

        list_of_states = ['Good', 'Warning', 'Alarm']
        if status_index == 0 : self.tit_status.setStyleSheet(cp.styleStatusGood)
        if status_index == 1 : self.tit_status.setStyleSheet(cp.styleStatusWarning)
        if status_index == 2 : self.tit_status.setStyleSheet(cp.styleStatusAlarm)

        self.tit_status.setText('Status: ' + list_of_states[status_index] + msg)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIRunInput ()
    widget.show()
    app.exec_()

#-----------------------------
