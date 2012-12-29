#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIRunInfo ...
#
#------------------------------------------------------------------------

"""GUI for run information"""

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
from FileNameManager        import fnm

#---------------------
#  Class definition --
#---------------------
class GUIRunInfo ( QtGui.QWidget ) :
    """GUI for run information"""

    def __init__ ( self, parent=None ) :

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(100, 10, 740, 350)
        self.setWindowTitle('Run information')
        self.setFrame()
 
        self.tit_title  = QtGui.QLabel('Run information')
        self.tit_status = QtGui.QLabel('Status:')
        self.tit_data   = QtGui.QLabel('Data:')
        self.tit_dark   = QtGui.QLabel('Dark:')
        self.tit_flat   = QtGui.QLabel('Flat:')
        self.tit_blam   = QtGui.QLabel('Blam:')

        self.lab_start  = QtGui.QLabel('Start:')
        self.lab_end    = QtGui.QLabel('End:')
        self.lab_total  = QtGui.QLabel('Total:')
        self.lab_time   = QtGui.QLabel(u'\u0394t(sec):')


        #self.but_close  = QtGui.QPushButton('Close') 
        #self.but_apply  = QtGui.QPushButton('Save') 

        self.edi_data = QtGui.QLineEdit('Data')        
        self.edi_dark = QtGui.QLineEdit('Dark')        
        self.edi_flat = QtGui.QLineEdit('Flat')        
        self.edi_blam = QtGui.QLineEdit('Blam')        

        self.edi_bat_start  = QtGui.QLineEdit ( 'Start' )        
        self.edi_bat_end    = QtGui.QLineEdit ( 'End'   )        
        self.edi_bat_total  = QtGui.QLineEdit ( 'Total' )        
        self.edi_bat_time   = QtGui.QLineEdit ( 'Time'  )        

        self.table = QtGui.QTableWidget(1,6,self)
        self.table.setHorizontalHeaderLabels(['#Parts','#Rows', '#Cols', 'Img size', 'Part size', 'Rest'])
        self.table.setVerticalHeaderLabels(['Set:'])

        self.table.horizontalHeader().setDefaultSectionSize(60)
        self.table.horizontalHeader().resizeSection(3,80)
        self.table.horizontalHeader().resizeSection(4,80)


        #self.edi_bat_nparts = QtGui.QLineEdit       (str(cp.bat_img_nparts.value()))        
        #self.item_nparts    = QtGui.QTableWidget(self.edi_bat_nparts)

        self.item_nparts    = QtGui.QTableWidgetItem(str(cp.bat_img_nparts.value()))
        self.item_rows      = QtGui.QTableWidgetItem(str(cp.bat_img_rows.value()))
        self.item_cols      = QtGui.QTableWidgetItem(str(cp.bat_img_cols.value()))
        self.item_size      = QtGui.QTableWidgetItem(str(cp.bat_img_size.value()))
        self.item_part_size = QtGui.QTableWidgetItem('y')
        self.item_rest_size = QtGui.QTableWidgetItem('z')

        item_flags = QtCore.Qt.ItemFlags(QtCore.Qt.NoItemFlags)
        self.item_rows     .setFlags(item_flags)
        self.item_cols     .setFlags(item_flags)
        self.item_size     .setFlags(item_flags)
        self.item_part_size.setFlags(item_flags)
        self.item_rest_size.setFlags(item_flags)

        self.setTableItems()

        self.table.setItem(0, 0, self.item_nparts)
        self.table.setItem(0, 1, self.item_rows)
        self.table.setItem(0, 2, self.item_cols)
        self.table.setItem(0, 3, self.item_size)
        self.table.setItem(0, 4, self.item_part_size)
        self.table.setItem(0, 5, self.item_rest_size)
        #self.table.setCellWidget(0, 3, self.edi_bat_nparts)
        
        self.table.itemClicked.connect(self.onItem)
        self.table.itemChanged.connect(self.onItemChanged)
        #self.connect( self.edi_bat_nparts, QtCore.SIGNAL('editingFinished()'), self.onEdiNParts)

        self.hboxT = QtGui.QHBoxLayout()
        self.hboxT.addWidget(self.table)

        self.hboxS = QtGui.QHBoxLayout()
        self.hboxS.addWidget(self.tit_status)
        self.hboxS.addStretch(1)     
        #self.hboxB.addWidget(self.but_close)
        #self.hboxB.addWidget(self.but_apply)

        self.grid = QtGui.QGridLayout()
        self.grid_row = 1
        self.grid.addWidget(self.tit_title,     self.grid_row,   0, 1, 10)          
        self.grid.addWidget(self.tit_data,      self.grid_row+1, 0)
        self.grid.addWidget(self.edi_data,      self.grid_row+1, 1, 1,  9)

        self.grid.addWidget(self.lab_start,     self.grid_row+2, 1)
        self.grid.addWidget(self.edi_bat_start, self.grid_row+2, 2)
        self.grid.addWidget(self.lab_end,       self.grid_row+2, 3)
        self.grid.addWidget(self.edi_bat_end,   self.grid_row+2, 4)
        self.grid.addWidget(self.lab_total,     self.grid_row+2, 5)
        self.grid.addWidget(self.edi_bat_total, self.grid_row+2, 6)
        self.grid.addWidget(self.lab_time,      self.grid_row+2, 7)
        self.grid.addWidget(self.edi_bat_time , self.grid_row+2, 8, 1,  2)

        self.grid.addWidget(self.tit_dark,      self.grid_row+3, 0)
        self.grid.addWidget(self.edi_dark,      self.grid_row+3, 1, 1,  9)
        self.grid.addWidget(self.tit_flat,      self.grid_row+4, 0)
        self.grid.addWidget(self.edi_flat,      self.grid_row+4, 1, 1,  9)
        self.grid.addWidget(self.tit_blam,      self.grid_row+5, 0)
        self.grid.addWidget(self.edi_blam,      self.grid_row+5, 1, 1,  9)
        #self.grid.addLayout(self.hboxB,         self.grid_row+7, 0, 1, 10)
        self.grid.addLayout(self.hboxT,         self.grid_row+8, 0, 1, 10)
        self.grid.addLayout(self.hboxS,         self.grid_row+10,0, 1, 10)


        self.setLayout(self.grid)
        
        #self.connect( self.but_close, QtCore.SIGNAL('clicked()'), self.onClose  )
        #self.connect( self.but_apply, QtCore.SIGNAL('clicked()'), self.onSave   )

        self.showToolTips()
        self.setStyle()
        self.setFields()
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
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setMinimumSize(740,350)
        self.           setStyleSheet (cp.styleBkgd)
        self.tit_title .setStyleSheet (cp.styleTitleBold)
        self.tit_status.setStyleSheet (cp.styleTitle)
        #self.but_close .setStyleSheet (cp.styleButton)
        #self.but_apply .setStyleSheet (cp.styleButton) 
        #self.but_show  .setStyleSheet (cp.styleButton) 

        self.tit_data  .setStyleSheet (cp.styleLabel)
        self.tit_dark  .setStyleSheet (cp.styleLabel)
        self.tit_flat  .setStyleSheet (cp.styleLabel)
        self.tit_blam  .setStyleSheet (cp.styleLabel)

        self.tit_title.setAlignment(QtCore.Qt.AlignCenter)

        self.lab_start .setStyleSheet (cp.styleLabel)
        self.lab_end   .setStyleSheet (cp.styleLabel)
        self.lab_total .setStyleSheet (cp.styleLabel)
        self.lab_time  .setStyleSheet (cp.styleLabel)

        self.edi_data.setStyleSheet (cp.styleEditInfo)
        self.edi_dark.setStyleSheet (cp.styleEditInfo)
        self.edi_flat.setStyleSheet (cp.styleEditInfo)
        self.edi_blam.setStyleSheet (cp.styleEditInfo)

        width = 60
        self.edi_bat_start.setFixedWidth(width)   
        self.edi_bat_end  .setFixedWidth(width)   
        self.edi_bat_total.setFixedWidth(width)   
        self.edi_bat_time .setFixedWidth(140)   

        self.edi_bat_start.setStyleSheet   (cp.styleEditInfo)
        self.edi_bat_end  .setStyleSheet   (cp.styleEditInfo)
        self.edi_bat_total.setStyleSheet   (cp.styleEditInfo)
        self.edi_bat_time .setStyleSheet   (cp.styleEditInfo)

        self.edi_data.setAlignment    (QtCore.Qt.AlignRight)
        self.edi_dark.setAlignment    (QtCore.Qt.AlignRight)
        self.edi_flat.setAlignment    (QtCore.Qt.AlignRight)
        self.edi_blam.setAlignment    (QtCore.Qt.AlignRight)

        self.tit_title .setAlignment(QtCore.Qt.AlignCenter)
        #self.titTitle .setBold()


    def setFields(self):
        self.edi_data.setText( fnm.path_data_xtc_cond() )        

        if cp.bat_dark_is_used.value() : self.edi_dark.setText( fnm.path_pedestals_ave() )
        else                           : self.edi_dark.setText( 'is not used' )

        if cp.ccdcorr_flatfield.value() : self.edi_flat.setText( fnm.path_flat() )
        else                            : self.edi_flat.setText( 'is not used' )

        if cp.ccdcorr_blemish.value()   : self.edi_blam.setText( fnm.path_blam() )
        else                            : self.edi_blam.setText( 'is not used' )

        self.edi_bat_start.setText ( str( cp.bat_data_start.value() ) )        
        self.edi_bat_end  .setText ( str( cp.bat_data_end  .value() ) )        
        self.edi_bat_total.setText ( str( cp.bat_data_total.value() ) )        
        self.edi_bat_time .setText ( str( cp.bat_data_dt_ave.value() ) + u'\u00B1'
                                     + str( cp.bat_data_dt_rms.value() ) )        
 
        self.edi_bat_start.setReadOnly( True )
        self.edi_bat_end  .setReadOnly( True )
        self.edi_bat_total.setReadOnly( True )
        self.edi_bat_time .setReadOnly( True )

        self.edi_data.setReadOnly( True )   
        self.edi_dark.setReadOnly( True )   
        self.edi_flat.setReadOnly( True )   
        self.edi_blam.setReadOnly( True )   


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

        try    : del cp.guiruninfo # GUIRunInfo
        except : pass


    def onItem(self, item):
        print 'Clicked on item: ', str(item.text()) 

    def onItemChanged(self, item):
        print 'Changed item: '
        if item == self.item_nparts :
            s = str(item.text()) 
            print 'Changed item: ', s
            cp.bat_img_nparts.setValue(s)
            self.setTableItems()
        #else :
        #    print 'Changed non-allowed item: ',  str(item.text())  

    def onEdiNParts(self) :
        s = str(self.edi_bat_nparts.text()) 
        print 'onEdiNParts: ', s
        cp.bat_img_nparts.setValue(s)
        self.setTableItems()


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
        self.item_nparts   .setBackgroundColor (cp.colorEdit)
        self.item_part_size.setBackgroundColor (cp.colorEditInfo)

        if rest_size == 0 :
            self.item_rest_size.setBackgroundColor (cp.colorEditInfo)
        else :
            self.item_rest_size.setBackgroundColor (cp.colorEditBad)
            #self.item_nparts   .setBackgroundColor (cp.colorEditBad)


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

        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0 : self.tit_status.setStyleSheet(cp.styleStatusGood)
        if status_index == 1 : self.tit_status.setStyleSheet(cp.styleStatusWarning)
        if status_index == 2 : self.tit_status.setStyleSheet(cp.styleStatusAlarm)

        self.tit_status.setText('Status: ' + list_of_states[status_index] + msg)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIRunInfo ()
    widget.show()
    app.exec_()

#-----------------------------
