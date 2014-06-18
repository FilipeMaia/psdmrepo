#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISetupPars...
#
#------------------------------------------------------------------------

"""GUI sets setup parameters"""

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

#---------------------
#  Class definition --
#---------------------
class GUISetupPars ( QtGui.QWidget ) :
    """GUI sets setup parameters"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Setup Parameters')
        self.setFrame()

        #self.tit_bat_num_max     = QtGui.QLabel('Batches To Be Analysied:')
        #self.tit_bat_num         = QtGui.QLabel('View and Edit Batch #')
        self.tit_bat             = QtGui.QLabel('Setup Parameters:')
        self.tit_bat_start       = QtGui.QLabel('start')
        self.tit_bat_end         = QtGui.QLabel('end')
        self.tit_bat_total       = QtGui.QLabel('total')
        self.tit_bat_time        = QtGui.QLabel(u'\u0394t(sec)')
        self.tit_bat_data        = QtGui.QLabel('data')
        self.tit_bat_dark        = QtGui.QLabel('dark')
        self.tit_bat_flat        = QtGui.QLabel('flat')

        #self.but_bat_num         = QtGui.QPushButton(str( cp.bat_num       .value() ) + self.char_expand  ) 
        #self.edi_bat_num_max     = QtGui.QLineEdit ( str( cp.bat_num_max   .value() ) )        
        self.edi_bat_data_start  = QtGui.QLineEdit ( str( cp.bat_data_start.value() ) )        
        self.edi_bat_data_end    = QtGui.QLineEdit ( str( cp.bat_data_end  .value() ) )        
        self.edi_bat_data_total  = QtGui.QLineEdit ( str( cp.bat_data_total.value() ) )        
        self.edi_bat_data_time   = QtGui.QLineEdit ( str( cp.bat_data_time .value() ) )        
        self.edi_bat_dark_start  = QtGui.QLineEdit ( str( cp.bat_dark_start.value() ) )        
        self.edi_bat_dark_end    = QtGui.QLineEdit ( str( cp.bat_dark_end  .value() ) )        
        self.edi_bat_dark_total  = QtGui.QLineEdit ( str( cp.bat_dark_total.value() ) )        
        self.edi_bat_dark_time   = QtGui.QLineEdit ( str( cp.bat_dark_time .value() ) )        
        self.edi_bat_flat_start  = QtGui.QLineEdit ( str( cp.bat_flat_start.value() ) )        
        self.edi_bat_flat_end    = QtGui.QLineEdit ( str( cp.bat_flat_end  .value() ) )        
        self.edi_bat_flat_total  = QtGui.QLineEdit ( str( cp.bat_flat_total.value() ) )        
        self.edi_bat_flat_time   = QtGui.QLineEdit ( str( cp.bat_flat_time .value() ) )        


        self.edi_bat_data_time.setReadOnly( True ) 
        self.edi_bat_dark_time.setReadOnly( True ) 
        self.edi_bat_flat_time.setReadOnly( True ) 

        self.grid = QtGui.QGridLayout()
        #self.grid.addWidget(self.tit_bat_num_max,                0, 0, 1, 4)
        #self.grid.addWidget(self.tit_bat_num    ,                1, 0, 1, 4)
        self.grid.addWidget(self.tit_bat        ,                0, 0, 1, 6)
        self.grid.addWidget(self.tit_bat_start  ,                2, 3)
        self.grid.addWidget(self.tit_bat_end    ,                2, 4)
        self.grid.addWidget(self.tit_bat_total  ,                2, 5)
        self.grid.addWidget(self.tit_bat_time   ,                2, 6)
        self.grid.addWidget(self.tit_bat_data   ,                3, 1)
        self.grid.addWidget(self.tit_bat_dark   ,                4, 1)
        self.grid.addWidget(self.tit_bat_flat   ,                5, 1)

        #self.grid.addWidget(self.edi_bat_num_max   ,             0, 4)
        #self.grid.addWidget(self.but_bat_num       ,             1, 4)
        self.grid.addWidget(self.edi_bat_data_start,             3, 3)
        self.grid.addWidget(self.edi_bat_data_end  ,             3, 4)
        self.grid.addWidget(self.edi_bat_data_total,             3, 5)
        self.grid.addWidget(self.edi_bat_data_time ,             3, 6)
        self.grid.addWidget(self.edi_bat_dark_start,             4, 3)
        self.grid.addWidget(self.edi_bat_dark_end  ,             4, 4)
        self.grid.addWidget(self.edi_bat_dark_total,             4, 5)
        self.grid.addWidget(self.edi_bat_dark_time ,             4, 6)
        self.grid.addWidget(self.edi_bat_flat_start,             5, 3)
        self.grid.addWidget(self.edi_bat_flat_end  ,             5, 4)
        self.grid.addWidget(self.edi_bat_flat_total,             5, 5)
        self.grid.addWidget(self.edi_bat_flat_time ,             5, 6)

        self.setLayout(self.grid)

        #self.connect( self.but_bat_num        ,     QtCore.SIGNAL('clicked()'),          self.on_but_bat_num        )
        #self.connect( self.edi_bat_num_max    ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_num_max    )
        self.connect( self.edi_bat_data_start ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_data_start )
        self.connect( self.edi_bat_data_end   ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_data_end   )
        self.connect( self.edi_bat_dark_start ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_dark_start )
        self.connect( self.edi_bat_dark_end   ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_dark_end   )
        self.connect( self.edi_bat_flat_start ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_flat_start )
        self.connect( self.edi_bat_flat_end   ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_flat_end   )
  
        self.showToolTips()
        self.setStyle()
        self.set_fields()
        
    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg_edit = 'Edit field'
        msg_info = 'Information field'
        msg_sele = 'Selection field'
        
        #self.but_bat_num       .setToolTip( msg_sele )
        #self.edi_bat_num_max   .setToolTip( msg_edit )        
        self.edi_bat_data_start.setToolTip( msg_edit )        
        self.edi_bat_dark_start.setToolTip( msg_edit )        
        self.edi_bat_flat_start.setToolTip( msg_edit )        
        self.edi_bat_data_end  .setToolTip( msg_edit )        
        self.edi_bat_dark_end  .setToolTip( msg_edit )        
        self.edi_bat_flat_end  .setToolTip( msg_edit )        
        self.edi_bat_data_time .setToolTip( msg_info )
        self.edi_bat_dark_time .setToolTip( msg_info )
        self.edi_bat_flat_time .setToolTip( msg_info )
        self.edi_bat_data_total.setToolTip( msg_info )
        self.edi_bat_dark_total.setToolTip( msg_info )
        self.edi_bat_flat_total.setToolTip( msg_info )

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):
        width = 50

        self.                setStyleSheet (cp.styleBkgd)
        #self.tit_bat_num_max.setStyleSheet (cp.styleTitle)
        #self.tit_bat_num    .setStyleSheet (cp.styleTitle)
        self.tit_bat        .setStyleSheet (cp.styleTitle)
        self.tit_bat_start  .setStyleSheet (cp.styleLabel)
        self.tit_bat_end    .setStyleSheet (cp.styleLabel)
        self.tit_bat_total  .setStyleSheet (cp.styleLabel)
        self.tit_bat_time   .setStyleSheet (cp.styleLabel)
        self.tit_bat_data   .setStyleSheet (cp.styleLabel)
        self.tit_bat_dark   .setStyleSheet (cp.styleLabel)
        self.tit_bat_flat   .setStyleSheet (cp.styleLabel)

        #self.tit_bat_num_max.setAlignment(QtCore.Qt.AlignRight)
        #self.tit_bat_num    .setAlignment(QtCore.Qt.AlignRight)
        self.tit_bat        .setAlignment(QtCore.Qt.AlignLeft)
        self.tit_bat_start  .setAlignment(QtCore.Qt.AlignCenter)
        self.tit_bat_end    .setAlignment(QtCore.Qt.AlignCenter)
        self.tit_bat_total  .setAlignment(QtCore.Qt.AlignCenter)
        self.tit_bat_time   .setAlignment(QtCore.Qt.AlignCenter)
        self.tit_bat_data   .setAlignment(QtCore.Qt.AlignRight)
        self.tit_bat_dark   .setAlignment(QtCore.Qt.AlignRight)
        self.tit_bat_flat   .setAlignment(QtCore.Qt.AlignRight)

        #self.edi_bat_num_max   .setFixedWidth(60)
        #self.but_bat_num       .setFixedWidth(60)
        self.edi_bat_data_start.setFixedWidth(width)
        self.edi_bat_dark_start.setFixedWidth(width)
        self.edi_bat_flat_start.setFixedWidth(width)
        self.edi_bat_data_end  .setFixedWidth(width)
        self.edi_bat_dark_end  .setFixedWidth(width)
        self.edi_bat_flat_end  .setFixedWidth(width)
        self.edi_bat_data_total.setFixedWidth(width)
        self.edi_bat_dark_total.setFixedWidth(width)
        self.edi_bat_flat_total.setFixedWidth(width)
        self.edi_bat_data_time .setFixedWidth(140)
        self.edi_bat_dark_time .setFixedWidth(140)
        self.edi_bat_flat_time .setFixedWidth(140)
                               
        #self.edi_bat_num_max   .setStyleSheet(cp.styleEdit)
        #self.but_bat_num       .setStyleSheet(cp.styleButton)
        self.edi_bat_data_start.setStyleSheet(cp.styleEdit)
        self.edi_bat_dark_start.setStyleSheet(cp.styleEdit)
        self.edi_bat_flat_start.setStyleSheet(cp.styleEdit)
        self.edi_bat_data_end  .setStyleSheet(cp.styleEdit)
        self.edi_bat_dark_end  .setStyleSheet(cp.styleEdit)
        self.edi_bat_flat_end  .setStyleSheet(cp.styleEdit)
        self.edi_bat_data_total.setStyleSheet(cp.styleEditInfo)
        self.edi_bat_dark_total.setStyleSheet(cp.styleEditInfo)
        self.edi_bat_flat_total.setStyleSheet(cp.styleEditInfo)
        self.edi_bat_data_time .setStyleSheet(cp.styleEditInfo)
        self.edi_bat_dark_time .setStyleSheet(cp.styleEditInfo)
        self.edi_bat_flat_time .setStyleSheet(cp.styleEditInfo)

        #self.edi_bat_num_max   .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_data_start.setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_dark_start.setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_flat_start.setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_data_end  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_dark_end  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_flat_end  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_data_total.setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_dark_total.setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_flat_total.setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_data_time .setAlignment(QtCore.Qt.AlignLeading)
        self.edi_bat_dark_time .setAlignment(QtCore.Qt.AlignLeading)
        self.edi_bat_flat_time .setAlignment(QtCore.Qt.AlignLeading)


    def set_fields(self):
        self.edi_bat_data_start.setText( str( cp.bat_data_start.value() ) )        
        self.edi_bat_data_end  .setText( str( cp.bat_data_end  .value() ) )        
        self.edi_bat_data_total.setText( str( cp.bat_data_total.value() ) )        
        self.edi_bat_data_time .setText( str( cp.bat_data_dt_ave.value() ) + u'\u00B1' + str( cp.bat_data_dt_rms.value() ) )        

        self.edi_bat_dark_start.setText( str( cp.bat_dark_start.value() ) )        
        self.edi_bat_dark_end  .setText( str( cp.bat_dark_end  .value() ) )        
        self.edi_bat_dark_total.setText( str( cp.bat_dark_total.value() ) )        
        self.edi_bat_dark_time .setText( str( cp.bat_dark_dt_ave.value() ) + u'\u00B1' + str( cp.bat_dark_dt_rms.value() ) )        


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
        try    : del cp.guisetuppars # GUISetupPars
        except : pass

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def on_edi_bat_num_max(self):
        cp.bat_num_max.setValue( int(self.edi_bat_num_max.displayText()) )
        logger.info('Set bat_num_max =' + str(cp.bat_num_max.value()), __name__)

    def on_edi_bat_data_start(self):
        cp.bat_data_start.setValue( int(self.edi_bat_data_start.displayText()) )
        logger.info('Set bat_data_start =' + str(cp.bat_data_start.value()), __name__)

    def on_edi_bat_data_end(self):
        cp.bat_data_end.setValue( int(self.edi_bat_data_end.displayText()) )
        logger.info('Set bat_data_end =' + str(cp.bat_data_end.value()), __name__)

    def on_edi_bat_dark_start(self):
        cp.bat_dark_start.setValue( int(self.edi_bat_dark_start.displayText()) )
        logger.info('Set bat_dark_start =' + str(cp.bat_dark_start.value()), __name__)

    def on_edi_bat_dark_end(self):
        cp.bat_dark_end.setValue( int(self.edi_bat_dark_end.displayText()) )
        logger.info('Set bat_dark_end =' + str(cp.bat_dark_end.value()), __name__)

    def on_edi_bat_flat_start(self):
        cp.bat_flat_start.setValue( int(self.edi_bat_flat_start.displayText()) )
        logger.info('Set bat_flat_start =' + str(cp.bat_flat_start.value()), __name__)

    def on_edi_bat_flat_end(self):
        cp.bat_flat_end.setValue( int(self.edi_bat_flat_end.displayText()) )
        logger.info('Set bat_flat_end =' + str(cp.bat_flat_end.value()), __name__)

    def setPopupMenuMode(self):
        self.list_of_nums = range(1,cp.bat_num_max.value()+1)
        self.popupMenuMode = QtGui.QMenu()
        for num in self.list_of_nums :
            self.popupMenuMode.addAction( str(num) )

    def on_but_bat_num(self):
        self.setPopupMenuMode()
        action_selected = self.popupMenuMode.exec_(QtGui.QCursor.pos())
        if action_selected is None : return
        str_selected = action_selected.text()
        cp.bat_num.setValue( str_selected )
        self.but_bat_num.setText( str_selected + self.char_expand )
        logger.info(' ---> selected batch number: ' + str_selected, __name__)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUISetupPars ()
    widget.show()
    app.exec_()

#-----------------------------
