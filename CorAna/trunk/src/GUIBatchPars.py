#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIBatchPars...
#
#------------------------------------------------------------------------

"""GUI sets parameters for batch mode"""

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

#---------------------
#  Class definition --
#---------------------
class GUIBatchPars ( QtGui.QWidget ) :
    """GUI sets parameters for batch mode"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Batch Parameters')
        self.setFrame()

        self.char_expand         = u' \u25BE' # down-head triangle
 
        self.tit_bat_num_max     = QtGui.QLabel('Batches To Be Analysied:')
        self.tit_bat_num         = QtGui.QLabel('View and Edit Batch #')
        self.tit_bat_start       = QtGui.QLabel('start')
        self.tit_bat_end         = QtGui.QLabel('end')
        self.tit_bat_time        = QtGui.QLabel('time (sec)')
        self.tit_bat_data        = QtGui.QLabel('data')
        self.tit_bat_dark        = QtGui.QLabel('dark')
        self.tit_bat_flat        = QtGui.QLabel('flat')
        self.tit_bat_flux        = QtGui.QLabel('flux')
        self.tit_bat_current     = QtGui.QLabel('current')
        self.tit_bat_photons     = QtGui.QLabel('photons/sec')
        self.tit_bat_ma          = QtGui.QLabel('mA')

        self.but_bat_num         = QtGui.QPushButton(str( cp.bat_num       .value() ) + self.char_expand  ) 
        self.edi_bat_num_max     = QtGui.QLineEdit ( str( cp.bat_num_max   .value() ) )        
        self.edi_bat_data_start  = QtGui.QLineEdit ( str( cp.bat_data_start.value() ) )        
        self.edi_bat_data_end    = QtGui.QLineEdit ( str( cp.bat_data_end  .value() ) )        
        self.edi_bat_data_time   = QtGui.QLineEdit ( str( cp.bat_data_time .value() ) )        
        self.edi_bat_dark_start  = QtGui.QLineEdit ( str( cp.bat_dark_start.value() ) )        
        self.edi_bat_dark_end    = QtGui.QLineEdit ( str( cp.bat_dark_end  .value() ) )        
        self.edi_bat_dark_time   = QtGui.QLineEdit ( str( cp.bat_dark_time .value() ) )        
        self.edi_bat_flat_start  = QtGui.QLineEdit ( str( cp.bat_flat_start.value() ) )        
        self.edi_bat_flat_end    = QtGui.QLineEdit ( str( cp.bat_flat_end  .value() ) )        
        self.edi_bat_flat_time   = QtGui.QLineEdit ( str( cp.bat_flat_time .value() ) )        
        self.edi_bat_flux        = QtGui.QLineEdit ( str( cp.bat_flux      .value() ) )        
        self.edi_bat_current     = QtGui.QLineEdit ( str( cp.bat_current   .value() ) )        

        self.edi_bat_data_time.setReadOnly( True ) 
        self.edi_bat_dark_time.setReadOnly( True ) 
        self.edi_bat_flat_time.setReadOnly( True ) 
        self.edi_bat_flux     .setReadOnly( True ) 
        self.edi_bat_current  .setReadOnly( True ) 

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.tit_bat_num_max,                0, 0, 1, 4)
        self.grid.addWidget(self.tit_bat_num    ,                1, 0, 1, 4)
        self.grid.addWidget(self.tit_bat_start  ,                2, 3)
        self.grid.addWidget(self.tit_bat_end    ,                2, 4)
        self.grid.addWidget(self.tit_bat_time   ,                2, 5)
        self.grid.addWidget(self.tit_bat_data   ,                3, 1)
        self.grid.addWidget(self.tit_bat_dark   ,                4, 1)
        self.grid.addWidget(self.tit_bat_flat   ,                5, 1)
        self.grid.addWidget(self.tit_bat_flux   ,                6, 1)
        self.grid.addWidget(self.tit_bat_current,                7, 1)
        self.grid.addWidget(self.tit_bat_photons,                6, 5)
        self.grid.addWidget(self.tit_bat_ma     ,                7, 5)

        self.grid.addWidget(self.edi_bat_num_max   ,             0, 4)
        self.grid.addWidget(self.but_bat_num       ,             1, 4)
        self.grid.addWidget(self.edi_bat_data_start,             3, 3)
        self.grid.addWidget(self.edi_bat_data_end  ,             3, 4)
        self.grid.addWidget(self.edi_bat_data_time ,             3, 5)
        self.grid.addWidget(self.edi_bat_dark_start,             4, 3)
        self.grid.addWidget(self.edi_bat_dark_end  ,             4, 4)
        self.grid.addWidget(self.edi_bat_dark_time ,             4, 5)
        self.grid.addWidget(self.edi_bat_flat_start,             5, 3)
        self.grid.addWidget(self.edi_bat_flat_end  ,             5, 4)
        self.grid.addWidget(self.edi_bat_flat_time ,             5, 5)
        self.grid.addWidget(self.edi_bat_flux      ,             6, 3, 1, 2)
        self.grid.addWidget(self.edi_bat_current   ,             7, 3, 1, 2)

        self.setLayout(self.grid)

        self.connect( self.but_bat_num        ,     QtCore.SIGNAL('clicked()'),          self.on_but_bat_num        )
        self.connect( self.edi_bat_num_max    ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_num_max    )
        self.connect( self.edi_bat_data_start ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_data_start )
        self.connect( self.edi_bat_data_end   ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_data_end   )
        self.connect( self.edi_bat_dark_start ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_dark_start )
        self.connect( self.edi_bat_dark_end   ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_dark_end   )
        self.connect( self.edi_bat_flat_start ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_flat_start )
        self.connect( self.edi_bat_flat_end   ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_bat_flat_end   )
 
        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg_edit = 'Edit field'
        msg_info = 'Information field'
        msg_sele = 'Selection field'
        
        self.but_bat_num       .setToolTip( msg_sele )
        self.edi_bat_num_max   .setToolTip( msg_edit )        
        self.edi_bat_data_start.setToolTip( msg_edit )        
        self.edi_bat_data_end  .setToolTip( msg_edit )        
        self.edi_bat_data_time .setToolTip( msg_edit )        
        self.edi_bat_dark_start.setToolTip( msg_edit )        
        self.edi_bat_dark_end  .setToolTip( msg_edit )        
        self.edi_bat_dark_time .setToolTip( msg_edit )        
        self.edi_bat_flat_start.setToolTip( msg_edit )        
        self.edi_bat_flat_end  .setToolTip( msg_edit )        
        self.edi_bat_data_time .setToolTip( msg_info )
        self.edi_bat_dark_time .setToolTip( msg_info )
        self.edi_bat_flat_time .setToolTip( msg_info )
        self.edi_bat_flux      .setToolTip( msg_info )
        self.edi_bat_current   .setToolTip( msg_info )

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):
        width = 80

        self.                setStyleSheet (cp.styleBkgd)
        self.tit_bat_num_max.setStyleSheet (cp.styleTitle)
        self.tit_bat_num    .setStyleSheet (cp.styleTitle)
        self.tit_bat_start  .setStyleSheet (cp.styleLabel)
        self.tit_bat_end    .setStyleSheet (cp.styleLabel)
        self.tit_bat_time   .setStyleSheet (cp.styleLabel)
        self.tit_bat_data   .setStyleSheet (cp.styleLabel)
        self.tit_bat_dark   .setStyleSheet (cp.styleLabel)
        self.tit_bat_flat   .setStyleSheet (cp.styleLabel)
        self.tit_bat_flux   .setStyleSheet (cp.styleLabel)
        self.tit_bat_current.setStyleSheet (cp.styleLabel)
        self.tit_bat_photons.setStyleSheet (cp.styleLabel)
        self.tit_bat_ma     .setStyleSheet (cp.styleLabel)


        #self.tit_bat_num_max.setAlignment(QtCore.Qt.AlignRight)
        #self.tit_bat_num    .setAlignment(QtCore.Qt.AlignRight)
        self.tit_bat_start  .setAlignment(QtCore.Qt.AlignCenter)
        self.tit_bat_end    .setAlignment(QtCore.Qt.AlignCenter)
        self.tit_bat_time   .setAlignment(QtCore.Qt.AlignCenter)
        self.tit_bat_data   .setAlignment(QtCore.Qt.AlignRight)
        self.tit_bat_dark   .setAlignment(QtCore.Qt.AlignRight)
        self.tit_bat_flat   .setAlignment(QtCore.Qt.AlignRight)
        self.tit_bat_flux   .setAlignment(QtCore.Qt.AlignRight)
        self.tit_bat_current.setAlignment(QtCore.Qt.AlignRight)
        self.tit_bat_photons.setAlignment(QtCore.Qt.AlignLeft)
        self.tit_bat_ma     .setAlignment(QtCore.Qt.AlignLeft)

        self.edi_bat_num_max   .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_data_start.setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_data_end  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_data_time .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_dark_start.setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_dark_end  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_dark_time .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_flat_start.setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_flat_end  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_flat_time .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_flux      .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_current   .setAlignment(QtCore.Qt.AlignRight)

        self.edi_bat_num_max   .setFixedWidth(60)
        self.but_bat_num       .setFixedWidth(60)
        self.edi_bat_data_start.setFixedWidth(width)
        self.edi_bat_data_end  .setFixedWidth(width)
        self.edi_bat_data_time .setFixedWidth(width)
        self.edi_bat_dark_start.setFixedWidth(width)
        self.edi_bat_dark_end  .setFixedWidth(width)
        self.edi_bat_dark_time .setFixedWidth(width)
        self.edi_bat_flat_start.setFixedWidth(width)
        self.edi_bat_flat_end  .setFixedWidth(width)
        self.edi_bat_flat_time .setFixedWidth(width)
        #self.edi_bat_flux      .setFixedWidth(width)
        #self.edi_bat_current   .setFixedWidth(width)
                               
        self.edi_bat_num_max   .setStyleSheet(cp.styleEdit)
        self.but_bat_num       .setStyleSheet(cp.styleButton)
        self.edi_bat_data_start.setStyleSheet(cp.styleEdit)
        self.edi_bat_data_end  .setStyleSheet(cp.styleEdit)
        self.edi_bat_data_time .setStyleSheet(cp.styleEditInfo)
        self.edi_bat_dark_start.setStyleSheet(cp.styleEdit)
        self.edi_bat_dark_end  .setStyleSheet(cp.styleEdit)
        self.edi_bat_dark_time .setStyleSheet(cp.styleEditInfo)
        self.edi_bat_flat_start.setStyleSheet(cp.styleEdit)
        self.edi_bat_flat_end  .setStyleSheet(cp.styleEdit)
        self.edi_bat_flat_time .setStyleSheet(cp.styleEditInfo)
        self.edi_bat_flux      .setStyleSheet(cp.styleEditInfo)
        self.edi_bat_current   .setStyleSheet(cp.styleEditInfo)

    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp
            del cp.guibatchpars # GUIBatchPars
        except AttributeError:
            pass # silently ignore

    def processClose(self):
        #print 'Close button'
        self.close()

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #print 'moveEvent' 
        pass
#        cp.posGUIMain = (self.pos().x(),self.pos().y())

    def on_edi_bat_num_max(self):
        cp.bat_num_max.setValue( int(self.edi_bat_num_max.displayText()) )
        print 'Set bat_num_max =', cp.bat_num_max.value()

    def on_edi_bat_data_start(self):
        cp.bat_data_start.setValue( float(self.edi_bat_data_start.displayText()) )
        print 'Set bat_data_start =', cp.bat_data_start.value()

    def on_edi_bat_data_end(self):
        cp.bat_data_end.setValue( float(self.edi_bat_data_end.displayText()) )
        print 'Set bat_data_end =', cp.bat_data_end.value()

    def on_edi_bat_dark_start(self):
        cp.bat_dark_start.setValue( float(self.edi_bat_dark_start.displayText()) )
        print 'Set bat_dark_start =', cp.bat_dark_start.value()

    def on_edi_bat_dark_end(self):
        cp.bat_dark_end.setValue( float(self.edi_bat_dark_end.displayText()) )
        print 'Set bat_dark_end =', cp.bat_dark_end.value()

    def on_edi_bat_flat_start(self):
        cp.bat_flat_start.setValue( float(self.edi_bat_flat_start.displayText()) )
        print 'Set bat_flat_start =', cp.bat_flat_start.value()

    def on_edi_bat_flat_end(self):
        cp.bat_flat_end.setValue( float(self.edi_bat_flat_end.displayText()) )
        print 'Set bat_flat_end =', cp.bat_flat_end.value()

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
        print ' ---> selected batch number: ' + str_selected

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIBatchPars ()
    widget.show()
    app.exec_()

#-----------------------------
