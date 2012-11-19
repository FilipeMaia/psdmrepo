#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIDark...
#
#------------------------------------------------------------------------

"""GUI works with dark run"""

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
from BatchJobPedestals      import bjpeds
from ImgSpeWithGUI          import *
from BatchLogParser         import blp
#import GlobalGraphics       as gg

#---------------------
#  Class definition --
#---------------------
class GUIDark ( QtGui.QWidget ) :
    """GUI works with dark run"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Dark run processing')
        self.setFrame()

        self.lab_status  = QtGui.QLabel('Status')
        self.lab_batch   = QtGui.QLabel('Batch')
        self.lab_start   = QtGui.QLabel('Start')
        self.lab_end     = QtGui.QLabel('End')
        self.lab_total   = QtGui.QLabel('Total')
        self.lab_time    = QtGui.QLabel('Time(sec)')

        self.edi_path    = QtGui.QLineEdit( fnm.path_dark_xtc() )        
        self.edi_path.setReadOnly( True )  

        self.edi_bat_start  = QtGui.QLineEdit ( str( cp.bat_dark_start.value() ) )        
        self.edi_bat_end    = QtGui.QLineEdit ( str( cp.bat_dark_end  .value() ) )        
        self.edi_bat_total  = QtGui.QLineEdit ( str( cp.bat_dark_total.value() ) )        
        self.edi_bat_time   = QtGui.QLineEdit ( str( cp.bat_dark_time .value() ) )        
 
        self.but_path    = QtGui.QPushButton('File:')
        self.but_status  = QtGui.QPushButton('Check status')
        self.but_wfiles  = QtGui.QPushButton('Check files')
        self.but_submit  = QtGui.QPushButton('Submit')
        self.but_scanner = QtGui.QPushButton('Scanner')
        self.but_plot    = QtGui.QPushButton('Plot')
        self.but_remove  = QtGui.QPushButton('Remove')

        self.grid = QtGui.QGridLayout()
        self.grid_row = 1
        #self.grid.addWidget(self.tit_path,     self.grid_row,   0)
        self.grid.addWidget(self.but_path,      self.grid_row,   0)
        self.grid.addWidget(self.edi_path,      self.grid_row,   1, 1, 6)
        self.grid.addWidget(self.lab_batch,     self.grid_row+1, 0)
        self.grid.addWidget(self.lab_status,    self.grid_row+1, 1, 1, 2)
        self.grid.addWidget(self.lab_start,     self.grid_row+1, 3)
        self.grid.addWidget(self.lab_end,       self.grid_row+1, 4)
        self.grid.addWidget(self.lab_total,     self.grid_row+1, 5)
        self.grid.addWidget(self.lab_time,      self.grid_row+1, 6)
        self.grid.addWidget(self.but_scanner,   self.grid_row+2, 0)
        self.grid.addWidget(self.but_status,    self.grid_row+2, 1, 1, 2)
        self.grid.addWidget(self.edi_bat_start, self.grid_row+2, 3)
        self.grid.addWidget(self.edi_bat_end,   self.grid_row+2, 4)
        self.grid.addWidget(self.edi_bat_total, self.grid_row+2, 5)
        self.grid.addWidget(self.edi_bat_time,  self.grid_row+2, 6)
        self.grid.addWidget(self.but_submit,    self.grid_row+3, 0)
        self.grid.addWidget(self.but_wfiles,    self.grid_row+3, 1, 1, 2)
        self.grid.addWidget(self.but_plot,      self.grid_row+3, 3)
        self.grid.addWidget(self.but_remove,    self.grid_row+3, 6)
        self.grid_row += 3

        self.connect(self.but_path,      QtCore.SIGNAL('clicked()'),          self.on_but_path      )
        self.connect(self.but_status,    QtCore.SIGNAL('clicked()'),          self.on_but_status    )
        self.connect(self.but_submit,    QtCore.SIGNAL('clicked()'),          self.on_but_submit    )
        self.connect(self.but_scanner,   QtCore.SIGNAL('clicked()'),          self.on_but_scanner   )
        self.connect(self.but_wfiles,    QtCore.SIGNAL('clicked()'),          self.on_but_wfiles    )
        self.connect(self.but_plot,      QtCore.SIGNAL('clicked()'),          self.on_but_plot      )
        self.connect(self.but_remove,    QtCore.SIGNAL('clicked()'),          self.on_but_remove    )
        self.connect(self.edi_bat_start, QtCore.SIGNAL('editingFinished()'),  self.on_edi_bat_start )
        self.connect(self.edi_bat_end,   QtCore.SIGNAL('editingFinished()'),  self.on_edi_bat_end   )


        #self.connect(edi, QtCore.SIGNAL('editingFinished()'),        self.onEdit )
        #self.connect(box, QtCore.SIGNAL('currentIndexChanged(int)'), self.onBox  )

        self.setLayout(self.grid)

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        #self           .setToolTip('Use this GUI to work with xtc file.')
        self.edi_path   .setToolTip('The path to the xtc file for processing in this GUI')
        self.but_status .setToolTip('Print batch job status \nin the logger')
        self.but_submit .setToolTip('Submit job in batch for pedestals')
        self.but_scanner.setToolTip('Submit job in batch for scanner')
        self.but_wfiles .setToolTip('List pedestal work files \nand check their availability')
        self.but_plot   .setToolTip('Plot image and spectrum for pedestals')
        self.but_remove .setToolTip('Remove all pedestal work\nfiles for selected run')
        
    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        width = 60
        self.setMinimumWidth(400)
        self.setStyleSheet(cp.styleBkgd)
        #tit0   .setStyleSheet (cp.styleTitle)
        self.lab_status.setStyleSheet (cp.styleLabel)
        self.lab_batch .setStyleSheet (cp.styleLabel)
        self.lab_start .setStyleSheet (cp.styleLabel)
        self.lab_end   .setStyleSheet (cp.styleLabel)
        self.lab_total .setStyleSheet (cp.styleLabel)
        self.lab_time  .setStyleSheet (cp.styleLabel)

        self.edi_path   .setStyleSheet (cp.styleEditInfo) # cp.styleEditInfo
        self.edi_path   .setAlignment  (QtCore.Qt.AlignRight)

        self.edi_bat_start.setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_end  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_total.setAlignment(QtCore.Qt.AlignRight)
        self.edi_bat_time .setAlignment(QtCore.Qt.AlignRight)

        self.edi_bat_start.setFixedWidth(width)
        self.edi_bat_end  .setFixedWidth(width)
        self.edi_bat_total.setFixedWidth(width)
        self.edi_bat_time .setFixedWidth(width)

        self.edi_bat_start.setStyleSheet(cp.styleEdit)
        self.edi_bat_end  .setStyleSheet(cp.styleEdit)
        self.edi_bat_total.setStyleSheet(cp.styleEditInfo)
        self.edi_bat_time .setStyleSheet(cp.styleEditInfo)

        self.edi_bat_total.setReadOnly( True ) 
        self.edi_bat_time .setReadOnly( True ) 

        self.but_path   .setStyleSheet (cp.styleButton)
        self.but_status .setStyleSheet (cp.styleButton)
        self.but_submit .setStyleSheet (cp.styleButton) 
        self.but_scanner.setStyleSheet (cp.styleButton) 
        self.but_wfiles .setStyleSheet (cp.styleButtonOn) 
        self.but_plot   .setStyleSheet (cp.styleButton) 
        self.but_remove .setStyleSheet (cp.styleButtonBad) 
  
        self.but_path   .setFixedWidth(40)
        self.but_submit .setFixedWidth(width)
        self.but_scanner.setFixedWidth(width)
        self.but_plot   .setFixedWidth(width)
        self.but_remove .setFixedWidth(width)
        #self.but_wfiles .setFixedWidth(width)
        #self.but_status .setFixedWidth(width)

        self.on_but_status()
    
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

        try    : cp.imgspewithgui.close()
        except : pass

        try    : del cp.imgspewithgui
        except : pass

        try    : del cp.guidark # GUIDark
        except : pass # silently ignore


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def on_but_path(self):
        logger.debug('Dark file browser', __name__ )
        path = str(self.edi_path.text())        
        path  = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',path) )
        dname, fname = os.path.split(path)

        if dname == '' or fname == '' :
            logger.info('Input directiry name or file name is empty... keep file path unchanged...')
            return

        self.edi_path.setText(path)
        cp.in_dir_dark .setValue(dname)
        cp.in_file_dark.setValue(fname)
        logger.info('selected file: ' + str(fnm.path_dark_xtc()), __name__ )


    def on_but_status(self):
        logger.debug('on_but_status - not implemented yet...', __name__)
        if bjpeds.status_for_pedestals() : self.but_status.setStyleSheet(cp.styleButtonGood)
        else                             : self.but_status.setStyleSheet(cp.styleButtonBad)
        bjpeds.check_batch_status_for_pedestals_tahometer()
        bjpeds.check_batch_status_for_pedestals()
        blp.parse_batch_log_pedestals_tahometer()
        self.set_fields()


    def set_fields(self):
        self.edi_bat_start.setText( str( cp.bat_dark_start.value() ) )        
        self.edi_bat_end  .setText( str( cp.bat_dark_end  .value() ) )        
        self.edi_bat_total.setText( str( cp.bat_dark_total.value() ) )        
        self.edi_bat_time .setText( str( cp.bat_dark_time .value() ) )        

    def on_but_submit(self):
        logger.debug('on_but_submit', __name__)
        bjpeds.submit_batch_for_pedestals()

    def on_but_scanner(self):
        logger.debug('on_but_scanner', __name__)
        bjpeds.submit_batch_for_tahometer()

    def on_but_wfiles(self):
        logger.debug('on_but_wfiles', __name__)
        #bjpeds.print_work_files_for_pedestals()
        bjpeds.check_work_files_for_pedestals()

    def on_edi_bat_start(self):
        cp.bat_dark_start.setValue( int(self.edi_bat_start.displayText()) )
        logger.info('Set bat_dark_start =' + str(cp.bat_dark_start.value()), __name__)

    def on_edi_bat_end(self):
        cp.bat_dark_end.setValue( int(self.edi_bat_end.displayText()) )
        logger.info('Set bat_dark_end =' + str(cp.bat_dark_end.value()), __name__)

    def on_but_plot(self):
        logger.debug('on_but_plot', __name__)
        try :
            cp.imgspewithgui.close()
            del cp.imgspewithgui
            #but.setStyleSheet(cp.styleButtonBad)
        except :
            arr = bjpeds.get_pedestals_from_file()
            if arr == None : return
            #print arr.shape,'\n', arr
            cp.imgspewithgui = ImgSpeWithGUI(None, arr)
            #cp.imgspewithgui.setParent(self)
            cp.imgspewithgui.move(self.parentWidget().pos().__add__(QtCore.QPoint(400,20)))
            cp.imgspewithgui.show()
            #but.setStyleSheet(cp.styleButtonGood)

    def on_but_remove(self):
        logger.debug('on_but_remove', __name__)
        bjpeds.remove_files_pedestals()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIDark ()
    widget.show()
    app.exec_()

#-----------------------------
