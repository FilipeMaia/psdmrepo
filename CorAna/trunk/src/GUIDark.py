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

        self.tit_path   = QtGui.QLabel('Path:')
        self.tit_batch  = QtGui.QLabel('Batch:')
        self.edi_path   = QtGui.QLineEdit( fnm.path_dark_xtc() )        
        self.edi_path.setReadOnly( True )  
        self.but_submit = QtGui.QPushButton('Submit')
        self.but_status = QtGui.QPushButton('Status')
        self.but_wfiles = QtGui.QPushButton('Work files')
        self.but_plot   = QtGui.QPushButton('Plot')

        self.grid = QtGui.QGridLayout()
        self.grid_row = 1

        self.grid.addWidget(self.tit_path,   self.grid_row,   0)
        self.grid.addWidget(self.edi_path,   self.grid_row,   1, 1, 9)
        self.grid.addWidget(self.but_submit, self.grid_row+1, 1)
        self.grid.addWidget(self.but_status, self.grid_row+1, 2)
        self.grid.addWidget(self.but_wfiles, self.grid_row+1, 3)
        self.grid.addWidget(self.but_plot,   self.grid_row+1, 4)
        self.grid_row += 3

        self.connect(self.but_submit, QtCore.SIGNAL('clicked()'),     self.on_but_submit  )
        self.connect(self.but_status, QtCore.SIGNAL('clicked()'),     self.on_but_status  )
        self.connect(self.but_wfiles, QtCore.SIGNAL('clicked()'),     self.on_but_wfiles  )
        self.connect(self.but_plot,   QtCore.SIGNAL('clicked()'),     self.on_but_plot    )
        #self.connect(edi, QtCore.SIGNAL('editingFinished()'),        self.onEdit )
        #self.connect(box, QtCore.SIGNAL('currentIndexChanged(int)'), self.onBox  )

        self.setLayout(self.grid)

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        self           .setToolTip('Use this GUI to work with xtc file.')
        self.edi_path  .setToolTip('The path to the xtc file for processing in this GUI.')
        self.but_submit.setToolTip('Click on this button\nto submit job in batch.')
        self.but_status.setToolTip('Click on this button\nto check the batch job status.\nBatch job status will be\nprinted in the GUI Logger.')
        self.but_wfiles.setToolTip('Click on this button\nto check the files availability.')
        self.but_plot  .setToolTip('Click on this button\nto plot pedestals.')
        
    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        width = 80
        self.setMinimumWidth(400)
        self.setStyleSheet(cp.styleBkgd)
        #tit0   .setStyleSheet (cp.styleTitle)
        self.tit_path   .setStyleSheet (cp.styleLabel)
        self.edi_path   .setStyleSheet (cp.styleEditInfo) # cp.styleEditInfo
        self.edi_path   .setAlignment  (QtCore.Qt.AlignRight)

        self.but_submit .setStyleSheet (cp.styleButton) 
        self.but_status .setStyleSheet (cp.styleButton) 
        self.but_wfiles .setStyleSheet (cp.styleButton) 
        self.but_plot   .setStyleSheet (cp.styleButton) 

        self.but_submit .setFixedWidth(width)
        self.but_status .setFixedWidth(width)
        self.but_wfiles .setFixedWidth(width)
        self.but_plot   .setFixedWidth(width)

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
        try    : del cp.guidark # GUIDark
        except : pass # silently ignore

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def on_but_submit(self):
        logger.debug('on_but_submit', __name__)
        bjpeds.submit_batch_for_pedestals()

    def on_but_status(self):
        logger.debug('on_but_status', __name__)
        bjpeds.check_batch_status_for_pedestals()

    def on_but_wfiles(self):
        logger.debug('on_but_wfiles', __name__)
        #bjpeds.print_work_files_for_pedestals()
        bjpeds.check_work_files_for_pedestals()

    def on_but_plot(self):
        logger.debug('on_but_plot', __name__)
        try :
            cp.imgspewithgui.close()
            del cp.imgspewithgui
            #but.setStyleSheet(cp.styleButtonBad)
        except :
            arr = bjpeds.get_pedestals_from_file()
            print arr.shape
            print arr

            #gg.plotImageAndSpectrum(arr,range=(100,300))
            #gg.show()

            cp.imgspewithgui = ImgSpeWithGUI(None, arr)
            #cp.imgspewithgui.setParent(self)
            #cp.imgspewithgui.set_image_array( arr )
            cp.imgspewithgui.move(self.pos().__add__(QtCore.QPoint(20,82))) # open window with offset w.r.t. parent
            cp.imgspewithgui.show()
            #but.setStyleSheet(cp.styleButtonGood)


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIDark ()
    widget.show()
    app.exec_()

#-----------------------------
