#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIFlatField...
#
#------------------------------------------------------------------------

"""GUI sets the flat field file"""

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
from ImgSpeWithGUI          import *
import GlobalUtils          as     gu
from GUIFileBrowser         import *

#---------------------
#  Class definition --
#---------------------
class GUIFlatField ( QtGui.QWidget ) :
    """GUI sets the flat field file"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Flat field file')
        self.setFrame()

        self.edi_path = QtGui.QLineEdit( fnm.path_flat() )        
        self.edi_path.setReadOnly( True )   
        self.but_path = QtGui.QPushButton('File:')
        self.but_plot = QtGui.QPushButton('Plot')
        self.but_brow = QtGui.QPushButton('File browser')

        self.grid = QtGui.QGridLayout()
        self.grid_row = 1
        #self.grid.addWidget(self.tit_path, self.grid_row,   0)
        self.grid.addWidget(self.but_path,  self.grid_row,   0)
        self.grid.addWidget(self.edi_path,  self.grid_row,   1, 1, 6)
        self.grid.addWidget(self.but_plot,  self.grid_row+1, 0)
        self.grid.addWidget(self.but_brow,  self.grid_row+1, 1, 1, 2)

        self.connect(self.but_path, QtCore.SIGNAL('clicked()'), self.on_but_path )
        self.connect(self.but_plot, QtCore.SIGNAL('clicked()'), self.on_but_plot )
        self.connect(self.but_brow, QtCore.SIGNAL('clicked()'), self.on_but_browser )

        self.setLayout(self.grid)

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        #self           .setToolTip('Use this GUI to work with xtc file.')
        self.edi_path   .setToolTip('The path to the xtc file for processing in this GUI')
        self.but_plot   .setToolTip('Plot image and spectrum for flat field file')
        self.but_brow   .setToolTip('Browse flat field file')
        
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
        self.edi_path.setStyleSheet (cp.styleEditInfo)
        self.edi_path.setAlignment  (QtCore.Qt.AlignRight)

        self.but_path.setStyleSheet (cp.styleButton)
        self.but_plot.setStyleSheet (cp.styleButton) 
        self.but_brow.setStyleSheet (cp.styleButton) 
  
        self.but_path.setFixedWidth(width)
        self.but_plot.setFixedWidth(width)
    
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

        try    : cp.guifilebrowser.close()
        except : pass

        try    : del cp.guiflatfield # GUIFlatField
        except : pass # silently ignore


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def on_but_path(self):
        logger.debug('Flat field file browser', __name__ )
        path = str(self.edi_path.text())        
        path = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',path) )
        dname, fname = os.path.split(path)

        if dname == '' or fname == '' :
            logger.info('Input directiry name or file name is empty... keep file path unchanged...')
            return

        self.edi_path.setText(path)
        cp.dname_flat.setValue(dname)
        cp.fname_flat.setValue(fname)
        logger.info('selected file: ' + str(fnm.path_flat()), __name__ )


    def on_but_plot(self):
        logger.debug('on_but_plot', __name__)
        try :
            cp.imgspewithgui.close()
        except :
            arr = gu.get_array_from_file(fnm.path_flat())
            if arr == None : return
            logger.debug('Array shape: ' + str(arr.shape), __name__)
            cp.imgspewithgui = ImgSpeWithGUI(None, arr)
            cp.imgspewithgui.move(self.parentWidget().pos().__add__(QtCore.QPoint(400,20)))
            cp.imgspewithgui.show()


    def on_but_browser (self):       
        logger.debug('on_but_browser', __name__)
        try    :
            cp.guifilebrowser.close()
        except :
            cp.guifilebrowser = GUIFileBrowser(None, [fnm.path_flat()])
            cp.guifilebrowser.move(self.parentWidget().pos().__add__(QtCore.QPoint(240,40)))
            cp.guifilebrowser.show()


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIFlatField ()
    widget.show()
    app.exec_()

#-----------------------------
