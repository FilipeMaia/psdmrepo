#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIBlamish...
#
#------------------------------------------------------------------------

"""GUI sets the blamish file"""

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
from PlotImgSpe             import *
import GlobalUtils          as     gu
from GUIFileBrowser         import *


#---------------------
#  Class definition --
#---------------------
class GUIBlamish ( QtGui.QWidget ) :
    """GUI sets the blamish file"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 530, 30)
        self.setWindowTitle('Blamish file')
        self.setFrame()

        self.cbx_use = QtGui.QCheckBox('Use blemish correction', self)
        self.cbx_use.setChecked( cp.ccdcorr_blemish.value() )

        self.edi_path = QtGui.QLineEdit( fnm.path_blam() )        
        self.edi_path.setReadOnly( True )   
        self.but_path = QtGui.QPushButton('File:')
        self.but_plot = QtGui.QPushButton('Plot')
        self.but_brow = QtGui.QPushButton('Browse')

        self.grid = QtGui.QGridLayout()
        self.grid_row = 1
        #self.grid.addWidget(self.tit_path, self.grid_row,   0)
        self.grid.addWidget(self.cbx_use,  self.grid_row,   0, 1, 6)
        self.grid.addWidget(self.but_path, self.grid_row+1, 0)
        self.grid.addWidget(self.edi_path, self.grid_row+1, 1, 1, 6)
        self.grid.addWidget(self.but_plot, self.grid_row+2, 0)
        self.grid.addWidget(self.but_brow, self.grid_row+2, 1)

        self.connect(self.cbx_use,  QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 
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
        self.edi_path   .setToolTip('The path to the blamish mask file')
        self.but_path   .setToolTip('Push this button and select the blamish mask file')
        self.but_plot   .setToolTip('Plot image and spectrum for blamish file')
        self.but_brow   .setToolTip('Browse blamish file')
        self.cbx_use    .setToolTip('Check box \nto set and use \nblamish mask correction')
        
    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        width = 60
        self.setMinimumWidth(530)
        self.setStyleSheet(cp.styleBkgd)
        self.edi_path.setStyleSheet (cp.styleEditInfo)
        self.edi_path.setAlignment  (QtCore.Qt.AlignRight)

        self.but_path.setStyleSheet (cp.styleButton)
        self.but_plot.setStyleSheet (cp.styleButton) 
        self.but_brow.setStyleSheet (cp.styleButton) 
   
        self.but_path.setFixedWidth(width)
        self.but_plot.setFixedWidth(width)
        self.but_brow.setFixedWidth(width)
        self.cbx_use   .setStyleSheet (cp.styleLabel)

        self.setButtonState()


    def setButtonState(self):
        self.but_path.setEnabled(cp.ccdcorr_blemish.value())
        self.but_plot.setEnabled(cp.ccdcorr_blemish.value())
        self.but_brow.setEnabled(cp.ccdcorr_blemish.value())

        #self.but_path.setFlat(not cp.ccdcorr_blemish.value())
        #self.but_plot.setFlat(not cp.ccdcorr_blemish.value())
        #self.but_brow.setFlat(not cp.ccdcorr_blemish.value())

    
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

        try    : cp.plotimgspe.close()
        except : pass

        try    : cp.guifilebrowser.close()
        except : pass
            
        #try    : del cp.guiblamish # GUIBlamish
        #except : pass # silently ignore


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def on_but_path(self):
        logger.debug('Blamish file browser', __name__ )
        path = str(self.edi_path.text())        
        path = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',path) )
        dname, fname = os.path.split(path)

        if dname == '' or fname == '' :
            logger.info('Input directiry name or file name is empty... keep file path unchanged...')
            return

        self.edi_path.setText(path)
        cp.dname_blam.setValue(dname)
        cp.fname_blam.setValue(fname)
        logger.info('selected file: ' + str(fnm.path_blam()), __name__ )


    def on_but_plot(self):
        logger.debug('on_but_plot', __name__)
        try :
            logger.debug('try to close', __name__)
            cp.plotimgspe.close()
            #but.setStyleSheet(cp.styleButtonBad)
        except :
            logger.debug('except and open', __name__)
            arr = gu.get_array_from_file(fnm.path_blam())
            if arr == None : return
            logger.debug('Array shape: ' + str(arr.shape), __name__)
            cp.plotimgspe = PlotImgSpe(None, arr, ofname=fnm.path_blam_plot())
            cp.plotimgspe.move(cp.guimain.pos().__add__(QtCore.QPoint(740,140))) # self.parentWidget().pos()
            cp.plotimgspe.show()


    def on_but_browser (self):       
        logger.debug('on_but_browser', __name__)
        try    :
            cp.guifilebrowser.close()
        except :
            cp.guifilebrowser = GUIFileBrowser(None, [fnm.path_blam()])
            cp.guifilebrowser.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.guifilebrowser.show()


    def onCBox(self):
        #if self.cbx_use .hasFocus() :
        par = cp.ccdcorr_blemish
        par.setValue( self.cbx_use.isChecked() )
        msg = 'onCBox - set status of ccdcorr_blemish: ' + str(par.value())
        logger.info(msg, __name__ )
        self.setButtonState()

#
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIBlamish ()
    widget.show()
    app.exec_()

#-----------------------------
