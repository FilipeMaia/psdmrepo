#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIImgSizePosition...
#
#------------------------------------------------------------------------

"""GUI sets the beam and spec coordinates w.r.t. camera frame for specular mode"""

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
class GUIImgSizePosition ( QtGui.QWidget ) :
    """GUI sets the beam and spec coordinates w.r.t. camera frame for specular mode"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Specularly Reflected Beam Parameters')
        self.setFrame()

        self.tit_img_size  = QtGui.QLabel('CCD Image Size (pix):')
        self.tit_col       = QtGui.QLabel('column')
        self.tit_row       = QtGui.QLabel('row')
        self.tit_begin     = QtGui.QLabel('begin')
        self.tit_end       = QtGui.QLabel('end')
        self.tit_size      = QtGui.QLabel('size')
 
        self.edi_col_begin    = QtGui.QLineEdit( str( cp.col_begin   .value() ) )        
        self.edi_col_end      = QtGui.QLineEdit( str( cp.col_end     .value() ) )        
        self.edi_col_size     = QtGui.QLineEdit( str( cp.bat_img_cols.value() ) )        
        self.edi_row_begin    = QtGui.QLineEdit( str( cp.row_begin   .value() ) )        
        self.edi_row_end      = QtGui.QLineEdit( str( cp.row_end     .value() ) )        
        self.edi_row_size     = QtGui.QLineEdit( str( cp.bat_img_rows.value() ) )        

        self.grid = QtGui.QGridLayout()

        self.grid.addWidget(self.tit_img_size,       2, 0, 1,8)
        self.grid.addWidget(self.tit_col     ,       3, 3)
        self.grid.addWidget(self.tit_row     ,       3, 5)
        self.grid.addWidget(self.tit_begin   ,       4, 2)
        self.grid.addWidget(self.tit_end     ,       5, 2)
        self.grid.addWidget(self.tit_size    ,       6, 2)
        
        self.grid.addWidget(self.edi_col_begin  ,    4, 3)
        self.grid.addWidget(self.edi_col_end    ,    5, 3)
        self.grid.addWidget(self.edi_col_size   ,    6, 3)
        self.grid.addWidget(self.edi_row_begin  ,    4, 5)
        self.grid.addWidget(self.edi_row_end    ,    5, 5)
        self.grid.addWidget(self.edi_row_size   ,    6, 5)

        self.setLayout(self.grid)

        self.connect( self.edi_col_begin  ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_col_begin   )
        self.connect( self.edi_col_end    ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_col_end     )
        self.connect( self.edi_row_begin  ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_row_begin   )
        self.connect( self.edi_row_end    ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_row_end     )
 
        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit field'
        self.tit_img_size   .setToolTip('This section allows to monitor/modify\nthe frame image size and position')
        self.edi_col_begin  .setToolTip( msg )
        self.edi_col_end    .setToolTip( msg )
        self.edi_row_begin  .setToolTip( msg )
        self.edi_row_end    .setToolTip( msg )

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):

        width = 80
        width_label = 50

        self.              setStyleSheet (cp.styleBkgd)
        self.tit_img_size .setStyleSheet (cp.styleTitle)
        self.tit_col      .setStyleSheet (cp.styleLabel)
        self.tit_row      .setStyleSheet (cp.styleLabel)
        self.tit_begin    .setStyleSheet (cp.styleLabel) 
        self.tit_end      .setStyleSheet (cp.styleLabel) 
        self.tit_size     .setStyleSheet (cp.styleLabel) 

        self.tit_begin    .setAlignment(QtCore.Qt.AlignRight)
        self.tit_end      .setAlignment(QtCore.Qt.AlignRight)
        self.tit_size     .setAlignment(QtCore.Qt.AlignRight)
        self.tit_col      .setAlignment(QtCore.Qt.AlignCenter)
        self.tit_row      .setAlignment(QtCore.Qt.AlignCenter)

        self.tit_begin    .setFixedWidth(width_label)
        self.tit_end      .setFixedWidth(width_label)
        self.tit_size     .setFixedWidth(width_label)

        self.edi_col_begin  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_col_end    .setAlignment(QtCore.Qt.AlignRight)
        self.edi_col_size   .setAlignment(QtCore.Qt.AlignRight)
        self.edi_row_begin  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_row_end    .setAlignment(QtCore.Qt.AlignRight)
        self.edi_row_size   .setAlignment(QtCore.Qt.AlignRight)

        self.edi_col_begin  .setFixedWidth(width)
        self.edi_col_end    .setFixedWidth(width)
        self.edi_col_size   .setFixedWidth(width)
        self.edi_row_begin  .setFixedWidth(width)
        self.edi_row_end    .setFixedWidth(width)
        self.edi_row_size   .setFixedWidth(width)

        self.edi_col_begin  .setStyleSheet(cp.styleEdit) 
        self.edi_col_end    .setStyleSheet(cp.styleEdit) 
        self.edi_col_size   .setStyleSheet(cp.styleEditInfo) 
        self.edi_row_begin  .setStyleSheet(cp.styleEdit) 
        self.edi_row_end    .setStyleSheet(cp.styleEdit) 
        self.edi_row_size   .setStyleSheet(cp.styleEditInfo) 

        self.edi_col_size   .setReadOnly(True)
        self.edi_row_size   .setReadOnly(True)



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
        try    : del cp.guiimgsizeposition # GUIImgSizePosition
        except : pass # silently ignore

    def onClose(self):
        logger.debug('onClose', __name__) 
        self.close()

    def on_edi_col_begin(self):
        cp.col_begin.setValue( float(self.edi_col_begin.displayText()) )
        logger.info('Set col_begin =' + str(cp.col_begin.value()), __name__)

    def on_edi_col_end(self):
        cp.col_end.setValue( float(self.edi_col_end.displayText()) )
        logger.info('Set col_end =' + str(cp.col_end.value()), __name__)

    def on_edi_row_begin(self):
        cp.row_begin.setValue( float(self.edi_row_begin.displayText()) )
        logger.info('Set row_begin =' + str(cp.row_begin.value()), __name__)

    def on_edi_row_end(self):
        cp.row_end.setValue( float(self.edi_row_end.displayText()) )
        logger.info('Set row_end =' + str(cp.row_end.value()), __name__)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIImgSizePosition ()
    widget.show()
    app.exec_()

#-----------------------------
