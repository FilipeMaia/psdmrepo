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

import ConfigParametersCorAna as cp

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

        self.titImageSize  = QtGui.QLabel('Saved Camera Image Size:')
        self.tit_col       = QtGui.QLabel('column (x)')
        self.tit_row       = QtGui.QLabel('row (y)')
        self.tit_begin     = QtGui.QLabel('begin')
        self.tit_end       = QtGui.QLabel('end')
        self.titCameraPos  = QtGui.QLabel('Camera Posituion During Data Collection:')
        self.tit_frame_x   = QtGui.QLabel('Pos x')
        self.tit_frame_z   = QtGui.QLabel('Pos z')
 
        self.edi_col_begin    = QtGui.QLineEdit( str( cp.confpars.col_begin  .value() ) )        
        self.edi_col_end      = QtGui.QLineEdit( str( cp.confpars.col_end    .value() ) )        
        self.edi_row_begin    = QtGui.QLineEdit( str( cp.confpars.row_begin  .value() ) )        
        self.edi_row_end      = QtGui.QLineEdit( str( cp.confpars.row_end    .value() ) )        
        self.edi_x_frame_pos  = QtGui.QLineEdit( str( cp.confpars.x_frame_pos.value() ) )        
        self.edi_z_frame_pos  = QtGui.QLineEdit( str( cp.confpars.z_frame_pos.value() ) )        

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.titImageSize,       0, 0, 1,8)
        self.grid.addWidget(self.tit_col     ,       1, 3)
        self.grid.addWidget(self.tit_row     ,       1, 5)
        self.grid.addWidget(self.tit_begin   ,       2, 2)
        self.grid.addWidget(self.tit_end     ,       3, 2)
        self.grid.addWidget(self.titCameraPos,       4, 0, 1,8)
        self.grid.addWidget(self.tit_frame_x ,       5, 2)
        self.grid.addWidget(self.tit_frame_z ,       5, 4)

        self.grid.addWidget(self.edi_col_begin  ,       2, 3)
        self.grid.addWidget(self.edi_col_end    ,       3, 3)
        self.grid.addWidget(self.edi_row_begin  ,       2, 5)
        self.grid.addWidget(self.edi_row_end    ,       3, 5)
        self.grid.addWidget(self.edi_x_frame_pos,       5, 3)
        self.grid.addWidget(self.edi_z_frame_pos,       5, 5)

        self.setLayout(self.grid)

        self.connect( self.edi_col_begin  ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_col_begin   )
        self.connect( self.edi_col_end    ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_col_end     )
        self.connect( self.edi_row_begin  ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_row_begin   )
        self.connect( self.edi_row_end    ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_row_end     )
        self.connect( self.edi_x_frame_pos,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_x_frame_pos )
        self.connect( self.edi_z_frame_pos,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_z_frame_pos )
 
        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'You may edit this field...\non your own risk, of cause...'
        self.titImageSize.setToolTip('This section allows to monitor/modify\nthe frame image size and position')
        self.edi_col_begin  .setToolTip( msg )
        self.edi_col_end    .setToolTip( msg )
        self.edi_row_begin  .setToolTip( msg )
        self.edi_row_end    .setToolTip( msg )
        self.edi_x_frame_pos.setToolTip( msg )
        self.edi_z_frame_pos.setToolTip( msg )

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

        self.              setStyleSheet (cp.confpars.styleYellow)
        self.titImageSize .setStyleSheet (cp.confpars.styleTitle)
        self.tit_col      .setStyleSheet (cp.confpars.styleLabel)
        self.tit_row      .setStyleSheet (cp.confpars.styleLabel)
        self.tit_begin    .setStyleSheet (cp.confpars.styleLabel) 
        self.tit_end      .setStyleSheet (cp.confpars.styleLabel) 
        self.titCameraPos .setStyleSheet (cp.confpars.styleTitle) 
        self.tit_frame_x  .setStyleSheet (cp.confpars.styleLabel) 
        self.tit_frame_z  .setStyleSheet (cp.confpars.styleLabel) 

        self.tit_begin    .setAlignment(QtCore.Qt.AlignRight)
        self.tit_end      .setAlignment(QtCore.Qt.AlignRight)
        self.tit_col      .setAlignment(QtCore.Qt.AlignCenter)
        self.tit_row      .setAlignment(QtCore.Qt.AlignCenter)
        self.tit_frame_x  .setAlignment(QtCore.Qt.AlignRight)
        self.tit_frame_z  .setAlignment(QtCore.Qt.AlignRight)

        self.tit_begin    .setFixedWidth(width_label)
        self.tit_end      .setFixedWidth(width_label)
        self.tit_frame_x  .setFixedWidth(width_label)
        self.tit_frame_z  .setFixedWidth(width_label)


        self.edi_col_begin  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_col_end    .setAlignment(QtCore.Qt.AlignRight)
        self.edi_row_begin  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_row_end    .setAlignment(QtCore.Qt.AlignRight)
        self.edi_x_frame_pos.setAlignment(QtCore.Qt.AlignRight)
        self.edi_z_frame_pos.setAlignment(QtCore.Qt.AlignRight)

        self.edi_col_begin  .setFixedWidth(width)
        self.edi_col_end    .setFixedWidth(width)
        self.edi_row_begin  .setFixedWidth(width)
        self.edi_row_end    .setFixedWidth(width)
        self.edi_x_frame_pos.setFixedWidth(width)
        self.edi_z_frame_pos.setFixedWidth(width)


        self.edi_col_begin  .setStyleSheet(cp.confpars.styleEdit) 
        self.edi_col_end    .setStyleSheet(cp.confpars.styleEdit) 
        self.edi_row_begin  .setStyleSheet(cp.confpars.styleEdit) 
        self.edi_row_end    .setStyleSheet(cp.confpars.styleEdit) 
        self.edi_x_frame_pos.setStyleSheet(cp.confpars.styleEdit) 
        self.edi_z_frame_pos.setStyleSheet(cp.confpars.styleEdit) 


    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp.confpars
            del cp.confpars.guiimgsizeposition # GUIImgSizePosition
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
#        cp.confpars.posGUIMain = (self.pos().x(),self.pos().y())

    def on_edi_col_begin(self):
        cp.confpars.col_begin.setValue( float(self.edi_col_begin.displayText()) )
        print 'Set col_begin =', cp.confpars.col_begin.value()

    def on_edi_col_end(self):
        cp.confpars.col_end.setValue( float(self.edi_col_end.displayText()) )
        print 'Set col_end =', cp.confpars.col_end.value()

    def on_edi_row_begin(self):
        cp.confpars.row_begin.setValue( float(self.edi_row_begin.displayText()) )
        print 'Set row_begin =', cp.confpars.row_begin.value()

    def on_edi_row_end(self):
        cp.confpars.row_end.setValue( float(self.edi_row_end.displayText()) )
        print 'Set row_end =', cp.confpars.row_end.value()

    def on_edi_x_frame_pos(self):
        cp.confpars.x_frame_pos.setValue( float(self.edi_x_frame_pos.displayText()) )
        print 'Set x_frame_pos =', cp.confpars.x_frame_pos.value()

    def on_edi_z_frame_pos(self):
        cp.confpars.z_frame_pos.setValue( float(self.edi_z_frame_pos.displayText()) )
        print 'Set z_frame_pos =', cp.confpars.z_frame_pos.value()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIImgSizePosition ()
    widget.show()
    app.exec_()

#-----------------------------
