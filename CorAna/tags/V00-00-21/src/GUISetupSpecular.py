#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISetupSpecular...
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
class GUISetupSpecular ( QtGui.QWidget ) :
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
 
        self.tit_specular    = QtGui.QLabel('Specularly Reflected Beam Coords (pix):')
        self.tit_ccd_pos     = QtGui.QLabel('CCD Position In Specular Meas. (mm):')

        self.tit_x_coord     = QtGui.QLabel('x:')      
        self.tit_y_coord     = QtGui.QLabel('y:')      
        self.tit_x0_pos_spec = QtGui.QLabel('x:')   
        self.tit_y0_pos_spec = QtGui.QLabel('y:')   

        self.edi_x_coord     = QtGui.QLineEdit( str( cp.x_coord_specular.value() ) )        
        self.edi_y_coord     = QtGui.QLineEdit( str( cp.y_coord_specular.value() ) )        
        self.edi_x0_pos_spec = QtGui.QLineEdit( str( cp.x0_pos_in_specular.value() ) )        
        self.edi_y0_pos_spec = QtGui.QLineEdit( str( cp.y0_pos_in_specular.value() ) )        

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.tit_specular,      2, 0, 1, 8)
        self.grid.addWidget(self.tit_x_coord,       3, 2)
        self.grid.addWidget(self.tit_y_coord,       3, 4)
        self.grid.addWidget(self.edi_x_coord,       3, 3)
        self.grid.addWidget(self.edi_y_coord,       3, 5)

        self.grid.addWidget(self.tit_ccd_pos,       0, 0, 1, 8)
        self.grid.addWidget(self.tit_x0_pos_spec ,  1, 2)
        self.grid.addWidget(self.tit_y0_pos_spec ,  1, 4)
        self.grid.addWidget(self.edi_x0_pos_spec ,  1, 3)
        self.grid.addWidget(self.edi_y0_pos_spec ,  1, 5)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.grid)
        self.vbox.addStretch(1) 

        self.setLayout(self.vbox)

        self.connect( self.edi_x_coord,      QtCore.SIGNAL('editingFinished ()'), self.on_edi_x_coord )
        self.connect( self.edi_y_coord,      QtCore.SIGNAL('editingFinished ()'), self.on_edi_y_coord )
        self.connect( self.edi_x0_pos_spec , QtCore.SIGNAL('editingFinished ()'), self.on_edi_x0_pos_spec )
        self.connect( self.edi_y0_pos_spec , QtCore.SIGNAL('editingFinished ()'), self.on_edi_y0_pos_spec )
 
        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit coordinate'
        self.tit_specular.setToolTip('This section allows to monitor/modify\nthe beam/spec parameters\nin specular mode')
        self.edi_x_coord .setToolTip( msg )
        self.edi_y_coord .setToolTip( msg )
        self.edi_x0_pos_spec  .setToolTip( msg )
        self.edi_y0_pos_spec  .setToolTip( msg )

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):
        self.setFixedHeight(150)

        width = 80
        width_label = 50

        self.                  setStyleSheet (cp.styleBkgd)
        self.tit_specular     .setStyleSheet (cp.styleTitle)
        self.tit_ccd_pos      .setStyleSheet (cp.styleTitle)
        self.tit_x_coord      .setStyleSheet (cp.styleLabel)
        self.tit_y_coord      .setStyleSheet (cp.styleLabel)
        self.tit_x0_pos_spec  .setStyleSheet (cp.styleLabel) 
        self.tit_y0_pos_spec  .setStyleSheet (cp.styleLabel) 

        self.tit_x_coord      .setAlignment(QtCore.Qt.AlignRight)
        self.tit_y_coord      .setAlignment(QtCore.Qt.AlignRight)
        self.tit_x0_pos_spec  .setAlignment(QtCore.Qt.AlignRight)
        self.tit_y0_pos_spec  .setAlignment(QtCore.Qt.AlignRight)

        self.tit_x_coord      .setFixedWidth(width_label)
        self.tit_y_coord      .setFixedWidth(width_label)
        self.tit_x0_pos_spec  .setFixedWidth(width_label)
        self.tit_y0_pos_spec  .setFixedWidth(width_label)

        self.edi_x_coord      .setAlignment(QtCore.Qt.AlignRight)
        self.edi_y_coord      .setAlignment(QtCore.Qt.AlignRight)
        self.edi_x0_pos_spec  .setAlignment(QtCore.Qt.AlignRight)
        self.edi_y0_pos_spec  .setAlignment(QtCore.Qt.AlignRight)

        self.edi_x_coord      .setFixedWidth(width)
        self.edi_y_coord      .setFixedWidth(width)
        self.edi_x0_pos_spec  .setFixedWidth(width)
        self.edi_y0_pos_spec  .setFixedWidth(width)

        self.edi_x_coord      .setStyleSheet(cp.styleEdit) 
        self.edi_y_coord      .setStyleSheet(cp.styleEdit) 
        self.edi_x0_pos_spec  .setStyleSheet(cp.styleEdit) 
        self.edi_y0_pos_spec  .setStyleSheet(cp.styleEdit) 


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
        try    : del cp.guisetupspecular # GUISetupSpecular
        except : pass # silently ignore

    def on_edi_x_coord(self):
        cp.x_coord_specular.setValue( float(self.edi_x_coord.displayText()) )
        logger.info('Set x_coord_specular =' + str(cp.x_coord_specular.value()), __name__)

    def on_edi_y_coord(self):
        cp.y_coord_specular.setValue( float(self.edi_y_coord.displayText()) )
        logger.info('Set y_coord_specular =' + str(cp.y_coord_specular.value()), __name__)

    def on_edi_x0_pos(self):
        cp.x0_pos_in_beam0.setValue( float(self.edi_x0_pos.displayText()) )
        logger.info('Set x0_pos_in_beam0 =' + str(cp.x0_pos_in_beam0.value()), __name__)

    def on_edi_y0_pos(self):
        cp.y0_pos_in_beam0.setValue( float(self.edi_y0_pos.displayText()) )
        logger.info('Set y0_pos_in_beam0 =' + str(cp.y0_pos_in_beam0.value()), __name__)

    def on_edi_x0_pos_spec(self):
        cp.x0_pos_in_specular.setValue( float(self.edi_x0_pos_spec.displayText()) )
        logger.info('Set x0_pos_in_specular =' + str(cp.x0_pos_in_specular.value()), __name__)

    def on_edi_y0_pos_spec(self):
        cp.y0_pos_in_specular.setValue( float(self.edi_y0_pos_spec.displayText()) )
        logger.info('Set y0_pos_in_specular =' + str(cp.y0_pos_in_specular.value()), __name__)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUISetupSpecular ()
    widget.show()
    app.exec_()

#-----------------------------
