#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISpecularPars...
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

from ConfigParametersCorAna import confpars as cp
from Logger                 import logger

#---------------------
#  Class definition --
#---------------------
class GUISpecularPars ( QtGui.QWidget ) :
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
 
        self.tit_specular    = QtGui.QLabel('Specularly Reflected Beam Parameters:')
        self.tit_x_coord     = QtGui.QLabel('x-coordinate in full frame mode')      
        self.tit_y_coord     = QtGui.QLabel('y-coordinate in full frame mode')      
        #self.tit_x0_pos      = QtGui.QLabel('CCD x0 position in beam0 measurement')
        #self.tit_z0_pos      = QtGui.QLabel('CCD z0 position in beam0 measurement')
        self.tit_x0_pos_spec = QtGui.QLabel('CCD x spec in specular measurement')   
        self.tit_z0_pos_spec = QtGui.QLabel('CCD z spec in specular measurement')   


        self.edi_x_coord     = QtGui.QLineEdit( str( cp.x_coord_specular.value() ) )        
        self.edi_y_coord     = QtGui.QLineEdit( str( cp.y_coord_specular.value() ) )        
        #self.edi_x0_pos      = QtGui.QLineEdit( str( cp.x0_pos_in_beam0.value() ) )        
        #self.edi_z0_pos      = QtGui.QLineEdit( str( cp.z0_pos_in_beam0.value() ) )        
        self.edi_x0_pos_spec = QtGui.QLineEdit( str( cp.x0_pos_in_specular.value() ) )        
        self.edi_z0_pos_spec = QtGui.QLineEdit( str( cp.z0_pos_in_specular.value() ) )        

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.tit_specular,      0, 0, 1, 9)
        self.grid.addWidget(self.tit_x_coord,       1, 1, 1, 9)
        self.grid.addWidget(self.tit_y_coord,       2, 1, 1, 9)
        #self.grid.addWidget(self.tit_x0_pos ,       3, 1, 1, 9)
        #self.grid.addWidget(self.tit_z0_pos ,       4, 1, 1, 9)
        self.grid.addWidget(self.tit_x0_pos_spec ,  5, 1, 1, 9)
        self.grid.addWidget(self.tit_z0_pos_spec ,  6, 1, 1, 9)

        self.grid.addWidget(self.edi_x_coord,       1, 10)
        self.grid.addWidget(self.edi_y_coord,       2, 10)
        #self.grid.addWidget(self.edi_x0_pos ,       3, 10)
        #self.grid.addWidget(self.edi_z0_pos ,       4, 10)
        self.grid.addWidget(self.edi_x0_pos_spec ,  5, 10)
        self.grid.addWidget(self.edi_z0_pos_spec ,  6, 10)
        self.setLayout(self.grid)

        self.connect( self.edi_x_coord,      QtCore.SIGNAL('editingFinished ()'), self.on_edi_x_coord )
        self.connect( self.edi_y_coord,      QtCore.SIGNAL('editingFinished ()'), self.on_edi_y_coord )
        #self.connect( self.edi_x0_pos ,      QtCore.SIGNAL('editingFinished ()'), self.on_edi_x0_pos )
        #self.connect( self.edi_z0_pos ,      QtCore.SIGNAL('editingFinished ()'), self.on_edi_z0_pos )
        self.connect( self.edi_x0_pos_spec , QtCore.SIGNAL('editingFinished ()'), self.on_edi_x0_pos_spec )
        self.connect( self.edi_z0_pos_spec , QtCore.SIGNAL('editingFinished ()'), self.on_edi_z0_pos_spec )
 
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
        self.edi_z0_pos_spec  .setToolTip( msg )

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):
        self.setFixedHeight(150)

        self.                  setStyleSheet (cp.styleBkgd)
        self.tit_specular     .setStyleSheet (cp.styleTitle)
        self.tit_x_coord      .setStyleSheet (cp.styleLabel)
        self.tit_y_coord      .setStyleSheet (cp.styleLabel)
        #self.tit_x0_pos       .setStyleSheet (cp.styleLabel) 
        #self.tit_z0_pos       .setStyleSheet (cp.styleLabel) 
        self.tit_x0_pos_spec  .setStyleSheet (cp.styleLabel) 
        self.tit_z0_pos_spec  .setStyleSheet (cp.styleLabel) 

        self.edi_x_coord.setAlignment(QtCore.Qt.AlignRight)
        self.edi_y_coord.setAlignment(QtCore.Qt.AlignRight)
        #self.edi_x0_pos      .setAlignment(QtCore.Qt.AlignRight)
        #self.edi_z0_pos      .setAlignment(QtCore.Qt.AlignRight)
        self.edi_x0_pos_spec .setAlignment(QtCore.Qt.AlignRight)
        self.edi_z0_pos_spec .setAlignment(QtCore.Qt.AlignRight)


        width = 80

        self.edi_x_coord.setFixedWidth(width)
        self.edi_y_coord.setFixedWidth(width)
        #self.edi_x0_pos      .setFixedWidth(width)
        #self.edi_z0_pos      .setFixedWidth(width)
        self.edi_x0_pos_spec .setFixedWidth(width)
        self.edi_z0_pos_spec .setFixedWidth(width)

        self.edi_x_coord.setStyleSheet(cp.styleEdit) 
        self.edi_y_coord.setStyleSheet(cp.styleEdit) 
        #self.edi_x0_pos      .setStyleSheet(cp.styleEdit) 
        #self.edi_z0_pos      .setStyleSheet(cp.styleEdit) 
        self.edi_x0_pos_spec .setStyleSheet(cp.styleEdit) 
        self.edi_z0_pos_spec .setStyleSheet(cp.styleEdit) 


    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        logger.debug('moveEvent', __name__) 
#        cp.posGUIMain = (self.pos().x(),self.pos().y())

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__) 
        try    : del cp.guispecularpars # GUISpecularPars
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

    def on_edi_z0_pos(self):
        cp.z0_pos_in_beam0.setValue( float(self.edi_z0_pos.displayText()) )
        logger.info('Set z0_pos_in_beam0 =' + str(cp.z0_pos_in_beam0.value()), __name__)

    def on_edi_x0_pos_spec(self):
        cp.x0_pos_in_specular.setValue( float(self.edi_x0_pos_spec.displayText()) )
        logger.info('Set x0_pos_in_specular =' + str(cp.x0_pos_in_specular.value()), __name__)

    def on_edi_z0_pos_spec(self):
        cp.z0_pos_in_specular.setValue( float(self.edi_z0_pos_spec.displayText()) )
        logger.info('Set z0_pos_in_specular =' + str(cp.z0_pos_in_specular.value()), __name__)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUISpecularPars ()
    widget.show()
    app.exec_()

#-----------------------------
