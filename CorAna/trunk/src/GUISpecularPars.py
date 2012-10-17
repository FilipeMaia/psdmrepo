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

import ConfigParametersCorAna as cp

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
 
        self.titSpecular = QtGui.QLabel('Specularly Reflected Beam Parameters:')
        self.tit_x_coord = QtGui.QLabel('x-coordinate in full frame mode')      
        self.tit_y_coord = QtGui.QLabel('y-coordinate in full frame mode')      
        self.tit_x0_pos  = QtGui.QLabel('CCD x spec in specular measurement')   
        self.tit_z0_pos  = QtGui.QLabel('CCD z spec in specular measurement')   

        self.edi_x_coord = QtGui.QLineEdit( str( cp.confpars.x_coord_specular.value() ) )        
        self.edi_y_coord = QtGui.QLineEdit( str( cp.confpars.y_coord_specular.value() ) )        
        self.edi_x0_pos  = QtGui.QLineEdit( str( cp.confpars.x0_pos_in_specular.value() ) )        
        self.edi_z0_pos  = QtGui.QLineEdit( str( cp.confpars.z0_pos_in_specular.value() ) )        

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.titSpecular,       0, 0, 1, 9)
        self.grid.addWidget(self.tit_x_coord,       1, 1, 1, 9)
        self.grid.addWidget(self.tit_y_coord,       2, 1, 1, 9)
        self.grid.addWidget(self.tit_x0_pos ,       3, 1, 1, 9)
        self.grid.addWidget(self.tit_z0_pos ,       4, 1, 1, 9)
        self.grid.addWidget(self.edi_x_coord,       1, 10)
        self.grid.addWidget(self.edi_y_coord,       2, 10)
        self.grid.addWidget(self.edi_x0_pos ,       3, 10)
        self.grid.addWidget(self.edi_z0_pos ,       4, 10)
        self.setLayout(self.grid)

        self.connect( self.edi_x_coord,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_x_coord )
        self.connect( self.edi_y_coord,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_y_coord )
        self.connect( self.edi_x0_pos ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_x0_pos )
        self.connect( self.edi_z0_pos ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_z0_pos )
 
        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit coordinate'
        self.titSpecular.setToolTip('This section allows to monitor/modify\nthe beam/spec parameters\nin specular mode')
        self.edi_x_coord.setToolTip( msg )
        self.edi_y_coord.setToolTip( msg )
        self.edi_x0_pos .setToolTip( msg )
        self.edi_z0_pos .setToolTip( msg )

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.            setStyleSheet (cp.confpars.styleYellow)
        self.titSpecular.setStyleSheet (cp.confpars.styleTitle)
        self.tit_x_coord.setStyleSheet (cp.confpars.styleLabel)
        self.tit_y_coord.setStyleSheet (cp.confpars.styleLabel)
        self.tit_x0_pos .setStyleSheet (cp.confpars.styleLabel) 
        self.tit_z0_pos .setStyleSheet (cp.confpars.styleLabel) 

        self.edi_x_coord.setAlignment(QtCore.Qt.AlignRight)
        self.edi_y_coord.setAlignment(QtCore.Qt.AlignRight)
        self.edi_x0_pos .setAlignment(QtCore.Qt.AlignRight)
        self.edi_z0_pos .setAlignment(QtCore.Qt.AlignRight)

        width = 80

        self.edi_x_coord.setFixedWidth(width)
        self.edi_y_coord.setFixedWidth(width)
        self.edi_x0_pos .setFixedWidth(width)
        self.edi_z0_pos .setFixedWidth(width)

        self.edi_x_coord.setStyleSheet(cp.confpars.styleEdit) 
        self.edi_y_coord.setStyleSheet(cp.confpars.styleEdit) 
        self.edi_x0_pos .setStyleSheet(cp.confpars.styleEdit) 
        self.edi_z0_pos .setStyleSheet(cp.confpars.styleEdit) 


    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp.confpars
            del cp.confpars.guibatchinfoleft # GUISpecularPars
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

    def on_edi_x_coord(self):
        cp.confpars.x_coord_specular.setValue( float(self.edi_x_coord.displayText()) )
        print 'Set x_coord_specular =', cp.confpars.x_coord_specular.value()

    def on_edi_y_coord(self):
        print 'on_edi_y_coord'
        cp.confpars.y_coord_specular.setValue( float(self.edi_y_coord.displayText()) )
        print 'Set y_coord_specular =', cp.confpars.y_coord_specular.value()

    def on_edi_x0_pos(self):
        cp.confpars.x0_pos_in_specular.setValue( float(self.edi_x0_pos.displayText()) )
        print 'Set x0_pos_in_specular =', cp.confpars.x0_pos_in_specular.value()

    def on_edi_z0_pos(self):
        cp.confpars.z0_pos_in_specular.setValue( float(self.edi_z0_pos.displayText()) )
        print 'Set z0_pos_in_specular =', cp.confpars.z0_pos_in_specular.value()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUISpecularPars ()
    widget.show()
    app.exec_()

#-----------------------------
