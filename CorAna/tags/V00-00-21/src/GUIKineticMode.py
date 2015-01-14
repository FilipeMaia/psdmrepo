#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIKineticMode...
#
#------------------------------------------------------------------------

"""GUI sets the kinetic mode parameters"""

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
from Logger import logger
from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------
class GUIKineticMode ( QtGui.QWidget ) :
    """GUI sets the kinetic mode parameters"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Kinetic Mode')
        self.setFrame()

        #self.list_of_kin_modes  = ['Non-Kinetics', 'Kinetics']
 
        self.tit_kinetic         = QtGui.QLabel('Kinetic Mode parameters:')
        self.tit_kin_win_size    = QtGui.QLabel('kinetics window size')
        self.tit_kin_top_row     = QtGui.QLabel('top row number of visible slice')
        self.tit_kin_slice_first = QtGui.QLabel('first usable kinetics slice')
        self.tit_kin_slice_last  = QtGui.QLabel('last usable kinetics slice')
        self.edi_kin_win_size    = QtGui.QLineEdit( str( cp.kin_win_size   .value() ) )        
        self.edi_kin_top_row     = QtGui.QLineEdit( str( cp.kin_top_row    .value() ) )        
        self.edi_kin_slice_first = QtGui.QLineEdit( str( cp.kin_slice_first.value() ) )        
        self.edi_kin_slice_last  = QtGui.QLineEdit( str( cp.kin_slice_last .value() ) )        
        #self.box_kin_mode        = QtGui.QComboBox( self ) 
        #self.box_kin_mode.addItems(self.list_of_kin_modes)
        #self.box_kin_mode.setCurrentIndex( self.list_of_kin_modes.index(cp.kin_mode.value()) )

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.tit_kinetic,               0, 0, 1, 8)
        self.grid.addWidget(self.tit_kin_win_size   ,       1, 1, 1, 8)
        self.grid.addWidget(self.tit_kin_top_row    ,       2, 1, 1, 8)
        self.grid.addWidget(self.tit_kin_slice_first,       3, 1, 1, 8)
        self.grid.addWidget(self.tit_kin_slice_last ,       4, 1, 1, 8)
        #self.grid.addWidget(self.box_kin_mode       ,       0, 8) 
        self.grid.addWidget(self.edi_kin_win_size   ,       1, 8)
        self.grid.addWidget(self.edi_kin_top_row    ,       2, 8)
        self.grid.addWidget(self.edi_kin_slice_first,       3, 8)
        self.grid.addWidget(self.edi_kin_slice_last ,       4, 8)
        self.setLayout(self.grid)

        #self.connect( self.box_kin_mode       , QtCore.SIGNAL('currentIndexChanged(int)'), self.on_box_kin_mode        )
        self.connect( self.edi_kin_win_size   , QtCore.SIGNAL('editingFinished ()'),       self.on_edi_kin_win_size    )
        self.connect( self.edi_kin_top_row    , QtCore.SIGNAL('editingFinished ()'),       self.on_edi_kin_top_row     )
        self.connect( self.edi_kin_slice_first, QtCore.SIGNAL('editingFinished ()'),       self.on_edi_kin_slice_first )
        self.connect( self.edi_kin_slice_last , QtCore.SIGNAL('editingFinished ()'),       self.on_edi_kin_slice_last  )
 
        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit field'
        self.tit_kinetic.setToolTip('This section allows to monitor/modify\nthe beam zero parameters\nin transmission mode')
        self.edi_kin_win_size   .setToolTip( msg )
        self.edi_kin_top_row    .setToolTip( msg )
        self.edi_kin_slice_first.setToolTip( msg )
        self.edi_kin_slice_last .setToolTip( msg )

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

        self.                    setStyleSheet (cp.styleBkgd)
        self.tit_kinetic        .setStyleSheet (cp.styleTitle)
        self.tit_kin_win_size   .setStyleSheet (cp.styleLabel)
        self.tit_kin_top_row    .setStyleSheet (cp.styleLabel)
        self.tit_kin_slice_first.setStyleSheet (cp.styleLabel) 
        self.tit_kin_slice_last .setStyleSheet (cp.styleLabel) 

        self.edi_kin_win_size   .setAlignment(QtCore.Qt.AlignRight)
        self.edi_kin_top_row    .setAlignment(QtCore.Qt.AlignRight)
        self.edi_kin_slice_first.setAlignment(QtCore.Qt.AlignRight)
        self.edi_kin_slice_last .setAlignment(QtCore.Qt.AlignRight)

        #self.box_kin_mode       .setFixedWidth(100)
        self.edi_kin_win_size   .setFixedWidth(width)
        self.edi_kin_top_row    .setFixedWidth(width)
        self.edi_kin_slice_first.setFixedWidth(width)
        self.edi_kin_slice_last .setFixedWidth(width)

        #self.box_kin_mode       .setStyleSheet(cp.styleBox) 
        self.edi_kin_win_size   .setStyleSheet(cp.styleEdit) 
        self.edi_kin_top_row    .setStyleSheet(cp.styleEdit) 
        self.edi_kin_slice_first.setStyleSheet(cp.styleEdit) 
        self.edi_kin_slice_last .setStyleSheet(cp.styleEdit) 


    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        try: # try to delete self object in the cp
            del cp.guikineticmode # GUIKineticMode
        except AttributeError:
            pass # silently ignore

    def processClose(self):
        logger.debug('processClose', __name__)
        self.close()

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__)
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__)
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass

    def on_edi_kin_win_size(self):
        cp.kin_win_size.setValue( float(self.edi_kin_win_size.displayText()) )
        logger.info('Set kin_win_size = ' + str(cp.kin_win_size.value()), __name__ )

    def on_edi_kin_top_row(self):
        cp.kin_top_row.setValue( float(self.edi_kin_top_row.displayText()) )
        logger.info('Set kin_top_row =' + str(cp.kin_top_row.value()), __name__ )

    def on_edi_kin_slice_first(self):
        cp.kin_slice_first.setValue( float(self.edi_kin_slice_first.displayText()) )
        logger.info('Set kin_slice_first =' + str(cp.kin_slice_first.value()), __name__ )

    def on_edi_kin_slice_last(self):
        cp.kin_slice_last.setValue( float(self.edi_kin_slice_last.displayText()) )
        logger.info('Set kin_slice_last =' + str(cp.kin_slice_last.value()), __name__ )

#    def on_box_kin_mode(self):
#        self.mode_name = self.box_kin_mode.currentText()
#        cp.kin_mode.setValue( self.mode_name )
#        logger.info(' ---> selected kinematic mode: ' + self.mode_name, __name__ )
 
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIKineticMode ()
    widget.show()
    app.exec_()

#-----------------------------
