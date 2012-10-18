#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIKineticMode...
#
#------------------------------------------------------------------------

"""GUI sets the beam coordinates w.r.t. camera frame for transmission/beam-zero mode"""

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
class GUIKineticMode ( QtGui.QWidget ) :
    """GUI sets the beam coordinates w.r.t. camera frame for transmission/beam-zero mode"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Beam Zero Parameters')
        self.setFrame()

        self.list_of_kin_modes  = ['Non-Kinetics', 'Kinetics']
        self.char_expand        = u' \u25BE' # down-head triangle

        self.popupMenuMode = QtGui.QMenu()
        for mode in self.list_of_kin_modes :
            self.popupMenuMode.addAction( mode )
 
        self.titKinetic          = QtGui.QLabel('Camera Working Mode:')
        self.tit_kin_win_size    = QtGui.QLabel('kinetics window size')
        self.tit_kin_top_row     = QtGui.QLabel('top row number of visible slice')
        self.tit_kin_slice_first = QtGui.QLabel('first usable kinetics slice')
        self.tit_kin_slice_last  = QtGui.QLabel('last usable kinetics slice')
        self.edi_kin_win_size    = QtGui.QLineEdit( str( cp.confpars.kin_win_size   .value() ) )        
        self.edi_kin_top_row     = QtGui.QLineEdit( str( cp.confpars.kin_top_row    .value() ) )        
        self.edi_kin_slice_first = QtGui.QLineEdit( str( cp.confpars.kin_slice_first.value() ) )        
        self.edi_kin_slice_last  = QtGui.QLineEdit( str( cp.confpars.kin_slice_last .value() ) )        
        self.but_kin_mode        = QtGui.QPushButton( cp.confpars.kin_mode.value() + self.char_expand  ) 

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.titKinetic,                0, 0, 1, 8)
        self.grid.addWidget(self.tit_kin_win_size   ,       1, 1, 1, 8)
        self.grid.addWidget(self.tit_kin_top_row    ,       2, 1, 1, 8)
        self.grid.addWidget(self.tit_kin_slice_first,       3, 1, 1, 8)
        self.grid.addWidget(self.tit_kin_slice_last ,       4, 1, 1, 8)
        self.grid.addWidget(self.but_kin_mode       ,       0, 8) 
        self.grid.addWidget(self.edi_kin_win_size   ,       1, 8)
        self.grid.addWidget(self.edi_kin_top_row    ,       2, 8)
        self.grid.addWidget(self.edi_kin_slice_first,       3, 8)
        self.grid.addWidget(self.edi_kin_slice_last ,       4, 8)
        self.setLayout(self.grid)

        self.connect( self.but_kin_mode       ,     QtCore.SIGNAL('clicked()'),          self.on_but_kin_mode        )
        self.connect( self.edi_kin_win_size   ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_kin_win_size    )
        self.connect( self.edi_kin_top_row    ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_kin_top_row     )
        self.connect( self.edi_kin_slice_first,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_kin_slice_first )
        self.connect( self.edi_kin_slice_last ,     QtCore.SIGNAL('editingFinished ()'), self.on_edi_kin_slice_last  )
 
        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit field'
        self.titKinetic.setToolTip('This section allows to monitor/modify\nthe beam zero parameters\nin transmission mode')
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

        width = 100

        self.                    setStyleSheet (cp.confpars.styleYellow)
        self.titKinetic         .setStyleSheet (cp.confpars.styleTitle)
        self.tit_kin_win_size   .setStyleSheet (cp.confpars.styleLabel)
        self.tit_kin_top_row    .setStyleSheet (cp.confpars.styleLabel)
        self.tit_kin_slice_first.setStyleSheet (cp.confpars.styleLabel) 
        self.tit_kin_slice_last .setStyleSheet (cp.confpars.styleLabel) 

        self.edi_kin_win_size   .setAlignment(QtCore.Qt.AlignRight)
        self.edi_kin_top_row    .setAlignment(QtCore.Qt.AlignRight)
        self.edi_kin_slice_first.setAlignment(QtCore.Qt.AlignRight)
        self.edi_kin_slice_last .setAlignment(QtCore.Qt.AlignRight)

        self.but_kin_mode       .setFixedWidth(width)
        self.edi_kin_win_size   .setFixedWidth(width)
        self.edi_kin_top_row    .setFixedWidth(width)
        self.edi_kin_slice_first.setFixedWidth(width)
        self.edi_kin_slice_last .setFixedWidth(width)

        self.but_kin_mode       .setStyleSheet(cp.confpars.styleGray) 
        self.edi_kin_win_size   .setStyleSheet(cp.confpars.styleEdit) 
        self.edi_kin_top_row    .setStyleSheet(cp.confpars.styleEdit) 
        self.edi_kin_slice_first.setStyleSheet(cp.confpars.styleEdit) 
        self.edi_kin_slice_last .setStyleSheet(cp.confpars.styleEdit) 


    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp.confpars
            del cp.confpars.guibeamzeropars # GUIKineticMode
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

    def on_edi_kin_win_size(self):
        cp.confpars.kin_win_size.setValue( float(self.edi_kin_win_size.displayText()) )
        print 'Set kin_win_size =', cp.confpars.kin_win_size.value()

    def on_edi_kin_top_row(self):
        cp.confpars.kin_top_row.setValue( float(self.edi_kin_top_row.displayText()) )
        print 'Set kin_top_row =', cp.confpars.kin_top_row.value()

    def on_edi_kin_slice_first(self):
        cp.confpars.kin_slice_first.setValue( float(self.edi_kin_slice_first.displayText()) )
        print 'Set kin_slice_first =', cp.confpars.kin_slice_first.value()

    def on_edi_kin_slice_last(self):
        cp.confpars.kin_slice_last.setValue( float(self.edi_kin_slice_last.displayText()) )
        print 'Set kin_slice_last =', cp.confpars.kin_slice_last.value()

    def on_but_kin_mode(self):
        action_selected = self.popupMenuMode.exec_(QtGui.QCursor.pos())
        if action_selected is None : return
        self.mode_name = action_selected.text()
        cp.confpars.kin_mode.setValue( self.mode_name )
        self.but_kin_mode.setText( self.mode_name + self.char_expand )
        print ' ---> selected kinematics mode: ' + self.mode_name
 
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIKineticMode ()
    widget.show()
    app.exec_()

#-----------------------------
