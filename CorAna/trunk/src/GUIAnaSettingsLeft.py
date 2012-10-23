#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIAnaSettingsLeft...
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

from ConfigParametersCorAna import confpars as cp

from GUIConfigParameters import *

#---------------------
#  Class definition --
#---------------------
class GUIAnaSettingsLeft ( QtGui.QWidget ) :
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

        self.lmethods = ['evenly spaced','non-evenly spaced']
        self.lfields  = []

        #self.tit_kin_win_size    = QtGui.QLabel('kinetics window size')
        #self.edi_kin_win_size    = QtGui.QLineEdit( str( cp.kin_win_size   .value() ) )        


        self.grid_row = 0
        self.grid = QtGui.QGridLayout()

        self.guiSection('Static  Q   Partition', self.lmethods[0], '1')
        self.guiSection('Static  Phi Partition', self.lmethods[1], '2')
        self.guiSection('Dynamic Q   Partition', self.lmethods[0], '3')
        self.guiSection('Dynamic Phi Partition', self.lmethods[1], '4')
 
        cp.guiconfigparameters = GUIConfigParameters()
        self.grid.addWidget(cp.guiconfigparameters,    self.grid_row, 0, 1, 10)
        self.setLayout(self.grid)

        #self.connect( self.box_kin_mode       , QtCore.SIGNAL('currentIndexChanged(int)'), self.on_box_kin_mode        )
        #self.connect( self.edi_kin_win_size   , QtCore.SIGNAL('editingFinished ()'),       self.on_edi_kin_win_size    )
 
        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit field'
#        self.titKinetic.setToolTip('This section allows to monitor/modify\nthe beam zero parameters\nin transmission mode')
#        self.edi_kin_win_size   .setToolTip( msg )


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):
        self.                    setStyleSheet (cp.styleBkgd)
#        self.titKinetic         .setStyleSheet(cp.styleTitle)
#        self.tit_kin_win_size   .setStyleSheet(cp.styleLabel)
#        self.edi_kin_win_size   .setAlignment (QtCore.Qt.AlignRight)
#        self.edi_kin_win_size   .setFixedWidth(width)
#        self.box_kin_mode       .setStyleSheet(cp.styleBox) 
#        self.edi_kin_win_size   .setStyleSheet(cp.styleEdit) 


    def guiSection(self, title, method, str_edit) :
        tit0 = QtGui.QLabel(title)
        tit1 = QtGui.QLabel('Method')
        tit2 = QtGui.QLabel('File/Number/Span')
        edi  = QtGui.QLineEdit( str_edit )        
        but  = QtGui.QPushButton('Browse')
        box  = QtGui.QComboBox( self ) 
        box.addItems(self.lmethods)
        box.setCurrentIndex( self.lmethods.index(method) )

        self.lfields.append( (tit0, tit1, tit2, edi, but, box ) )

        #self.grid.addWidget(self.lfields[0][0],     self.grid_row, 0)
        self.grid.addWidget(tit0, self.grid_row,   0, 1, 9)
        self.grid.addWidget(tit1, self.grid_row+1, 1)
        self.grid.addWidget(tit2, self.grid_row+2, 1)
        self.grid.addWidget(box,  self.grid_row+1, 2, 1, 8)
        self.grid.addWidget(edi,  self.grid_row+2, 2, 1, 7)
        self.grid.addWidget(but,  self.grid_row+2, 9)
        self.grid_row += 3








    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        #print 'closeEvent'
        try: # try to delete self object in the cp
            del cp.anasettingsleft # GUIAnaSettingsLeft
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
#        cp.posGUIMain = (self.pos().x(),self.pos().y())

    def on_edi_kin_win_size(self):
        cp.kin_win_size.setValue( float(self.edi_kin_win_size.displayText()) )
        print 'Set kin_win_size =', cp.kin_win_size.value()

    def on_edi_kin_top_row(self):
        cp.kin_top_row.setValue( float(self.edi_kin_top_row.displayText()) )
        print 'Set kin_top_row =', cp.kin_top_row.value()

    def on_edi_kin_slice_first(self):
        cp.kin_slice_first.setValue( float(self.edi_kin_slice_first.displayText()) )
        print 'Set kin_slice_first =', cp.kin_slice_first.value()

    def on_edi_kin_slice_last(self):
        cp.kin_slice_last.setValue( float(self.edi_kin_slice_last.displayText()) )
        print 'Set kin_slice_last =', cp.kin_slice_last.value()

    def on_box_kin_mode(self):
        self.mode_name = self.box_kin_mode.currentText()
        cp.kin_mode.setValue( self.mode_name )
        print ' ---> selected kinematics mode: ' + self.mode_name
 
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIAnaSettingsLeft ()
    widget.show()
    app.exec_()

#-----------------------------
