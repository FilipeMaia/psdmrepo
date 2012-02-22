#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgGUIProjXY...
#
#------------------------------------------------------------------------

"""GUI for ProjXY.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule
@version $Id: 
@author Mikhail S. Dubrovin
"""

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

#import ImgGUIMode as igm
from PyQt4 import QtGui, QtCore

#---------------------
#  Class definition --
#---------------------

class ImgGUIProjXY (QtGui.QWidget) :
    """GUI for ProjXY"""

    def __init__(self, icp=None):
        QtGui.QWidget.__init__(self)

        self.icp = icp # Image control parameters
        self.icp.typeCurrent = self.icp.typeProjXY     # <=================== DEPENDS ON TYPE
        self.icp.formCurrent = self.icp.formRect       # <=================== DEPENDS ON FORM   

        self.setWindowTitle('GUI for ProjXY')
        self.setFrame()

        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"

        #self.gui_mode   = igm.ImgGUIMode(self.icp)

        self.txt_msg      = QtGui.QTextEdit('Projections X and Y: 1) click on Add, Move, Select or Remove mode, 2) click-and-drag mouse on image.' + \
                                            'The number of slices for X and Y projections can be selected.')
        self.but_reset    = QtGui.QPushButton('Reset')
        self.tit_nslices  = QtGui.QLabel('N slices X,Y:')
        self.edi_nxslices = QtGui.QLineEdit(str(self.icp.nx_slices))
        self.edi_nyslices = QtGui.QLineEdit(str(self.icp.ny_slices))
  
        width     = 60
        width_edi = 40

        self.but_reset   .setMaximumWidth(width)
        self.edi_nxslices.setMaximumWidth(width_edi)
        self.edi_nyslices.setMaximumWidth(width_edi)
        self.edi_nxslices.setValidator(QtGui.QIntValidator(1,1000,self))
        self.edi_nyslices.setValidator(QtGui.QIntValidator(1,1000,self))
 
        self.connect(self.but_reset,    QtCore.SIGNAL('clicked()'),          self.onReset )
        self.connect(self.edi_nxslices, QtCore.SIGNAL('editingFinished ()'), self.onEditNXSlices )
        self.connect(self.edi_nyslices, QtCore.SIGNAL('editingFinished ()'), self.onEditNYSlices )
 
        # Layout with box sizers
        # 
        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.tit_nslices)
        self.hboxB.addWidget(self.edi_nxslices)
        self.hboxB.addWidget(self.edi_nyslices)
        self.hboxB.addWidget(self.but_reset)
        self.hboxB.addStretch(1)
        
        grid = QtGui.QGridLayout()
        row = 0
        grid.addWidget(self.txt_msg,       row, 0)
        row = 1
        #grid.addWidget(self.tit_nslices,  row, 0)
        #grid.addWidget(self.edi_nxslices, row, 1)
        #grid.addWidget(self.edi_nyslices, row, 2)
        #grid.addWidget(self.but_reset,    row, 3)

        vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        #vbox.addLayout(grid)
        vbox.addWidget(self.txt_msg)
        vbox.addLayout(self.hboxB)
        self.setLayout(vbox)

        self.showToolTips()


    def get_control(self) :
        return self.icp.control


    def showToolTips(self):
        # Tips for buttons and fields:
        self.but_reset.setToolTip('Reset the number of slices in X and Y to the default values.')
        pass

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def processOverlay(self):
        self.get_control().signal_to_control( self.icp.formRect, self.icp.modeOverlay )


    def onEditNXSlices(self):
        print 'onEditNXSlices'
        self.icp.nx_slices = int(self.edi_nxslices.displayText())


    def onEditNYSlices(self):
        print 'onEditNYSlices'
        self.icp.ny_slices = int(self.edi_nyslices.displayText())


    def onReset(self):
        self.icp.nx_slices=1
        self.icp.ny_slices=1
        self.edi_nxslices.setText(str(self.icp.nx_slices))
        self.edi_nyslices.setText(str(self.icp.ny_slices))

        
    def processDraw(self):
        self.get_control().signal_draw()


    def processQuit(self):
        print 'Quit'
        self.gui_mode.close() 
        self.close() # will call closeEvent()
        self.get_control().signal_quit()


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        print 'Close application'
           
#-----------------------------

def main():

    import ImgConfigParameters as gicp
    icp = gicp.giconfpars.addImgConfigPars( None )

    app = QtGui.QApplication(sys.argv)
    w = ImgGUIProjXY(icp)
    w.move(QtCore.QPoint(50,50))
    w.show()
    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()
    sys.exit ('End of test')

#-----------------------------
