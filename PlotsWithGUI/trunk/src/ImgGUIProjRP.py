#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgGUIProjRP...
#
#------------------------------------------------------------------------

"""GUI for ProjRP.

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

class ImgGUIProjRP (QtGui.QWidget) :
    """GUI for ProjRP"""

    def __init__(self, icp=None):
        QtGui.QWidget.__init__(self)

        self.icp = icp # Image control parameters
        self.icp.typeCurrent = self.icp.typeProjRP     # <=================== DEPENDS ON TYPE
        self.icp.formCurrent = self.icp.formWedge      # <=================== DEPENDS ON FORM   

        self.setWindowTitle('GUI for ProjRP')
        self.setFrame()

        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"

        #self.gui_mode   = igm.ImgGUIMode(self.icp)

        self.txt_msg      = QtGui.QTextEdit('Projections R and Phi: use Add, Move, Select or Remove mode and click-and-drag mouse on image.' + \
                                            'The number of R-rings and Phi-sectors for projections can be selected.')
        self.but_reset    = QtGui.QPushButton('Reset')
        self.tit_nslices  = QtGui.QLabel('Number of r-rings, phi-sectors:')
        self.edi_nrings   = QtGui.QLineEdit(str(self.icp.n_rings))
        self.edi_nsects   = QtGui.QLineEdit(str(self.icp.n_sects))
  
        width     = 60
        width_edi = 40

        self.but_reset   .setMaximumWidth(width)
        self.edi_nrings.setMaximumWidth(width_edi)
        self.edi_nsects.setMaximumWidth(width_edi)
        self.edi_nrings.setValidator(QtGui.QIntValidator(1,1000,self))
        self.edi_nsects.setValidator(QtGui.QIntValidator(1,1000,self))
 
        self.connect(self.but_reset,    QtCore.SIGNAL('clicked()'),          self.onReset )
        self.connect(self.edi_nrings, QtCore.SIGNAL('editingFinished ()'), self.onEditNRings )
        self.connect(self.edi_nsects, QtCore.SIGNAL('editingFinished ()'), self.onEditNSects )
 
        # Layout with box sizers
        # 
        self.hboxB = QtGui.QHBoxLayout()
        self.hboxB.addWidget(self.tit_nslices)
        self.hboxB.addWidget(self.edi_nrings)
        self.hboxB.addWidget(self.edi_nsects)
        self.hboxB.addWidget(self.but_reset)
        self.hboxB.addStretch(1)
        
        #grid = QtGui.QGridLayout()
        #row = 0
        #grid.addWidget(self.txt_msg,       row, 0)
        #row = 1
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
        self.but_reset.setToolTip('Reset the number of R-rings and Phi-sectors to the default values.')
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


    def onEditNRings(self):
        print 'onEditNRings'
        self.icp.n_rings = int(self.edi_nrings.displayText())


    def onEditNSects(self):
        print 'onEditNSects'
        self.icp.n_sects = int(self.edi_nsects.displayText())


    def onReset(self):
        self.icp.n_rings=1
        self.icp.n_sects=1
        self.edi_nrings.setText(str(self.icp.n_rings))
        self.edi_nsects.setText(str(self.icp.n_sects))

        
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
    w = ImgGUIProjRP(icp)
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
