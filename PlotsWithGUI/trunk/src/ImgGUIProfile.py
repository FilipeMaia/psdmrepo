#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgGUIProfile...
#
#------------------------------------------------------------------------

"""GUI for Profile.

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

class ImgGUIProfile (QtGui.QWidget) :
    """GUI for Profile"""

    def __init__(self, icp=None):
        QtGui.QWidget.__init__(self)

        self.icp = icp # Image control parameters
        self.icp.typeCurrent = self.icp.typeProfile     
        self.icp.formCurrent = self.icp.formLine     

        self.setWindowTitle('GUI for Spectrum')
        self.setFrame()

        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"

        self.txt_msg    = QtGui.QTextEdit('Profile: use Add, Move, Select or Remove mode and click and drag mouse on image.')
        #self.gui_mode   = igm.ImgGUIMode(self.icp)
        #self.but_overlay= QtGui.QPushButton("Overlay")
        #self.but_draw   = QtGui.QPushButton("Draw")

        width = 60

        #self.but_overlay.setMaximumWidth(width)
        #self.but_draw   .setMaximumWidth(width)
 
        #self.widg_xyz   = igxyz.ImgGUIXYZRanges()
        #self.widg_xyz.setXYZRanges(0,10,20,30,40,50)
        
        #self.connect(self.but_overlay,QtCore.SIGNAL('clicked()'), self.processOverlay)
        #self.connect(self.but_draw,   QtCore.SIGNAL('clicked()'), self.processDraw)

        # Layout with box sizers
        # 
        grid = QtGui.QGridLayout()
        row = 1
        grid.addWidget(self.txt_msg, row, 0, 1, 7)
        #row = 2
        #grid.addWidget(self.but_draw    , row, 0)
        #grid.addWidget(self.but_overlay , row, 3)

        vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        vbox.addLayout(grid)
        self.setLayout(vbox)

        self.showToolTips()


    def get_control(self) :
        return self.icp.control


    def showToolTips(self):
        # Tips for buttons and fields:
        #self.but_draw  .setToolTip('Draw spectrum for selected region')
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
    w = ImgGUIProfile(icp)
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
