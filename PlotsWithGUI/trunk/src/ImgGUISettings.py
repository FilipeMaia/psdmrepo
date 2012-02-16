#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgGUISettings...
#
#------------------------------------------------------------------------

"""GUI for Settings.

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

class ImgGUISettings (QtGui.QWidget) :
    """GUI for ProjRP"""

    def __init__(self, icp=None):
        QtGui.QWidget.__init__(self)

        self.icp = icp # Image control parameters

        self.setWindowTitle('GUI for Settings')
        self.setFrame()

        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"

        #self.txt_msg       = QtGui.QTextEdit('Settings')
        self.but_reset     = QtGui.QPushButton('Reset')
        self.tit_linewidth = QtGui.QLabel('Line width')
        self.tit_color     = QtGui.QLabel('Color')
        self.edi_linewidth = QtGui.QLineEdit(str(self.icp.lwAdd))
        self.edi_color     = QtGui.QLineEdit(str(self.icp.colAdd))
  
        width     = 60
        width_edi = 40

        self.but_reset    .setMaximumWidth(width)
        self.edi_linewidth.setMaximumWidth(width_edi)
        self.edi_color.    setMaximumWidth(width_edi)
        self.edi_linewidth.setValidator(QtGui.QIntValidator(1,1000,self))
 
        self.connect(self.but_reset,     QtCore.SIGNAL('clicked()'),          self.onReset )
        self.connect(self.edi_color,     QtCore.SIGNAL('editingFinished ()'), self.onEditColor )
        self.connect(self.edi_linewidth, QtCore.SIGNAL('editingFinished ()'), self.onEditLineWidth )
 
        # Layout with box sizers
        # 
        #self.hboxB = QtGui.QHBoxLayout()
        #self.hboxB.addWidget(self.tit_nslices)
        #self.hboxB.addWidget(self.edi_nrings)
        #self.hboxB.addWidget(self.edi_nsects)
        #self.hboxB.addWidget(self.but_reset)
        #self.hboxB.addStretch(1)
        
        grid = QtGui.QGridLayout()
        row = 0
        #grid.addWidget(self.txt_msg,       row, 0)
        #row = 1
        grid.addWidget(self.tit_linewidth, row, 0)
        grid.addWidget(self.edi_linewidth, row, 1)
        grid.addWidget(self.tit_color,     row, 2)
        grid.addWidget(self.edi_color,     row, 3)
        grid.addWidget(self.but_reset,     row, 4)

        vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        vbox.addLayout(grid)
        #vbox.addWidget(self.txt_msg)
        #vbox.addLayout(self.hboxB)
        self.setLayout(vbox)

        self.showToolTips()


    def get_control(self) :
        return self.icp.control


    def showToolTips(self):
        # Tips for buttons and fields:
        self.but_reset.setToolTip('Reset all numbers to the default values.')


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


    def onEditLineWidth(self):
        print 'onEditLineWidth'
        self.icp.lwAdd = int(self.edi_linewidth.displayText())
        print 'self.icp.lwAdd  =', self.icp.lwAdd  


    def onEditColor(self):
        print 'onEditColor'
        self.icp.colAdd = str(self.edi_color.displayText())
        print 'self.icp.colAdd =', self.icp.colAdd  


    def onReset(self):
        self.icp.lwAdd  =  1
        self.icp.colAdd = 'g'
        self.edi_linewidth.setText(str(self.icp.lwAdd))
        self.edi_color    .setText(str(self.icp.colAdd))
        print 'self.icp.lwAdd  =', self.icp.lwAdd  
        print 'self.icp.colAdd =', self.icp.colAdd  

        
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
    w = ImgGUISettings(icp)
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
