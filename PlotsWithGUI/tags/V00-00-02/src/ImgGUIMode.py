#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgGUIMode...
#
#------------------------------------------------------------------------

"""GUI for Spectrum.

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
#import ImgGUIXYZRanges as igxyz
#import ImgConfigParameters as icp

from PyQt4 import QtGui, QtCore

#---------------------
#  Class definition --
#---------------------

class ImgGUIMode (QtGui.QWidget) :
    """GUI for Mode selection"""

    def __init__(self, icp=None):
        QtGui.QWidget.__init__(self)

        self.icp = icp # Image control parameters
        #self.icp.typeCurrent = self.icp.typeSpectrum     
        #self.icp.formCurrent = self.icp.formRect     

        self.setWindowTitle('GUI for Spectrum')
        self.setFrame()

        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"

        self.tit_status = QtGui.QLabel('Status:')
        self.setStatus()

        self.but_add    = QtGui.QPushButton("&Add")
        self.but_move   = QtGui.QPushButton("&Move")
        self.but_select = QtGui.QPushButton("Select")
        self.but_remove = QtGui.QPushButton("Remove")
        self.cbox_grid  = QtGui.QCheckBox  ("Grid")
        self.cbox_grid   .setChecked(self.icp.gridIsOn)
        self.cbox_log   = QtGui.QCheckBox("Log")
        self.cbox_log    .setChecked(self.icp.logIsOn)

        width = 60
        self.but_add    .setMaximumWidth(width)
        self.but_remove .setMaximumWidth(width)
        self.but_select .setMaximumWidth(width)
        self.but_move   .setMaximumWidth(width)
 
        self.connect(self.but_add,    QtCore.SIGNAL('clicked()'),         self.processAdd)
        self.connect(self.but_move,   QtCore.SIGNAL('clicked()'),         self.processMove)
        self.connect(self.but_select, QtCore.SIGNAL('clicked()'),         self.processSelect)
        self.connect(self.but_remove, QtCore.SIGNAL('clicked()'),         self.processRemove)
        self.connect(self.cbox_grid,  QtCore.SIGNAL('stateChanged(int)'), self.processCBoxGrid)
        self.connect(self.cbox_log,   QtCore.SIGNAL('stateChanged(int)'), self.processCBoxLog)

        # Layout with box sizers
        # 
        grid = QtGui.QGridLayout()

        row = 1
        grid.addWidget(self.but_add     , row, 0)
        grid.addWidget(self.but_move    , row, 1)
        grid.addWidget(self.but_select  , row, 2)
        grid.addWidget(self.but_remove  , row, 3)
        grid.addWidget(self.cbox_log    , row, 4)
        grid.addWidget(self.cbox_grid   , row, 5)
        grid.addWidget(self.cbox_grid   , row, 6)
        grid.addWidget(self.tit_status  , row, 7)

        vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        vbox.addLayout(grid)
        self.setLayout(vbox)

        #self.setEditFieldColors()    
        #self.setEditFieldValues()

        self.showToolTips()

        #self.widg_img = icp.imgconfpars.widg_img
        #self.widg_img = self.parent.parent.wimg
        #self.list_of_circs = []


    def get_control(self) :
        #return self.parent.parent.contrl
        #print 'ImgGUIMode: get_control(): ', self.icp.print_icp()
        return self.icp.control


    def showToolTips(self):
        # Tips for buttons and fields:
        self.but_add   .setToolTip('Add object/figure to the image')
        self.but_remove.setToolTip('Remove object/figure from the image')
        self.but_select.setToolTip('Select object/figure on the image')
        self.but_move  .setToolTip('Move/edit object/figure on the image')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel, Plain, NoFrame | Sunken, Raised
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def processCBoxGrid(self):
        self.icp.gridIsOn = self.cbox_grid.isChecked()
        self.get_control().signal_grid_onoff()


    def processCBoxLog(self):
        self.icp.logIsOn = self.cbox_log.isChecked()
        self.get_control().signal_log_onoff()


    def processAdd(self):
        self.get_control().signal_to_control( self.icp.formCurrent, self.icp.modeAdd )
        self.setStatus()


    def processMove(self):
        self.get_control().signal_to_control( self.icp.formCurrent, self.icp.modeMove )
        self.setStatus()


    def processSelect(self):
        self.get_control().signal_to_control( self.icp.formCurrent, self.icp.modeSelect )
        self.setStatus()


    def processRemove(self):
        self.get_control().signal_to_control( self.icp.formCurrent, self.icp.modeRemove )
        self.setStatus()


    def setStatus(self):
        self.tit_status.setText('Status: ' + self.icp.typeCurrent + ' ' + self.icp.modeCurrent + ' ' + self.icp.formCurrent)


    def processQuit(self):
        print 'Quit'
        self.close() # will call closeEvent()
        self.get_control().signal_quit()


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        print 'Close application'
           
#-----------------------------

class Test :
    def __init__(self):
        pass


def main():

    import ImgControl           as ic
    import ImgConfigParameters as gicp

    control = ic.ImgControl()
    icp = gicp.giconfpars.addImgConfigPars( control )

    app = QtGui.QApplication(sys.argv)
    w = ImgGUIMode(icp)
    w.move(QtCore.QPoint(50,50))
    w.show()
    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
