#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgGUISpectrum...
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

class ImgGUISpectrum (QtGui.QWidget) :
    """GUI for Spectrum"""

    def __init__(self, icp=None):
        QtGui.QWidget.__init__(self)

        self.icp = icp # Image control parameters
        self.icp.typeCurrent = self.icp.typeSpectrum     

        self.setWindowTitle('GUI for Spectrum')
        self.setFrame()

        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"

        self.but_add    = QtGui.QPushButton("Add")
        self.but_remove = QtGui.QPushButton("Remove")
        self.but_select = QtGui.QPushButton("Select")
        self.but_overlay= QtGui.QPushButton("Overlay")
        self.but_normal = QtGui.QPushButton("Move")
        self.but_draw   = QtGui.QPushButton("Draw")
        self.cbox_grid  = QtGui.QCheckBox("Show &Grid")
        self.cbox_grid   .setChecked(self.icp.gridIsOn)
        self.cbox_log  = QtGui.QCheckBox("&Log")
        self.cbox_log    .setChecked(False)

        width = 60
        self.but_add    .setMaximumWidth(width)
        self.but_remove .setMaximumWidth(width)
        self.but_select .setMaximumWidth(width)
        self.but_overlay.setMaximumWidth(width)
        self.but_normal .setMaximumWidth(width)
        self.but_draw   .setMaximumWidth(width)
 
        #self.widg_xyz   = igxyz.ImgGUIXYZRanges()
        #self.widg_xyz.setXYZRanges(0,10,20,30,40,50)
        
        self.connect(self.but_add,    QtCore.SIGNAL('clicked()'),         self.processAdd)
        self.connect(self.but_remove, QtCore.SIGNAL('clicked()'),         self.processRemove)
        self.connect(self.but_select, QtCore.SIGNAL('clicked()'),         self.processSelect)
        self.connect(self.but_overlay,QtCore.SIGNAL('clicked()'),         self.processOverlay)
        self.connect(self.but_normal, QtCore.SIGNAL('clicked()'),         self.processNormal)
        self.connect(self.but_draw,   QtCore.SIGNAL('clicked()'),         self.processDraw)
        self.connect(self.cbox_grid,  QtCore.SIGNAL('stateChanged(int)'), self.processCBoxGrid)
        #self.connect(self.cbox_log,  QtCore.SIGNAL('stateChanged(int)'), self.processDraw)

        # Layout with box sizers
        # 
        grid = QtGui.QGridLayout()

        row = 1
        grid.addWidget(self.but_normal  , row, 0)
        grid.addWidget(self.but_add     , row, 1)
        grid.addWidget(self.but_select  , row, 2)
        grid.addWidget(self.but_overlay , row, 3)
        grid.addWidget(self.but_remove  , row, 4)
        grid.addWidget(self.cbox_log    , row, 6)
        #grid.addWidget(self.widg_xyz   , row, 3, 3, 3)
        row = 2
        grid.addWidget(self.but_draw    , row, 0)
        grid.addWidget(self.cbox_grid   , row, 6)

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
        #print 'ImgGUISpectrum: get_control(): ', self.icp.print_icp()
        return self.icp.control


    def showToolTips(self):
        # Tips for buttons and fields:
        self.but_add   .setToolTip('Add rectangular region for spectrum')
        self.but_remove.setToolTip('Remove region for spectrum')
        self.but_select.setToolTip('Select region for spectrum')
        self.but_draw  .setToolTip('Draw spectrum for selected region')


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


    def processCBoxGrid(self):
        self.icp.gridIsOn = self.cbox_grid.isChecked()
        self.get_control().signal_grid_onoff()


    def processAdd(self):
        self.get_control().signal_to_control( self.icp.formRect, self.icp.modeAdd )


    def processRemove(self):
        self.get_control().signal_to_control( self.icp.formRect, self.icp.modeRemove )


    def processSelect(self):
        self.get_control().signal_to_control( self.icp.formRect, self.icp.modeSelect )


    def processOverlay(self):
        self.get_control().signal_to_control( self.icp.formRect, self.icp.modeOverlay )


    def processNormal(self):
        self.get_control().signal_to_control( self.icp.formRect, self.icp.modeNone )


    def processDraw(self):
        self.get_control().signal_draw()


    def processQuit(self):
        print 'Quit'
        self.close() # will call closeEvent()
        self.get_control().signal_quit()


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        print 'Close application'
           
#-----------------------------

def main():

    import ImgConfigParameters as gicp
    icp = gicp.giconfpars.addImgConfigPars( None )

    app = QtGui.QApplication(sys.argv)
    w = ImgGUISpectrum(icp)
    w.move(QtCore.QPoint(50,50))
    w.show()
    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
