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
import ImgGUIXYZRanges as igxyz
import ImgConfigParameters as icp

from PyQt4 import QtGui, QtCore

#---------------------
#  Class definition --
#---------------------

class ImgGUISpectrum (QtGui.QWidget) :
    """GUI for Spectrum"""

    def __init__(self, parent=None ):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('GUI for Spectrum')
        self.setFrame()

        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"

        self.but_add    = QtGui.QPushButton("&Add")
        self.but_remove = QtGui.QPushButton("&Remove")
        self.but_select = QtGui.QPushButton("&Select")
        self.but_draw   = QtGui.QPushButton("&Draw")
        self.but_quit   = QtGui.QPushButton("&Quit")
        self.cbox_grid  = QtGui.QCheckBox("Show &Grid")
        self.cbox_grid.setChecked(False)
        self.cbox_log  = QtGui.QCheckBox("&Log")
        self.cbox_log.setChecked(False)

        width = 60
        self.but_add   .setMaximumWidth(width)
        self.but_remove.setMaximumWidth(width)
        self.but_select.setMaximumWidth(width)
        self.but_draw  .setMaximumWidth(width)
        self.but_quit  .setMaximumWidth(width)
 
        #self.widg_xyz   = igxyz.ImgGUIXYZRanges()
        #self.widg_xyz.setXYZRanges(0,10,20,30,40,50)
        
        self.connect(self.but_add,    QtCore.SIGNAL('clicked()'),         self.processAdd)
        self.connect(self.but_remove, QtCore.SIGNAL('clicked()'),         self.processRemove)
        self.connect(self.but_select, QtCore.SIGNAL('clicked()'),         self.processSelect)
        self.connect(self.but_draw,   QtCore.SIGNAL('clicked()'),         self.processDraw)
        self.connect(self.but_quit,   QtCore.SIGNAL('clicked()'),         self.processQuit)
        #self.connect(self.cbox_grid, QtCore.SIGNAL('stateChanged(int)'), self.processDraw)
        #self.connect(self.cbox_log,  QtCore.SIGNAL('stateChanged(int)'), self.processDraw)

        # Layout with box sizers
        # 
        grid = QtGui.QGridLayout()

        row = 1
        grid.addWidget(self.but_add      , row, 0)
        grid.addWidget(self.but_select   , row, 1)
        grid.addWidget(self.but_remove   , row, 2)
        #grid.addWidget(self.widg_xyz    , row, 3, 3, 3)
        row = 2
        grid.addWidget(self.but_draw    , row, 0)
        grid.addWidget(self.cbox_log    , row, 5)
        grid.addWidget(self.cbox_grid   , row, 6)
        grid.addWidget(self.but_quit    , row, 7)

        vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        vbox.addLayout(grid)
        self.setLayout(vbox)

        #self.setEditFieldColors()    
        #self.setEditFieldValues()

        self.showToolTips()

        self.widg_img = icp.imgconfpars.widg_img



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


    def processAdd(self):
        print 'Add'
        self.widg_img.selectRectangularRegion() 




    def processRemove(self):
        print 'Remove'
        #self.widg_img.on_draw()
        self.widg_img.resetImage()


    def processSelect(self):
        print 'Select'


    def processDraw(self):
        print 'Draw'


    def processQuit(self):
        print 'Quit'
        self.close()


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        print 'Close application'
           
#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    w = ImgGUISpectrum(None)
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
