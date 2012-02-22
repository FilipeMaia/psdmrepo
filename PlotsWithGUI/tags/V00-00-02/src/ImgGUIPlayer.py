#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgGUIPlayer...
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

class ImgGUIPlayer (QtGui.QWidget) :
    """GUI for Spectrum"""

    def __init__(self, icp=None):
        QtGui.QWidget.__init__(self)

        self.icp = icp # Image control parameters
        #self.icp.typeCurrent = self.icp.typeSpectrum     

        self.setWindowTitle('GUI for Spectrum')
        self.setFrame()

        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"

        self.tit_event      = QtGui.QLabel('Image:')
        self.tit_space      = QtGui.QLabel(' ')
        self.tit_increment = QtGui.QLabel('Incr:')
        self.edi_increment = QtGui.QLineEdit(str(self.icp.increment))
 
        self.but_previous = QtGui.QPushButton(u'\u25C0') # left-head triangle "Previous"
        self.but_current  = QtGui.QPushButton("Current")
        self.but_next     = QtGui.QPushButton(u'\u25B6') # right-head triangle "Next"
        self.but_print    = QtGui.QPushButton("&Print")
        self.but_save     = QtGui.QPushButton("&Save")
        self.but_quit     = QtGui.QPushButton("&Quit")

        self.cbox_onoff   = QtGui.QCheckBox  ("On/Off grid")
        if icp != None : self.cbox_onoff.setChecked(self.icp.gridIsOn)

        width      = 60
        width_half = 25
        width_edi  = 30
        #self.cbox_onoff  .setMaximumWidth(width)
        self.but_previous .setMaximumWidth(width_half)
        self.but_current  .setMaximumWidth(width)
        self.but_next     .setMaximumWidth(width_half)
        self.but_print    .setMaximumWidth(width)
        self.but_save     .setMaximumWidth(width)
        self.but_quit     .setMaximumWidth(width)
        self.tit_event    .setMaximumWidth(width)
        self.tit_space    .setMaximumWidth(width)
        self.edi_increment.setValidator(QtGui.QIntValidator(1,10000,self))
        self.edi_increment.setMaximumWidth(width_edi)

        
        self.connect(self.but_previous,  QtCore.SIGNAL('clicked()'),           self.processPrevious)
        self.connect(self.but_current,   QtCore.SIGNAL('clicked()'),           self.processCurrent)
        self.connect(self.but_next,      QtCore.SIGNAL('clicked()'),           self.processNext)
        self.connect(self.but_print,     QtCore.SIGNAL('clicked()'),           self.processPrint)
        self.connect(self.but_save,      QtCore.SIGNAL('clicked()'),           self.processSave)
        self.connect(self.but_quit,      QtCore.SIGNAL('clicked()'),           self.processQuit)
        self.connect(self.cbox_onoff,    QtCore.SIGNAL('stateChanged(int)'),   self.processOnOff)
        self.connect(self.edi_increment, QtCore.SIGNAL('editingFinished ()'),  self.onEditIncrement )

        # Layout with box sizers
        # 
        grid = QtGui.QGridLayout()

        row = 1
        grid.addWidget(self.cbox_onoff   , row, 0)
        grid.addWidget(self.tit_event    , row, 1)
        grid.addWidget(self.but_previous , row, 2)
        grid.addWidget(self.but_current  , row, 3)
        grid.addWidget(self.but_next     , row, 4)
        grid.addWidget(self.tit_increment, row, 5) 
        grid.addWidget(self.edi_increment, row, 6) 
        grid.addWidget(self.tit_space    , row, 7)
        grid.addWidget(self.but_print    , row, 8)
        grid.addWidget(self.but_save     , row, 9)
        grid.addWidget(self.but_quit     , row, 10)
        #row = 2

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
        #print 'ImgGUIPlayer: get_control(): ', self.icp.print_icp()
        return self.icp.control


    def showToolTips(self):
        # Tips for buttons and fields:
        self.cbox_onoff  .setToolTip('On/Off something')
        self.but_previous.setToolTip('Get previous event')
        self.but_current .setToolTip('Get current event')
        self.but_next    .setToolTip('Get next event')
        self.but_print   .setToolTip('Print configuration parameters')
        self.but_save    .setToolTip('Save current configuration parameters in file')
        self.but_quit    .setToolTip('Quit')


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


    def onEditIncrement(self):
        print 'onEditIncrement'
        self.icp.increment = int(self.edi_increment.displayText())


    def processOnOff(self):
        #print 'OnOff status', self.cbox_onoff.isChecked()
        self.icp.gridIsOn   = self.cbox_onoff.isChecked()
        self.get_control().signal_grid_onoff()


    def processPrevious(self):
        print 'Previous'
        self.get_control().signal_get_event_previous()
        pass


    def processCurrent(self):
        print 'Current'
        self.get_control().signal_get_event_current()
        pass


    def processNext(self):
        print 'Next'
        self.get_control().signal_get_event_next()
        pass


    def processPrint(self):
        print 'Print'
        self.get_control().signal_print()


    def processSave(self):
        print 'Save'

        self.get_control().signal_save()


    def processQuit(self):
        print 'Quit'
        self.close()
        self.get_control().signal_quit()


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        print 'Close application'
           
#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    w = ImgGUIPlayer(None)
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
