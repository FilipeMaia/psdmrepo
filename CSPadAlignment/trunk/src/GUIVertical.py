#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIVertical...
#
#------------------------------------------------------------------------

"""Module GUIVertical for CSPadAlignment package

CSPadAlignment package is intended to check quality of the CSPad alignment
using image of wires illuminated by flat field.
Shadow of wires are compared with a set of straight lines, which can be
interactively adjusted.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$
#------------------------------
#!/usr/bin/env python

import sys
import os

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)
import ImageParameters   as imp
import Draw              as drw


class GUIVertical( QtGui.QWidget ) :
    """Example of GUI with buttons, labels, input of numbers etc.""" 

    def __init__ (self, parent=None, app=None) :
        """Constructor"""

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 50, 30, 500)
        self.setWindowTitle('GUI with buttons')

        # see http://www.riverbankcomputing.co.uk/static/Docs/PyQt4/html/qframe.html
        self.frame = QtGui.QFrame(self)
        # Box, Pannel, VLine, HLine, NoFrame | Raised, Sunken
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setVisible(True)
        self.frame.setGeometry(self.rect())

        self.butPrint    = QtGui.QPushButton("Print Lines")    
        self.butSpectrum = QtGui.QPushButton("Draw Spectrum")
        self.butProfile  = QtGui.QPushButton("Draw Profile")
        self.butDeviation= QtGui.QPushButton("Draw Deviat.")
        self.butReadPars = QtGui.QPushButton("Read pars")
        self.butSavePars = QtGui.QPushButton("Save pars")
        self.butPrintPars= QtGui.QPushButton("Print pars")
        self.butExit     = QtGui.QPushButton("Exit")

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.butPrint    )
        self.vbox.addWidget(self.butSpectrum )
        self.vbox.addWidget(self.butProfile  )
        self.vbox.addWidget(self.butDeviation)
        self.vbox.addWidget(self.butPrintPars)
        self.vbox.addWidget(self.butSavePars )
        self.vbox.addWidget(self.butReadPars )
        self.vbox.addWidget(self.butExit     )
        
        if parent == None :
            self.setLayout(self.vbox)

        self.connect(self.butPrint  ,   QtCore.SIGNAL('clicked()'), self.processPrint    )
        self.connect(self.butSpectrum,  QtCore.SIGNAL('clicked()'), self.processSpectrum )
        self.connect(self.butProfile,   QtCore.SIGNAL('clicked()'), self.processProfile  )
        self.connect(self.butDeviation, QtCore.SIGNAL('clicked()'), self.processDeviation)
        self.connect(self.butReadPars,  QtCore.SIGNAL('clicked()'), self.processReadPars )
        self.connect(self.butSavePars,  QtCore.SIGNAL('clicked()'), self.processSavePars )
        self.connect(self.butPrintPars, QtCore.SIGNAL('clicked()'), self.processPrintPars)
        self.connect(self.butExit,      QtCore.SIGNAL('clicked()'), self.processExit     )
        print 'End of init\n'


    def getBoxLayout(self):
        print 'getBoxLayout'
        return self.vbox
    
    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def processExit(self):
        #print 'Exit'
        self.close() # This call closeEvent()

    def closeEvent(self, event):
        #print 'closeEvent'
        imp.impars.gmw.on_quit() # close main window

    def processProfile(self):
        #print 'Profile'
        imp.impars.msgw.addMessage('Draw the image profile')        
        drw.draw.drawImageProfileAlongLine()

    def processDeviation(self):
        #print 'Deviation'
        imp.impars.msgw.addMessage('Draw deviation form lines')        
        drw.draw.plotDeviationFromLines()

    def processPrint(self):
        print 'Print line coordinates'
        imp.impars.msgw.addMessage('Print line coordinates')        
        imp.impars.printLineCoordinates()

    def processReadPars(self):
        #print 'ReadPars'
        imp.impars.msgw.addMessage('Read parameters from file')
        imp.impars.readParameters()

    def processSavePars(self):
        #print 'SavePars'
        imp.impars.msgw.addMessage('Save parameters in file')
        imp.impars.writeParameters()

    def processPrintPars(self):
        #print 'PrintPars'
        imp.impars.msgw.addMessage('Print currrent parameters')
        imp.impars.printParameters()
              
    def processSpectrum(self):
        #print 'Spectrum'
        imp.impars.msgw.addMessage('Draw spectrum for image')        
        drw.draw.drawSpectrum(imp.impars.arr)

    def mousePressEvent(self, event):
        print 'Do not click on mouse just for fun!\n'
        print 'event.button() = %s at position' % (event.button()),        
        #print (event.pos()),
        print ' x=%d, y=%d' % (event.x(),event.y()),        
        print ' global x=%d, y=%d' % (event.globalX(),event.globalY())
        #self.emit(QtCore.SIGNAL('closeGUIApp()'))

    #http://doc.qt.nokia.com/4.6/qt.html#Key-enum
    def keyPressEvent(self, event):
        print 'event.key() = %s' % (event.key())
        if event.key() == QtCore.Qt.Key_Escape:
            print 'Escape'

        if event.key() == QtCore.Qt.Key_B:
            print 'event.key() = %s' % (QtCore.Qt.Key_B)

        if event.key() == QtCore.Qt.Key_Return:
            print 'event.key() = Return'

        if event.key() == QtCore.Qt.Key_Home:
            print 'event.key() = Home'

#----------------------------------

def main():
    app = QtGui.QApplication(sys.argv)
    widget = GUIVertical()
    widget.show()
    app.exec_()

if __name__ == '__main__':
    main()

#----------------------------------
