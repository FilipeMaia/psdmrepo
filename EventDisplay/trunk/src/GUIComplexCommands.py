
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIComplexCommands...
#
#------------------------------------------------------------------------

"""GUI which handles the Average and Correlations buttons in the event display application.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

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

from PyQt4 import QtGui, QtCore
import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp
import DrawEvent        as drev

#---------------------
#  Class definition --
#---------------------
class GUIComplexCommands ( QtGui.QWidget ) :
    """GUI which handles the Average and Correlations buttons

    @see BaseClass
    @see OtherClass
    """

    #--------------------
    #  Class variables --
    #--------------------
    #publicStaticVariable = 0 
    #__privateStaticVariable = "A string"

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, parent=None, wplayer=None) :
        """Constructor"""

        self.wplayer = wplayer

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(370, 10, 500, 50)
        self.setWindowTitle('Average and Correlation')
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(True)

        self.drawev   = drev.DrawEvent(self)

        self.titComplex = QtGui.QLabel('Multi-events:')
        self.titOver    = QtGui.QLabel('over')
        self.titEvents  = QtGui.QLabel('events')

        self.avevEdit = QtGui.QLineEdit(str(cp.confpars.numEventsAverage))
        self.avevEdit.setMaximumWidth(45)
        self.avevEdit.setValidator(QtGui.QIntValidator(1,1000000,self))

        self.butAverage   = QtGui.QPushButton("Average")
        self.butCorr      = QtGui.QPushButton("Correlations")

        #self.butAverage   .setStyleSheet("background-color: rgb(0, 255, 0); color: rgb(0, 0, 0)")
        #self.butCorr      .setStyleSheet("background-color: magenta; color: rgb(0, 0, 0)")
        self.butAverage   .setStyleSheet("background-color: rgb(230, 255, 230); color: rgb(0, 0, 0)")
        self.butCorr      .setStyleSheet("background-color: rgb(255, 230, 255); color: rgb(0, 0, 0)")

        #self.closeplts= QtGui.QPushButton("Close plots")
        #self.exit     = QtGui.QPushButton("Exit")
        
        hboxA = QtGui.QHBoxLayout()
        hboxA.addWidget(self.titComplex)
        hboxA.addWidget(self.butAverage)
        hboxA.addWidget(self.titOver)
        hboxA.addWidget(self.avevEdit)
        hboxA.addWidget(self.titEvents)
        hboxA.addStretch(2)
        hboxA.addWidget(self.butCorr)

        self.setLayout(hboxA)

        self.connect(self.butAverage,QtCore.SIGNAL('clicked()'),          self.processAverage )
        self.connect(self.butCorr,   QtCore.SIGNAL('clicked()'),          self.processCorrelations )
        self.connect(self.avevEdit,  QtCore.SIGNAL('editingFinished ()'), self.processAverageEventsEdit )

        #self.setFocus()
        #self.resize(500, 300)
        #print 'End of init'

    #-------------------
    # Private methods --
    #-------------------

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def closeEvent(self, event):
        #print 'closeEvent'
        self.drawev.quitDrawEvent()
        self.SHowIsOn = False


    def processQuit(self):
        #print 'Quit button is clicked'
        self.close() # this call closeEvent()


    def processCorrelations(self):
        print 'Correlations'
        self.drawev.drawCorrelationPlots()


    def processAverage(self):
        print 'Start Average'
        cp.confpars.eventCurrent     = int(self.wplayer.numbEdit.displayText())
        cp.confpars.numEventsAverage = int(self.avevEdit.displayText())
        self.drawev.averageOverEvents()        
        self.wplayer.numbEdit.setText(str(cp.confpars.eventCurrent))


    def processAverageEventsEdit(self):    
        print 'AverageEventsEdit',
        cp.confpars.numEventsAverage = int(self.avevEdit.displayText())
        print 'Set numEventsAverage : ', cp.confpars.numEventsAverage        


    def processClosePlots(self):
        #print 'Close plots',
        self.drawev.quitDrawEvent()

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIComplexCommands()
    ex.show()
    app.exec_()
#-----------------------------
