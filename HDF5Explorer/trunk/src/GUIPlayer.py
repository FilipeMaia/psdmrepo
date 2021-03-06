
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIPlayer...
#
#------------------------------------------------------------------------

"""GUI which handles the event player buttons in the HDF5Explorer project.

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
class GUIPlayer ( QtGui.QWidget ) :
    """GUI which handles the event player buttons

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
    def __init__ (self, parent=None, app=None) :
        """Constructor"""

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(370, 10, 500, 150)
        self.setWindowTitle('Event player')
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        #cp.confpars.readParameters()
        #if not cp.confpars.readParsFromFileAtStart :
        #    cp.confpars.setDefaultParameters()
        #cp.confpars.Print()
        #print 'Current event number directly : %d ' % (cp.confpars.eventCurrent)

        # see http://www.riverbankcomputing.co.uk/static/Docs/PyQt4/html/qframe.html
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(True)

        self.drawev   = drev.DrawEvent(self)

        self.titCurr  = QtGui.QLabel('Current event:')
        self.titDraw  = QtGui.QLabel('Draw:')
        self.titSpan  = QtGui.QLabel('Increment:')
        self.titSlShow= QtGui.QLabel('Slide show:')
        #self.titOver  = QtGui.QLabel('over')
        #self.titEvents= QtGui.QLabel('events')

        self.numbEdit = QtGui.QLineEdit(str(cp.confpars.eventCurrent))
        self.spanEdit = QtGui.QLineEdit(str(cp.confpars.span))
        #self.avevEdit = QtGui.QLineEdit(str(cp.confpars.numEventsAverage))
        self.numbEdit.setMaximumWidth(90)
        self.spanEdit.setMaximumWidth(45)
        #self.avevEdit.setMaximumWidth(45)
        self.numbEdit.setValidator(QtGui.QIntValidator(0,10000000,self))
        self.spanEdit.setValidator(QtGui.QIntValidator(1,1000000,self))
        #self.avevEdit.setValidator(QtGui.QIntValidator(1,1000000,self))

        self.reset        = QtGui.QPushButton("Reset")
        self.current      = QtGui.QPushButton("Current")
        self.previous     = QtGui.QPushButton("Previous")
        self.next         = QtGui.QPushButton("Next")
        self.slideShow    = QtGui.QPushButton("Slide show")
        self.start        = QtGui.QPushButton("Start")
        self.stop         = QtGui.QPushButton("Stop")
        #self.butAverage   = QtGui.QPushButton("Average")
        #self.butCorr      = QtGui.QPushButton("Correlations")

        #self.butAverage   .setStyleSheet("background-color: rgb(230, 255, 230); color: rgb(0, 0, 0)")
        #self.butCorr      .setStyleSheet("background-color: rgb(255, 230, 255); color: rgb(0, 0, 0)")

        #self.closeplts= QtGui.QPushButton("Close plots")
        #self.exit     = QtGui.QPushButton("Exit")
        
        #self.spaninc  = QtGui.QPushButton(u'\u25B6') # right-head triangle
        #self.spandec  = QtGui.QPushButton(u'\u25C0') # left-head triangle
        #self.spaninc.setMaximumWidth(20) 
        #self.spandec.setMaximumWidth(20) 
        self.reset  .setMaximumWidth(50)   

        self.cboxSelection = QtGui.QCheckBox('Apply selection', self)
        if cp.confpars.selectionIsOn : self.cboxSelection.setCheckState(2)
 
        hboxT = QtGui.QHBoxLayout() 
        hboxT.addWidget(self.titCurr)
        hboxT.addWidget(self.numbEdit)
        hboxT.addStretch(1)     
        hboxT.addWidget(self.titSpan)
        #hboxT.addWidget(self.spandec)
        hboxT.addWidget(self.spanEdit)
        #hboxT.addWidget(self.spaninc)
        hboxT.addStretch(1)     
        hboxT.addWidget(self.reset)

        hboxM = QtGui.QHBoxLayout()
        hboxM.addWidget(self.titDraw)
        hboxM.addWidget(self.previous)
        hboxM.addWidget(self.current)
        hboxM.addWidget(self.next)

        hboxS = QtGui.QHBoxLayout()
        hboxS.addWidget(self.titSlShow)
        hboxS.addWidget(self.start)
        hboxS.addWidget(self.stop)
        #hboxS.addWidget(self.closeplts)

        hboxC = QtGui.QHBoxLayout()
        hboxC.addWidget(self.cboxSelection)        
        hboxC.addStretch(1)

        #hboxA = QtGui.QHBoxLayout()
        #hboxA.addWidget(self.butAverage)
        #hboxA.addWidget(self.titOver)
        #hboxA.addWidget(self.avevEdit)
        #hboxA.addWidget(self.titEvents)
        #hboxA.addStretch(2)
        #hboxA.addWidget(self.butCorr)

        #hbox = QtGui.QHBoxLayout()
        #hbox.addStretch(3)
        #hbox.addWidget(self.exit)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hboxC)
        vbox.addLayout(hboxT)
        vbox.addLayout(hboxM)
        vbox.addLayout(hboxS)
        #vbox.addLayout(hboxA)
        #vbox.addStretch(1)
        #vbox.addLayout(hbox)

        self.setLayout(vbox)

        #self.connect(self.exit,      QtCore.SIGNAL('clicked()'), self.processQuit )
        #self.connect(self.closeplts, QtCore.SIGNAL('clicked()'), self.processClosePlots )
        self.connect(self.next,      QtCore.SIGNAL('clicked()'), self.incrimentEventNo )
        self.connect(self.previous,  QtCore.SIGNAL('clicked()'), self.decrimentEventNo )
        self.connect(self.current,   QtCore.SIGNAL('clicked()'), self.currentEventNo )
        self.connect(self.start,     QtCore.SIGNAL('clicked()'), self.processStart )
        self.connect(self.stop,      QtCore.SIGNAL('clicked()'), self.processStop )
        self.connect(self.reset,     QtCore.SIGNAL('clicked()'), self.processReset )
        #self.connect(self.spaninc,   QtCore.SIGNAL('clicked()'), self.processSpaninc )
        #self.connect(self.spandec,   QtCore.SIGNAL('clicked()'), self.processSpandec )
        #self.connect(self.butAverage,QtCore.SIGNAL('clicked()'), self.processAverage )
        #self.connect(self.butCorr,   QtCore.SIGNAL('clicked()'), self.processCorrelations )
        #self.connect(self.avevEdit,  QtCore.SIGNAL('editingFinished ()'), self.processAverageEventsEdit )
        
        self.connect(self.numbEdit,      QtCore.SIGNAL('editingFinished ()'), self.processNumbEdit )
        self.connect(self.spanEdit,      QtCore.SIGNAL('editingFinished ()'), self.processSpanEdit )
        self.connect(self.cboxSelection, QtCore.SIGNAL('stateChanged(int)'),  self.processCBoxSelection)

        #self.setFocus()
        #self.resize(500, 300)
        #print 'End of init'
        cp.confpars.playerGUIIsOpen = True


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
        cp.confpars.playerGUIIsOpen = False


    def processQuit(self):
        #print 'Quit button'
        self.close()


    def processCorrelations(self):
        print 'Correlations'
        self.drawev.drawCorrelationPlots()


    def processAverage(self):
        print 'Start Average'
        #self.drawev.stopDrawEvent() 
        #self.SHowIsOn = False
        cp.confpars.eventCurrent     = int(self.numbEdit.displayText())
        cp.confpars.numEventsAverage = int(self.avevEdit.displayText())
        #cp.confpars.span             = int(self.spanEdit.displayText())

        self.drawev.averageOverEvents()
        
        #cp.confpars.eventCurrent +=1
        self.numbEdit.setText(str(cp.confpars.eventCurrent))


    def processAverageEventsEdit(self):    
        print 'AverageEventsEdit',
        cp.confpars.numEventsAverage = int(self.avevEdit.displayText())
        print 'Set numEventsAverage : ', cp.confpars.numEventsAverage        


    def processCBoxSelection(self, value):
        if self.cboxSelection.isChecked():
            cp.confpars.selectionIsOn = True
        else:
            cp.confpars.selectionIsOn = False


    def processStart(self):
        print 'Start slide show'
        cp.confpars.eventCurrent = int(self.numbEdit.displayText())
        cp.confpars.span         = int(self.spanEdit.displayText())
        self.SHowIsOn            = True
        self.drawev.startSlideShow() 


    def processStop(self):
        print 'Stop slide show'
        self.drawev.stopSlideShow() 
        self.SHowIsOn = False


    def processReset(self):
        print 'Reset'
        if self.resetColorIsSet : # swap the background color
            self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('white'))
            self.resetColorIsSet = False          
        else :
            self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('yellow'))
            self.resetColorIsSet = True          
        cp.confpars.span = 1
        self.spanEdit.setPalette(self.palette)
        self.spanEdit.setText(str(cp.confpars.span))
        cp.confpars.eventCurrent = 0
        self.numbEdit.setPalette(self.palette)
        self.numbEdit.setText(str(cp.confpars.eventCurrent))
        #time.sleep(5)
        #self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('white'))
        #self.spanEdit.setPalette(self.palette)
        
    def processSpaninc(self):
        print 'Spaninc ',
        cp.confpars.span = int(self.spanEdit.displayText())
        cp.confpars.span+=1
        self.spanEdit.setText(str(cp.confpars.span))
        print cp.confpars.span

    def processSpandec(self):
        print 'Spandec ',
        cp.confpars.span = int(self.spanEdit.displayText())
        cp.confpars.span-=1
        if cp.confpars.span<1 : cp.confpars.span=1
        self.spanEdit.setText(str(cp.confpars.span))
        print cp.confpars.span

    def incrimentEventNo(self):
        print 'Next ',
        cp.confpars.span = int(self.spanEdit.displayText())
        cp.confpars.eventCurrent = int(self.numbEdit.displayText())
        self.drawev.drawNextEvent(mode=1) # Draw everything for the next (selected) event
        self.numbEdit.setText( str(cp.confpars.eventCurrent) )

    def decrimentEventNo(self):
        print 'Previous ',        
        cp.confpars.span = int(self.spanEdit.displayText())
        cp.confpars.eventCurrent = int(self.numbEdit.displayText())
        self.drawev.drawPreviousEvent(mode=1) # Draw everything for the previous (selected) event
        self.numbEdit.setText( str(cp.confpars.eventCurrent) )

    def currentEventNo(self):
        print 'Current ',
        cp.confpars.eventCurrent = int(self.numbEdit.displayText())
        self.drawev.drawEvent(mode=1) # Draw everything for current event

    def processClosePlots(self):
        print 'Close plots',
        self.drawev.quitDrawEvent()

    def mousePressEvent(self, event):
        print 'Do not click on mouse just for fun!\n'
        print 'event.button() = %s at position' % (event.button()),        
        #print (event.pos()),
        print ' x=%d, y=%d' % (event.x(),event.y()),        
        print ' global x=%d, y=%d' % (event.globalX(),event.globalY())
        #self.emit(QtCore.SIGNAL('closeGUIApp()'))

    def processNumbEdit(self):    
        print 'NumbEdit'
        cp.confpars.eventCurrent = int(self.numbEdit.displayText())
        print 'Set eventCurrent : ', cp.confpars.eventCurrent        

    def processSpanEdit(self):    
        print 'SpanEdit'
        cp.confpars.span = int(self.spanEdit.displayText())
        print 'Set span         : ', cp.confpars.span

#http://doc.qt.nokia.com/4.6/qt.html#Key-enum
    def keyPressEvent(self, event):
        print 'event.key() = %s' % (event.key())
        if event.key() == QtCore.Qt.Key_Escape:
    #        self.close()
            self.SHowIsOn = False    

        if event.key() == QtCore.Qt.Key_B:
            print 'event.key() = %s' % (QtCore.Qt.Key_B)

        if event.key() == QtCore.Qt.Key_Return:
            print 'event.key() = Return'

            #self.processFileEdit()
            #self.processNumbEdit()
            #self.processSpanEdit()
            #self.currentEventNo()

        if event.key() == QtCore.Qt.Key_Home:
            print 'event.key() = Home'

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIPlayer()
    ex.show()
    app.exec_()
#-----------------------------
