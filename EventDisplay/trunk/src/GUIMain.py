
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMain...
#
#------------------------------------------------------------------------

"""Renders the main GUI in the event display application.

Following paragraphs provide detailed description of the module.

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
import GUIWhatToDisplay as guiwtd
import ConfigParameters as cp
import DrawEvent        as drev
import PrintHDF5        as printh5 # for my print_group(g,offset)

#---------------------
#  Class definition --
#---------------------
class GUIMain ( QtGui.QWidget ) :
    """Deals with the main GUI for the event display project

    Full description of this class.
    
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
        """Constructor.

        @param x   first parameter
        @param y   second parameter
        """

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(300, 300, 500, 300)
        self.setWindowTitle('HDF5 Event Display')

        cp.confpars.Print()
        print 'Current event number directly : %d ' % (cp.confpars.eventCurrent)

        self.drawev   = drev.DrawEvent()

        self.titFile  = QtGui.QLabel('File:')
        self.titCurr  = QtGui.QLabel('Current event:')
        self.titDraw  = QtGui.QLabel('Draw:')
        self.titSpace4= QtGui.QLabel('    ')
        self.titSpan  = QtGui.QLabel('Span:')
        self.titSlShow= QtGui.QLabel('Slide show:')

        self.fileEdit = QtGui.QLineEdit(cp.confpars.dirName+'/'+cp.confpars.fileName)
        self.numbEdit = QtGui.QLineEdit(str(cp.confpars.eventCurrent))
        self.spanEdit = QtGui.QLineEdit(str(cp.confpars.span))
        self.spanEdit.setMaximumWidth(45)
        self.numbEdit.setMaximumWidth(90)

        self.browse   = QtGui.QPushButton("Browse")    
        self.printfile= QtGui.QPushButton("Print file cont.")    
        self.display  = QtGui.QPushButton("What to display")
        self.config   = QtGui.QPushButton("Configuration")
        self.save     = QtGui.QPushButton("Save conf.")
        self.current  = QtGui.QPushButton("Current")
        self.previous = QtGui.QPushButton("Previous")
        self.next     = QtGui.QPushButton("Next")
        self.slideShow= QtGui.QPushButton("Slide show")
        self.start    = QtGui.QPushButton("Start")
        self.stop     = QtGui.QPushButton("Stop")
        self.selection= QtGui.QPushButton("Selection")
        self.exit     = QtGui.QPushButton("Exit")
        self.spaninc  = QtGui.QPushButton(u'\u25B6') # right-head triangle
        self.spandec  = QtGui.QPushButton(u'\u25C0') # left-head triangle
        self.spaninc.setMaximumWidth(20) 
        self.spandec.setMaximumWidth(20) 

        #lcd      = QtGui.QLCDNumber(self)
        #slider   = QtGui.QSlider(QtCore.Qt.Horizontal, self)        

        #self.next.setFocusPolicy(QtCore.Qt.NoFocus)
        #self.previous.setFocusPolicy(QtCore.Qt.NoFocus)

        hboxF = QtGui.QHBoxLayout()
        hboxF.addWidget(self.titFile)
        hboxF.addWidget(self.fileEdit)
        hboxF.addWidget(self.browse)

        hboxC = QtGui.QHBoxLayout()
        hboxC.addWidget(self.config)
        hboxC.addWidget(self.save)
        hboxC.addStretch(2)
        hboxC.addWidget(self.printfile)
        
        hboxE = QtGui.QHBoxLayout()
        hboxE.addWidget(self.selection)
        hboxE.addStretch(1)
        hboxE.addWidget(self.display)

        hboxT = QtGui.QHBoxLayout() 
        hboxT.addWidget(self.titCurr)
        hboxT.addWidget(self.numbEdit)
        hboxT.addStretch(1)     
        hboxT.addWidget(self.titSpan)
        hboxT.addWidget(self.spandec)
        hboxT.addWidget(self.spanEdit)
        hboxT.addWidget(self.spaninc)
        hboxT.addStretch(1)     

        hboxM = QtGui.QHBoxLayout()
        hboxM.addWidget(self.titDraw)
        hboxM.addWidget(self.previous)
        hboxM.addWidget(self.current)
        hboxM.addWidget(self.next)

        hboxS = QtGui.QHBoxLayout()
        hboxS.addWidget(self.titSlShow)
        hboxS.addWidget(self.start)
        hboxS.addWidget(self.stop)
        
        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.exit)

        #hboxL = QtGui.QHBoxLayout()
        #hboxL.addWidget(self.lcd)
        #hboxL.addWidget(self.slider)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hboxF)
        vbox.addStretch(1)     
        vbox.addLayout(hboxC)
        vbox.addStretch(1)     
        vbox.addLayout(hboxE)
        vbox.addStretch(1)     
        vbox.addLayout(hboxT)
        vbox.addLayout(hboxM)
        vbox.addLayout(hboxS)
        #vbox.addLayout(hboxL)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        #self.connect(self.slider, QtCore.SIGNAL('valueChanged(int)'), lcd, QtCore.SLOT('display(int)') )
        #self.connect(self,   QtCore.SIGNAL('closeGUIApp()'), QtCore.SLOT('close()') )
        #self.connect(self.exit,     QtCore.SIGNAL('clicked()'), QtCore.SLOT('close()') )
        self.connect(self.exit,      QtCore.SIGNAL('clicked()'), self.processQuit )
        self.connect(self.browse,    QtCore.SIGNAL('clicked()'), self.processBrowse )
        self.connect(self.next,      QtCore.SIGNAL('clicked()'), self.incrimentEventNo )
        self.connect(self.previous,  QtCore.SIGNAL('clicked()'), self.decrimentEventNo )
        self.connect(self.current,   QtCore.SIGNAL('clicked()'), self.currentEventNo )
        self.connect(self.start,     QtCore.SIGNAL('clicked()'), self.processStart )

        self.connect(self.stop,      QtCore.SIGNAL('clicked()'),  self.processStop )
        #self.connect(self.stop,      QtCore.SIGNAL('pressed()'),  self.processStop )
        #self.connect(self.stop,      QtCore.SIGNAL('released()'), self.processStop )

        self.connect(self.display,   QtCore.SIGNAL('clicked()'), self.processDisplay )
        self.connect(self.save,      QtCore.SIGNAL('clicked()'), self.processSave )
        self.connect(self.config,    QtCore.SIGNAL('clicked()'), self.processConfig )
        self.connect(self.printfile, QtCore.SIGNAL('clicked()'), self.processPrint )
        self.connect(self.selection, QtCore.SIGNAL('clicked()'), self.processSelection )
        self.connect(self.spaninc,   QtCore.SIGNAL('clicked()'), self.processSpaninc )
        self.connect(self.spandec,   QtCore.SIGNAL('clicked()'), self.processSpandec )

        #self.setFocus()
        #self.resize(500, 300)
        print 'End of init\n'

    #-------------------
    # Private methods --
    #-------------------

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)  
        self.drawArt(qp)
        qp.end()

    def drawArt(self, qp):
       #pen = QtGui.QPen(QtGui.QColor(255, 100, 0, 100), 2, QtCore.Qt.SolidLine)
        pen = QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        size = self.size()
        YfrD = 0.84
        YfrU = 0.46
        qp.setBrush(QtGui.QColor(0, 15, 55, 55))
        qp.drawRect(5, YfrU*size.height(), size.width()-10, 0.38*size.height())
        #Rx = Ry = 10
        #qp.drawRoundedRect( 5, YfrU*size.height(), size.width()-10, 0.38*size.height(), Rx, Ry)
        #qp.drawLine       (12, YfrD*size.height(), size.width()-12, YfrU*size.height())
        #qp.drawLine       (12, YfrU*size.height(), size.width()-12, YfrU*size.height())

    def processSelection(self):
        print 'Selection\n'

    def processConfig(self):
        print 'Configuration\n'

    def processSave(self):
        print 'Save\n'

    def processPrint(self):
        fname = cp.confpars.dirName+'/'+cp.confpars.fileName
        print 'Print structure of the HDF5 file:\n %s' % (fname)
        printh5.print_hdf5_file_structure(fname)

    def processQuit(self):
        print 'Quit\n'
        self.drawev.stopDrawEvent()
        self.SHowIsOn = False
        self.close()
        
    def processStart(self):
        print 'Start slide show\n'
        cp.confpars.eventCurrent = int(self.numbEdit.displayText())
        cp.confpars.span         = int(self.spanEdit.displayText())
        self.SHowIsOn            = True
        eventStart = cp.confpars.eventCurrent
        eventEnd   = cp.confpars.eventCurrent + 1000*cp.confpars.span
        mode = 0 # for slide show

        while (self.SHowIsOn) :
            if cp.confpars.eventCurrent>eventEnd : break
            self.numbEdit.setText( str(cp.confpars.eventCurrent) )
            #print cp.confpars.eventCurrent
            #self.evloop.activeWindow ()
            QtGui.QApplication.processEvents()
            if not self.SHowIsOn : break
            #time.sleep(1) # in sec
            self.drawev.drawEvent(mode) # Draw everything for current event
            cp.confpars.eventCurrent+=cp.confpars.span

    def processStop(self):
        print 'Stop slide show\n'
        self.drawev.stopDrawEvent() 
        self.SHowIsOn = False
              
    def processBrowse(self):
        print 'Browse\n'
        str_path_file = str(self.fileEdit.displayText())
        cp.confpars.dirName,cp.confpars.fileName = os.path.split(str_path_file)
        print 'dirName  : %s' % (cp.confpars.dirName)         
        print 'fileName : %s' % (cp.confpars.fileName)         
        path_file = QtGui.QFileDialog.getOpenFileName(self,'Open file',cp.confpars.dirName)
        #fname = open(filename)
        #data = fname.read()
        #self.textEdit.setText(data)
        print path_file
        str_path_file = str(path_file)
        self.fileEdit.setText(str_path_file)
        cp.confpars.dirName,cp.confpars.fileName = os.path.split(str_path_file)
        print 'dirName  : %s' % (cp.confpars.dirName)         
        print 'fileName : %s' % (cp.confpars.fileName)         

    def processDisplay(self):
        print 'What to display?'
        self.guiwhat = guiwtd.GUIWhatToDisplay()
        self.guiwhat.show()

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
        cp.confpars.eventCurrent += cp.confpars.span
        self.numbEdit.setText( str(cp.confpars.eventCurrent) )
        mode = 1
        self.drawev.drawEvent(mode) # Draw everything for current event

    def decrimentEventNo(self):
        print 'Previous ',        
        cp.confpars.span = int(self.spanEdit.displayText())
        cp.confpars.eventCurrent = int(self.numbEdit.displayText())
        cp.confpars.eventCurrent -= cp.confpars.span
        if cp.confpars.eventCurrent<0 : cp.confpars.eventCurrent=0
        self.numbEdit.setText( str(cp.confpars.eventCurrent) )
        mode = 1
        self.drawev.drawEvent(mode) # Draw everything for current event
        #print cp.confpars.eventCurrent

    def currentEventNo(self):
        print 'Current ',
        cp.confpars.eventCurrent = int(self.numbEdit.displayText())
        mode = 1
        self.drawev.drawEvent(mode) # Draw everything for current event
        #print cp.confpars.eventCurrent

    def mousePressEvent(self, event):
        print 'Quit\n'
        self.emit(QtCore.SIGNAL('closeGUIApp()'))
                
    #def keyPressEvent(self, event):
    #    if event.key() == QtCore.Qt.Key_Escape:
    #        self.SHowIsOn = False

    def keyPressEvent(self, event):
        print 'event.key() = %s' % (event.key())
        if event.key() == QtCore.Qt.Key_Escape:
    #        self.close()
            self.SHowIsOn = False    

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
