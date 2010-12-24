#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMain...
#
#------------------------------------------------------------------------

"""Renders the main GUI in the event display application.

Following paragraphs provide detailed description of the module, its
contents and usage. This is a template module (or module template:)
which will be used by programmers to create new Python modules.
This is the "library module" as opposed to executable module. Library
modules provide class definitions or function definitions, but these
scripts cannot be run by themselves.

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
from PyQt4 import QtGui, QtCore
import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------
import GUIWhatToDisplay as guiwtd

#---------------------
#  Class definition --
#---------------------
class GUIMain ( QtGui.QWidget ) :
    """Brief description of a class.

    Full description of this class. The whole purpose of this class is 
    to serve as an example for SIT users. It shows the structure of
    the code inside the class. Class can have class (static) variables, 
    which can be private or public. It is good idea to define constructor 
    for your class (in Python there is only one constructor). Put your 
    public methods after constructor, and private methods after public.

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
    def __init__ (self, parent=None) :
        """Constructor.

        Explanation of what it does. So it does that and that, and also 
        that, but only if x is equal to that and y is not None.

        @param x   first parameter
        @param y   second parameter
        """

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(300, 300, 500, 300)
        self.setWindowTitle('HDF5 Event Display')
       
        self.dirName         = '/reg/d/psdm/XPP/xppcom10/hdf5/'
        self.dirName         = '/home'
        self.dirName         = './'
        self.fileName        = 'Click on "Browse" to select the file with exents ->'
        self.eventCurrentInt = 1
        self.span            = 1

        self.titFile  = QtGui.QLabel('File:')
        self.titCurr  = QtGui.QLabel('Current event:')
        self.titDraw  = QtGui.QLabel('Draw:')
        self.titSpace4= QtGui.QLabel('    ')
        self.titSpan  = QtGui.QLabel('Span:')
        self.titSlShow= QtGui.QLabel('Slide show:')

        self.fileEdit = QtGui.QLineEdit(self.fileName)
        self.numbEdit = QtGui.QLineEdit(str(self.eventCurrentInt))
        self.spanEdit = QtGui.QLineEdit(str(self.span))
        self.spanEdit.setMaximumWidth(45)
        self.numbEdit.setMaximumWidth(90)

        self.browse   = QtGui.QPushButton("Browse")    
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
        hboxC.addStretch(1)
        
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
        self.connect(self.stop,      QtCore.SIGNAL('clicked()'), self.processStop )
        self.connect(self.display,   QtCore.SIGNAL('clicked()'), self.processDisplay )
        self.connect(self.save,      QtCore.SIGNAL('clicked()'), self.processSave )
        self.connect(self.config,    QtCore.SIGNAL('clicked()'), self.processConfig )
        self.connect(self.selection, QtCore.SIGNAL('clicked()'), self.processSelection )
        self.connect(self.spaninc,   QtCore.SIGNAL('clicked()'), self.processSpaninc )
        self.connect(self.spandec,   QtCore.SIGNAL('clicked()'), self.processSpandec )

        #self.setFocus()
        #self.resize(500, 300)
        print 'End of init\n'

    #-------------------
    # Private methods --
    #-------------------

    def processSelection(self):
        print 'Selection\n'

    def processConfig(self):
        print 'Configuration\n'

    def processSave(self):
        print 'Save\n'

    def processQuit(self):
        print 'Quit\n'
        self.close()

    def processStart(self):
        print 'Start slide show\n'
        self.eventCurrentInt = int(self.numbEdit.displayText())
        self.span            = int(self.spanEdit.displayText())
        self.SHowIsOn = True
        eventStart = self.eventCurrentInt
        eventEnd   = self.eventCurrentInt + 100*self.span
        for self.eventCurrentInt in range(eventStart,eventEnd,self.span):
            print self.eventCurrentInt
            time.sleep(1) # in sec
            if not self.SHowIsOn : break

    def processStop(self):
        print 'Stop slide show\n'
        self.SHowIsOn = False
              
    def processBrowse(self):
        print 'Browse\n'
        #self.fileName = self.fileEdit.displayText()
        self.fileName = QtGui.QFileDialog.getOpenFileName(self,'Open file',self.dirName)
        #fname = open(filename)
        #data = fname.read()
        #self.textEdit.setText(data)
        print self.fileName
        self.fileEdit.setText(self.fileName)

    def processDisplay(self):
        print 'What to display?'
        self.guiwhat = guiwtd.GUIWhatToDisplay()
        self.guiwhat.show()
#        self.guiwhat.repaint()

    def processSpaninc(self):
        print 'Spaninc ',
        self.span = int(self.spanEdit.displayText())
        self.span+=1
        self.spanEdit.setText(str(self.span))
        print self.span

    def processSpandec(self):
        print 'Spandec ',
        self.span = int(self.spanEdit.displayText())
        self.span-=1
        if self.span<1 : self.span=1
        self.spanEdit.setText(str(self.span))
        print self.span

    def incrimentEventNo(self):
        print 'Next ',
        self.span = int(self.spanEdit.displayText())
        self.spanEdit.setText(str(self.span))
        self.eventCurrentInt = int(self.numbEdit.displayText())
        self.eventCurrentInt += self.span
        self.numbEdit.setText( str(self.eventCurrentInt) )
        print self.eventCurrentInt 

    def decrimentEventNo(self):
        print 'Previous ',        
        self.span = int(self.spanEdit.displayText())
        self.spanEdit.setText(str(self.span))
        self.eventCurrentInt = int(self.numbEdit.displayText())
        self.eventCurrentInt -= self.span
        if self.eventCurrentInt<0 : self.eventCurrentInt=0
        self.numbEdit.setText( str(self.eventCurrentInt) )
        print self.eventCurrentInt

    def currentEventNo(self):
        print 'Current ',
        self.eventCurrentInt = int(self.numbEdit.displayText())
        print self.eventCurrentInt

    def mousePressEvent(self, event):
        print 'Quit\n'
        self.emit(QtCore.SIGNAL('closeGUIApp()'))
                
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.SHowIsOn = False

#    def keyPressEvent(self, event):
#        if event.key() == QtCore.Qt.Key_Escape:
#            self.close()


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
