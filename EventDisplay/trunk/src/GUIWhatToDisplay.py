#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplay...
#
#------------------------------------------------------------------------

"""Generates GUI to select information for rendaring in the event display.

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

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp
import GUIWhatToDisplayForImage as wtdIM 
import GUIWhatToDisplayForCSpad as wtdCS 

#---------------------
#  Class definition --
#---------------------
class GUIWhatToDisplay ( QtGui.QWidget ) :
    """Provides GUI to select information for rendering.

    Detailed description should be here...
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

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(370, 350, 500, 300)
        self.setWindowTitle('What to display?')

        titFont = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        self.titImage    = QtGui.QLabel('Image')
        self.titCSpad    = QtGui.QLabel('SCpad')
        self.titWaveform = QtGui.QLabel('Waveform')

        #self.titImage.setTextFormat (2) # Qt.AutoText = 2
        self.titImage   .setFont (titFont) 
        self.titCSpad   .setFont (titFont) 
        self.titWaveform.setFont (titFont) 

        self.cboxIMImage    = QtGui.QCheckBox('Image',    self)
        self.cboxIMSpectrum = QtGui.QCheckBox('Spectrum', self)

        self.cboxCSImage    = QtGui.QCheckBox('Image',    self)
        self.cboxCSSpectrum = QtGui.QCheckBox('Spectrum', self)

        self.cboxWFImage    = QtGui.QCheckBox('Image',    self)
        self.cboxWFSpectrum = QtGui.QCheckBox('Spectrum', self)

        if cp.confpars.imageImageIsOn       : self.cboxIMImage   .setCheckState(2)
        if cp.confpars.imageSpectrumIsOn    : self.cboxIMSpectrum.setCheckState(2)
        if cp.confpars.cspadImageIsOn       : self.cboxCSImage   .setCheckState(2)
        if cp.confpars.cspadSpectrumIsOn    : self.cboxCSSpectrum.setCheckState(2)
        if cp.confpars.waveformImageIsOn    : self.cboxWFImage   .setCheckState(1)
        if cp.confpars.waveformSpectrumIsOn : self.cboxWFSpectrum.setCheckState(1)


        #self.cb.setFocusPolicy(QtCore.Qt.NoFocus) 
        #self.cb2.move(10, 50)
        #self.cb.move(10, 10)
        #self.cb.toggle() # set the check mark in the check box

        self.butClose      = QtGui.QPushButton("Close window")

        self.butIMOptions  = QtGui.QPushButton("Options for Image")
        self.butCSOptions  = QtGui.QPushButton("Options for CSpad")
        self.butWFOptions  = QtGui.QPushButton("Options for Waveform")

        gridIM = QtGui.QGridLayout()
        gridIM.addWidget(self.titImage,       0, 0)
        gridIM.addWidget(self.cboxIMImage,    1, 0)
        gridIM.addWidget(self.cboxIMSpectrum, 2, 0)
        gridIM.addWidget(self.butIMOptions,   0, 1)
        
        gridCS = QtGui.QGridLayout()
        gridCS.addWidget(self.titCSpad,       0, 0)
        gridCS.addWidget(self.cboxCSImage,    1, 0)
        gridCS.addWidget(self.cboxCSSpectrum, 2, 0)
        gridCS.addWidget(self.butCSOptions,   0, 1)
        
        gridWF = QtGui.QGridLayout()
        gridWF.addWidget(self.titWaveform,    0, 0)
        gridWF.addWidget(self.cboxWFImage,    1, 0)
        gridWF.addWidget(self.cboxWFSpectrum, 2, 0)
        gridWF.addWidget(self.butWFOptions,   0, 1)
        
        hboxC = QtGui.QHBoxLayout()
        hboxC.addStretch(1)
        hboxC.addWidget(self.butClose)
        
        vbox = QtGui.QVBoxLayout()
        #guiwtdIM = wtdIM.GUIWhatToDisplayForImage(self) # THIS
        #vbox.addLayout(guiwtdIM.getVBoxForLayout())     # WORKS
        vbox.addLayout(gridCS) 
        vbox.addStretch(1)     
        vbox.addLayout(gridIM) 
        vbox.addStretch(1)     
        vbox.addLayout(gridWF) 
        vbox.addStretch(1)     
        vbox.addLayout(hboxC)

        self.setLayout(vbox)

        self.connect(self.butClose,            QtCore.SIGNAL('clicked()'),         self.processClose )
        self.connect(self.butCSOptions,        QtCore.SIGNAL('clicked()'),         self.processCSOptions )
        self.connect(self.butIMOptions,        QtCore.SIGNAL('clicked()'),         self.processIMOptions )
        self.connect(self.butWFOptions,        QtCore.SIGNAL('clicked()'),         self.processWFOptions )

        self.connect(self.cboxIMImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxIMImage)
        self.connect(self.cboxIMSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxIMSpectrum)
        self.connect(self.cboxCSImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSImage)
        self.connect(self.cboxCSSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSSpectrum)
        self.connect(self.cboxWFImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxWFImage)
        self.connect(self.cboxWFSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxWFSpectrum)


    def closeEvent(self, event):
        print 'closeEvent'
        self.processClose()


    def processClose(self):
        print 'Close window'
        cp.confpars.wtdWindowIsOpen = False
        self.close()

    def processCSOptions(self):
        print 'CSOptions'
        self.guiwtdCS = wtdCS.GUIWhatToDisplayForCSpad()
        self.guiwtdCS.setParentWidget(self)
        self.guiwtdCS.move(self.pos().__add__(QtCore.QPoint(20,40))) # open window with offset w.r.t. parent
        #self.guiwtdCS.show()

    def processIMOptions(self):
        print 'IMOptions'
        self.guiwtdIM = wtdIM.GUIWhatToDisplayForImage()
        self.guiwtdIM.setParentWidget(self)
        self.guiwtdIM.move(self.pos().__add__(QtCore.QPoint(20,100))) # open window with offset w.r.t. parent

        #self.guiwtdIM.show()
        
    def processWFOptions(self):
        print 'WFOptions'


    def processCBoxIMImage(self, value):
        if self.cboxIMImage.isChecked():
            cp.confpars.imageImageIsOn = True
        else:
            cp.confpars.imageImageIsOn = False


    def processCBoxIMSpectrum(self, value):
        if self.cboxIMSpectrum.isChecked():
            cp.confpars.imageSpectrumIsOn = True
        else:
            cp.confpars.imageSpectrumIsOn = False


    def processCBoxCSImage(self, value):
        if self.cboxCSImage.isChecked():
            cp.confpars.cspadImageIsOn = True
        else:
            cp.confpars.cspadImageIsOn = False


    def processCBoxCSSpectrum(self, value):
        if self.cboxCSSpectrum.isChecked():
            cp.confpars.cspadSpectrumIsOn = True
        else:
            cp.confpars.cspadSpectrumIsOn = False


    def processCBoxWFImage(self, value):
        if self.cboxWFImage.isChecked():
            cp.confpars.waveformImageIsOn = True
        else:
            cp.confpars.waveformImageIsOn = False


    def processCBoxWFSpectrum(self, value):
        if self.cboxWFSpectrum.isChecked():
            cp.confpars.waveformSpectrumIsOn = True
        else:
            cp.confpars.waveformSpectrumIsOn = False


#    def paintEvent(self, e):
#        qp = QtGui.QPainter()
#        qp.begin(self)
#        self.drawWidget(qp)
#        qp.end()

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIWhatToDisplay()
    ex.show()
    app.exec_()
#-----------------------------

