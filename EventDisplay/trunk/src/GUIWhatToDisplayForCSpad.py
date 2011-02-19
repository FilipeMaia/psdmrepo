#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplayForCSpad...
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
import ConfigParameters     as cp
import GUISelectQuadAndPair as guiquadpair

#---------------------
#  Class definition --
#---------------------
class GUIWhatToDisplayForCSpad ( QtGui.QWidget ) :
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

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Adjust CSpad Parameters')

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(True)

        titFont12 = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        titFont10 = QtGui.QFont("Sans Serif", 10, QtGui.QFont.Bold)

        self.titCSpad            = QtGui.QLabel('SCpad')
        self.titCSImage          = QtGui.QLabel('Image plot')
        self.titCSSpectrum       = QtGui.QLabel('Spectrum')
        self.titNWin             = QtGui.QLabel('N detector images:')

        self.titCSpad      .setFont (titFont12) 
        self.titCSImage    .setFont (titFont10)   
        self.titCSSpectrum .setFont (titFont10)

        self.titCSAmpDash        = QtGui.QLabel('-')
        self.titCSImageAmin      = QtGui.QLabel('Amin:')
        self.titCSImageAmax      = QtGui.QLabel('Amax:')

        self.titCSSpectrumAmin   = QtGui.QLabel('Amin:')
        self.titCSSpectrumAmax   = QtGui.QLabel('Amax:')
        self.titCSSpectrumNbins  = QtGui.QLabel('N bins:')
        self.titCSSpectrumAlim   = QtGui.QLabel('Slider range:')

        self.editCSImageAmin     = QtGui.QLineEdit(str(cp.confpars.cspadImageAmin))
        self.editCSImageAmax     = QtGui.QLineEdit(str(cp.confpars.cspadImageAmax))
        self.editCSImageAmin     .setMaximumWidth(45)
        self.editCSImageAmax     .setMaximumWidth(45)

        self.editCSSpectrumAmin  = QtGui.QLineEdit(str(cp.confpars.cspadSpectrumAmin))
        self.editCSSpectrumAmax  = QtGui.QLineEdit(str(cp.confpars.cspadSpectrumAmax))
        self.editCSSpectrumNbins = QtGui.QLineEdit(str(cp.confpars.cspadSpectrumNbins))
        self.editCSSpectrumAmin  .setMaximumWidth(45)
        self.editCSSpectrumAmax  .setMaximumWidth(45)
        self.editCSSpectrumNbins .setMaximumWidth(45)

        self.editCSAmpRaMin      = QtGui.QLineEdit(str(cp.confpars.cspadAmplitudeRaMin))
        self.editCSAmpRange      = QtGui.QLineEdit(str(cp.confpars.cspadAmplitudeRange))
        self.editCSAmpRaMin      .setMaximumWidth(50)
        self.editCSAmpRange      .setMaximumWidth(50)

        self.sliderCSAmin  = QtGui.QSlider(QtCore.Qt.Horizontal, self)        
        self.sliderCSAmax  = QtGui.QSlider(QtCore.Qt.Horizontal, self)        
        self.setAmplitudeRange(cp.confpars.cspadAmplitudeRaMin, cp.confpars.cspadAmplitudeRange)
        self.sliderCSAmin.setValue(cp.confpars.cspadSpectrumAmin)
        self.sliderCSAmax.setValue(cp.confpars.cspadSpectrumAmax)
       #self.sliderCSAmin.setPageStep((cp.confpars.cspadAmplitudeRange - cp.confpars.cspadAmplitudeRaMin)/100)
       #self.sliderCSAmax.setTickPosition(QtGui.QSlider.TicksBelow)

        self.char_expand = u'\u25BE' # down-head triangle
        self.butMenuNWin = QtGui.QPushButton(str(cp.confpars.cspadImageNWindows) + self.char_expand)
        self.butMenuNWin.setMaximumWidth(30)

        self.listActMenuNWin = []
        self.popupMenuNWin = QtGui.QMenu()
        for nwin in range(1,cp.confpars.cspadImageNWindowsMax+1) :
            self.listActMenuNWin.append(self.popupMenuNWin.addAction(str(nwin)))

        self.butClose = QtGui.QPushButton("Close window")

        self.wquadpair = guiquadpair.GUISelectQuadAndPair()

        hboxCS01 = QtGui.QHBoxLayout()
        hboxCS01.addWidget(self.titCSpad)        
        hboxCS01.addStretch(1) 
        hboxCS01.addWidget(self.wquadpair)
                           
        hboxCS02 = QtGui.QHBoxLayout()
        hboxCS02.addWidget(self.titCSSpectrumAlim) 
        hboxCS02.addWidget(self.editCSAmpRaMin) 
        hboxCS02.addWidget(self.titCSAmpDash)
        hboxCS02.addWidget(self.editCSAmpRange) 
        hboxCS02.addWidget(self.sliderCSAmin)        
        hboxCS02.addWidget(self.sliderCSAmax)        

        hboxCS03 = QtGui.QHBoxLayout()
        hboxCS03.addWidget(self.titNWin)
        hboxCS03.addWidget(self.butMenuNWin)
        hboxCS03.addStretch(1) 

        gridCS = QtGui.QGridLayout()
        gridCS.addWidget(self.titCSImage,      0, 0)
        gridCS.addWidget(self.titCSImageAmin,  0, 1)
        gridCS.addWidget(self.editCSImageAmin, 0, 2)
        gridCS.addWidget(self.titCSImageAmax,  0, 3)
        gridCS.addWidget(self.editCSImageAmax, 0, 4)
        
        gridCS.addWidget(self.titCSSpectrum,       1, 0)
        gridCS.addWidget(self.titCSSpectrumAmin,   1, 1)
        gridCS.addWidget(self.editCSSpectrumAmin,  1, 2)
        gridCS.addWidget(self.titCSSpectrumAmax,   1, 3)
        gridCS.addWidget(self.editCSSpectrumAmax,  1, 4)
        gridCS.addWidget(self.titCSSpectrumNbins,  1, 5)
        gridCS.addWidget(self.editCSSpectrumNbins, 1, 6)
    
        hboxC = QtGui.QHBoxLayout()
        hboxC.addStretch(1)
        hboxC.addWidget(self.butClose)
        
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(hboxCS01)
        self.vbox.addLayout(gridCS) 
        self.vbox.addLayout(hboxCS02)
        self.vbox.addStretch(1)     
        self.vbox.addLayout(hboxCS03)

        if parent == None :
            #self.vbox.addLayout(hboxC)
            self.setLayout(self.vbox)
            self.show()

        self.connect(self.butClose,            QtCore.SIGNAL('clicked()'),         self.processClose )
        self.connect(self.butMenuNWin,         QtCore.SIGNAL('clicked()'),         self.processMenuNWin )

        #self.connect(self.cboxCSImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSImage)
        #self.connect(self.cboxCSSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSSpectrum)

        self.connect(self.sliderCSAmin,        QtCore.SIGNAL('valueChanged(int)'), self.processSliderCSAmin )
        self.connect(self.sliderCSAmax,        QtCore.SIGNAL('valueChanged(int)'), self.processSliderCSAmax )

        self.connect(self.editCSImageAmin,     QtCore.SIGNAL('editingFinished ()'), self.processEditCSImageAmin )
        self.connect(self.editCSImageAmax,     QtCore.SIGNAL('editingFinished ()'), self.processEditCSImageAmax )
        self.connect(self.editCSSpectrumAmin,  QtCore.SIGNAL('editingFinished ()'), self.processEditCSSpectrumAmin )
        self.connect(self.editCSSpectrumAmax,  QtCore.SIGNAL('editingFinished ()'), self.processEditCSSpectrumAmax )
        self.connect(self.editCSSpectrumNbins, QtCore.SIGNAL('editingFinished ()'), self.processEditCSSpectrumNbins )
        self.connect(self.editCSAmpRange,      QtCore.SIGNAL('editingFinished ()'), self.processEditCSAmplitudeRange )
        self.connect(self.editCSAmpRaMin,      QtCore.SIGNAL('editingFinished ()'), self.processEditCSAmplitudeRaMin )

        cp.confpars.wtdCSWindowIsOpen = True

    #-------------------
    # Private methods --
    #-------------------

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())
  
    def getVBoxForLayout(self):
        return self.vbox

    def setParentWidget(self,parent):
        self.parentWidget = parent

    def setAmplitudeRange(self,    amplitudeRaMin, amplitudeRange):
        self.sliderCSAmin.setRange(amplitudeRaMin, amplitudeRange)
        self.sliderCSAmax.setRange(amplitudeRaMin, amplitudeRange)

    def closeEvent(self, event):
        print 'closeEvent'
        self.processClose()

    def processClose(self):
        print 'Close window'
        cp.confpars.wtdCSWindowIsOpen = False
        self.close()

    def processMenuNWin(self):
        #print 'MenuNWin'
        actionSelected = self.popupMenuNWin.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        for action in self.listActMenuNWin :
            cp.confpars.cspadImageNWindows = int(actionSelected.text())
            if actionSelected == action : self.butMenuNWin.setText( str(cp.confpars.cspadImageNWindows) + self.char_expand )

    def processEditCSAmplitudeRange(self):
        print 'EditCSAmplitudeRange'
        cp.confpars.cspadAmplitudeRange = int(self.editCSAmpRange.displayText())
        self.setAmplitudeRange(cp.confpars.cspadAmplitudeRaMin, cp.confpars.cspadAmplitudeRange)

    def processEditCSAmplitudeRaMin(self):
        print 'EditCSAmplitudeRaMin'
        cp.confpars.cspadAmplitudeRaMin = int(self.editCSAmpRaMin.displayText())
        self.setAmplitudeRange(cp.confpars.cspadAmplitudeRaMin, cp.confpars.cspadAmplitudeRange)

    def processEditCSImageAmin(self):
        print 'EditCSImageAmin'
        cp.confpars.cspadImageAmin = int(self.editCSImageAmin.displayText())        

    def processEditCSImageAmax(self):
        print 'EditCSImageAmax'
        cp.confpars.cspadImageAmax = int(self.editCSImageAmax.displayText())        

    def processEditCSSpectrumAmin(self):
        print 'EditCSSpectrumAmin'
        cp.confpars.cspadSpectrumAmin = int(self.editCSSpectrumAmin.displayText())        
        #cp.confpars.cspadSpectrumNbins = cp.confpars.cspadSpectrumAmax - cp.confpars.cspadSpectrumAmin
        #self.editCSSpectrumNbins.setText( str(cp.confpars.cspadSpectrumNbins) )        

    def processEditCSSpectrumAmax(self):
        print 'EditCSSpectrumAmax'
        cp.confpars.cspadSpectrumAmax  = int(self.editCSSpectrumAmax.displayText())        
        #cp.confpars.cspadSpectrumNbins = cp.confpars.cspadSpectrumAmax - cp.confpars.cspadSpectrumAmin
        #self.editCSSpectrumNbins.setText( str(cp.confpars.cspadSpectrumNbins) )
        
    def processEditCSSpectrumNbins(self):
        print 'EditCSSpectrumNbins'
        cp.confpars.cspadSpectrumNbins = int(self.editCSSpectrumNbins.displayText())        

    def processEditCSQuad(self):
        print 'EditCSQuad',
        cp.confpars.cspadQuad = int(self.editCSQuad.displayText())        
        print cp.confpars.cspadQuad
        
    def processEditCSPair(self):
        print 'EditCSPair',
        cp.confpars.cspadPair = int(self.editCSPair.displayText())        
        print cp.confpars.cspadPair
        
    def processSliderCSAmin(self):
        #print 'SliderCSAmin',
        #print self.sliderCSAmin.value()
        value = self.sliderCSAmin.value()
        if value > cp.confpars.cspadSpectrumAmax :
            self.sliderCSAmax.setValue(value)
        cp.confpars.cspadImageAmin     = value
        cp.confpars.cspadSpectrumAmin  = value
        #cp.confpars.cspadSpectrumNbins = cp.confpars.cspadSpectrumAmax - value
        self.editCSImageAmin    .setText( str(value) )
        self.editCSSpectrumAmin .setText( str(value) )
        #self.editCSSpectrumNbins.setText( str(cp.confpars.cspadSpectrumNbins) )

    def processSliderCSAmax(self):
        #print 'SliderCSAmax',
        #print self.sliderIMAmax.value()
        value = self.sliderCSAmax.value()
        if value < cp.confpars.cspadSpectrumAmin :
            self.sliderCSAmin.setValue(value)
        cp.confpars.cspadSpectrumAmax  = value
        cp.confpars.cspadImageAmax     = value
        #cp.confpars.cspadSpectrumNbins = value - cp.confpars.cspadSpectrumAmin
        self.editCSImageAmax    .setText( str(value) )
        self.editCSSpectrumAmax .setText( str(value) )
        #self.editCSSpectrumNbins.setText( str(cp.confpars.cspadSpectrumNbins) )


    def processCBoxCSImage(self, value):
        if self.cboxCSImage.isChecked():
            cp.confpars.cspadImageIsOn = True
            #self.parentWidget.cboxCSImage   .setCheckState(2)
        else:
            cp.confpars.cspadImageIsOn = False
            #self.parentWidget.cboxCSImage   .setCheckState(0)


    def processCBoxCSSpectrum(self, value):
        if self.cboxCSSpectrum.isChecked():
            cp.confpars.cspadSpectrumIsOn = True
            #self.parentWidget.cboxCSSpectrum.setCheckState(2)
        else:
            cp.confpars.cspadSpectrumIsOn = False
            #self.parentWidget.cboxCSSpectrum.setCheckState(0)

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIWhatToDisplayForCSpad()
    ex.show()
    app.exec_()
#-----------------------------

