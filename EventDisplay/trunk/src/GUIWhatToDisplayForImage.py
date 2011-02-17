#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplayForImage...
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

#---------------------
#  Class definition --
#---------------------
class GUIWhatToDisplayForImage ( QtGui.QWidget ) :
    """Provides GUI to select information for rendering."""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Adjust Image Parameters')

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(True)

        titFont12 = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        titFont10 = QtGui.QFont("Sans Serif", 10, QtGui.QFont.Bold)

        self.titImage            = QtGui.QLabel('Image')
        self.titIMImage          = QtGui.QLabel('Image plot')
        self.titIMSpectrum       = QtGui.QLabel('Spectrum')

        self.titImage      .setFont (titFont12) 
        self.titIMImage    .setFont (titFont10)   
        self.titIMSpectrum .setFont (titFont10)

        self.titIMImageAmin      = QtGui.QLabel('Amin:')
        self.titIMImageAmax      = QtGui.QLabel('Amax:')

        self.titIMAmpDash        = QtGui.QLabel('-')
        self.titIMSpectrumAmin   = QtGui.QLabel('Amin:')
        self.titIMSpectrumAmax   = QtGui.QLabel('Amax:')
        self.titIMSpectrumNbins  = QtGui.QLabel('N bins:')
        self.titIMSpectrumAlim   = QtGui.QLabel('Slider range:')

        self.editIMImageAmin     = QtGui.QLineEdit(str(cp.confpars.imageImageAmin))
        self.editIMImageAmax     = QtGui.QLineEdit(str(cp.confpars.imageImageAmax))
        self.editIMImageAmin     .setMaximumWidth(45)
        self.editIMImageAmax     .setMaximumWidth(45)

        self.editIMSpectrumAmin  = QtGui.QLineEdit(str(cp.confpars.imageSpectrumAmin))
        self.editIMSpectrumAmax  = QtGui.QLineEdit(str(cp.confpars.imageSpectrumAmax))
        self.editIMSpectrumNbins = QtGui.QLineEdit(str(cp.confpars.imageSpectrumNbins))
        self.editIMSpectrumAmin  .setMaximumWidth(45)
        self.editIMSpectrumAmax  .setMaximumWidth(45)
        self.editIMSpectrumNbins .setMaximumWidth(45)

        self.editIMAmpRaMin      = QtGui.QLineEdit(str(cp.confpars.imageAmplitudeRaMin))
        self.editIMAmpRange      = QtGui.QLineEdit(str(cp.confpars.imageAmplitudeRange))
        self.editIMAmpRaMin      .setMaximumWidth(50)
        self.editIMAmpRange      .setMaximumWidth(50)

        #self.cboxIMImage    = QtGui.QCheckBox('Image',    self)
        #self.cboxIMSpectrum = QtGui.QCheckBox('Spectrum', self)

        #if cp.confpars.imageImageIsOn       : self.cboxIMImage   .setCheckState(2)
        #if cp.confpars.imageSpectrumIsOn    : self.cboxIMSpectrum.setCheckState(2)

        self.sliderIMAmin  = QtGui.QSlider(QtCore.Qt.Horizontal, self)        
        self.sliderIMAmax  = QtGui.QSlider(QtCore.Qt.Horizontal, self)        
        self.setAmplitudeRange(cp.confpars.imageAmplitudeRaMin, cp.confpars.imageAmplitudeRange)
        self.sliderIMAmin.setValue(cp.confpars.imageSpectrumAmin)
        self.sliderIMAmax.setValue(cp.confpars.imageSpectrumAmax)
        
        self.butClose = QtGui.QPushButton("Close window")

        hboxIM01 = QtGui.QHBoxLayout()
        hboxIM01.addWidget(self.titImage)        
        hboxIM02 = QtGui.QHBoxLayout()
        #hboxIM02.addStretch(1)
        hboxIM02.addWidget(self.titIMSpectrumAlim) 
        hboxIM02.addWidget(self.editIMAmpRaMin)
        hboxIM02.addWidget(self.titIMAmpDash)
        hboxIM02.addWidget(self.editIMAmpRange) 
        hboxIM02.addWidget(self.sliderIMAmin)        
        hboxIM02.addWidget(self.sliderIMAmax)        

        gridIM = QtGui.QGridLayout()
       #gridIM.addWidget(self.cboxIMImage,     0, 0)
        gridIM.addWidget(self.titIMImage,      0, 0)
        gridIM.addWidget(self.titIMImageAmin,  0, 1)
        gridIM.addWidget(self.editIMImageAmin, 0, 2)
        gridIM.addWidget(self.titIMImageAmax,  0, 3)
        gridIM.addWidget(self.editIMImageAmax, 0, 4)
        
       #gridIM.addWidget(self.cboxIMSpectrum,      1, 0)
        gridIM.addWidget(self.titIMSpectrum,       1, 0)
        gridIM.addWidget(self.titIMSpectrumAmin,   1, 1)
        gridIM.addWidget(self.editIMSpectrumAmin,  1, 2)
        gridIM.addWidget(self.titIMSpectrumAmax,   1, 3)
        gridIM.addWidget(self.editIMSpectrumAmax,  1, 4)
        gridIM.addWidget(self.titIMSpectrumNbins,  1, 5)
        gridIM.addWidget(self.editIMSpectrumNbins, 1, 6)
        
        hboxC = QtGui.QHBoxLayout()
        hboxC.addStretch(1)
        hboxC.addWidget(self.butClose)
        
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(hboxIM01)
        self.vbox.addLayout(gridIM) 
        self.vbox.addLayout(hboxIM02)

        self.vbox.addStretch(1)     

        if parent == None :
            #self.vbox.addLayout(hboxC)
            self.setLayout(self.vbox)
            self.show()

        self.connect(self.butClose,            QtCore.SIGNAL('clicked()'),         self.processClose )

        #self.connect(self.cboxIMImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxIMImage)
        #self.connect(self.cboxIMSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxIMSpectrum)

        self.connect(self.sliderIMAmin,        QtCore.SIGNAL('valueChanged(int)'), self.processSliderIMAmin )
        self.connect(self.sliderIMAmax,        QtCore.SIGNAL('valueChanged(int)'), self.processSliderIMAmax )

        self.connect(self.editIMImageAmin,     QtCore.SIGNAL('editingFinished ()'), self.processEditIMImageAmin )
        self.connect(self.editIMImageAmax,     QtCore.SIGNAL('editingFinished ()'), self.processEditIMImageAmax )
        self.connect(self.editIMSpectrumAmin,  QtCore.SIGNAL('editingFinished ()'), self.processEditIMSpectrumAmin )
        self.connect(self.editIMSpectrumAmax,  QtCore.SIGNAL('editingFinished ()'), self.processEditIMSpectrumAmax )
        self.connect(self.editIMSpectrumNbins, QtCore.SIGNAL('editingFinished ()'), self.processEditIMSpectrumNbins )
        self.connect(self.editIMAmpRange,      QtCore.SIGNAL('editingFinished ()'), self.processEditIMAmplitudeRange )
        self.connect(self.editIMAmpRaMin,      QtCore.SIGNAL('editingFinished ()'), self.processEditIMAmplitudeRaMin )

        cp.confpars.wtdIMWindowIsOpen = True

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
        self.sliderIMAmin.setRange(amplitudeRaMin, amplitudeRange)
        self.sliderIMAmax.setRange(amplitudeRaMin, amplitudeRange)
        #self.sliderIMAmin.setTickInterval(0.02*amplitudeRange)
        #self.sliderIMAmax.setTickInterval(0.02*amplitudeRange)

    def closeEvent(self, event):
        print 'closeEvent'
        self.processClose()

    def processClose(self):
        print 'Close window'
        cp.confpars.wtdIMWindowIsOpen = False
        self.close()

    def processEditIMAmplitudeRange(self):
        print 'EditIMAmplitudeRange'
        cp.confpars.imageAmplitudeRange = int(self.editIMAmpRange.displayText())
        self.setAmplitudeRange(cp.confpars.imageAmplitudeRaMin, cp.confpars.imageAmplitudeRange)

    def processEditIMAmplitudeRaMin(self):
        print 'EditIMAmplitudeRaMin'
        cp.confpars.imageAmplitudeRaMin = int(self.editIMAmpRaMin.displayText())
        self.setAmplitudeRange(cp.confpars.imageAmplitudeRaMin, cp.confpars.imageAmplitudeRange)

    def processEditIMImageAmin(self):
        print 'EditIMImageAmin'
        cp.confpars.imageImageAmin = int(self.editIMImageAmin.displayText())        

    def processEditIMImageAmax(self):
        print 'EditIMImageAmax'
        cp.confpars.imageImageAmax = int(self.editIMImageAmax.displayText())        

    def processEditIMSpectrumAmin(self):
        print 'EditIMSpectrumAmin'
        cp.confpars.imageSpectrumAmin = int(self.editIMSpectrumAmin.displayText())        
        #cp.confpars.imageSpectrumNbins = cp.confpars.imageSpectrumAmax - cp.confpars.imageSpectrumAmin
        #self.editIMSpectrumNbins.setText( str(cp.confpars.imageSpectrumNbins) )        

    def processEditIMSpectrumAmax(self):
        print 'EditIMSpectrumAmax'
        cp.confpars.imageSpectrumAmax  = int(self.editIMSpectrumAmax.displayText())        
        #cp.confpars.imageSpectrumNbins = cp.confpars.imageSpectrumAmax - cp.confpars.imageSpectrumAmin
        #self.editIMSpectrumNbins.setText( str(cp.confpars.imageSpectrumNbins) )
        
    def processEditIMSpectrumNbins(self):
        print 'EditIMSpectrumNbins'
        cp.confpars.imageSpectrumNbins = int(self.editIMSpectrumNbins.displayText())        

    def processSliderIMAmin(self):
        #print 'SliderIMAmin',
        #print self.sliderIMAmin.value()
        value = self.sliderIMAmin.value()
        if value > cp.confpars.imageSpectrumAmax :
            self.sliderIMAmax.setValue(value)
        cp.confpars.imageImageAmax     = value
        cp.confpars.imageSpectrumAmin  = value
        #cp.confpars.imageSpectrumNbins = cp.confpars.imageSpectrumAmax - value
        self.editIMImageAmin    .setText( str(value) )
        self.editIMSpectrumAmin .setText( str(value) )
        #self.editIMSpectrumNbins.setText( str(cp.confpars.imageSpectrumNbins) )

    def processSliderIMAmax(self):
        #print 'SliderIMAmax',
        #print self.sliderIMAmax.value()
        value = self.sliderIMAmax.value()
        if value < cp.confpars.imageSpectrumAmin :
            self.sliderIMAmin.setValue(value)
        cp.confpars.imageSpectrumAmax  = value
        cp.confpars.imageImageAmax     = value
        #cp.confpars.imageSpectrumNbins = value - cp.confpars.imageSpectrumAmin
        self.editIMImageAmax    .setText( str(value) )
        self.editIMSpectrumAmax .setText( str(value) )
        #self.editIMSpectrumNbins.setText( str(cp.confpars.imageSpectrumNbins) )


    def processCBoxIMImage(self, value):
        if self.cboxIMImage.isChecked():
            cp.confpars.imageImageIsOn = True
            #self.parentWidget.cboxIMImage   .setCheckState(2)
        else:
            cp.confpars.imageImageIsOn = False
            #self.parentWidget.cboxIMImage   .setCheckState(0)


    def processCBoxIMSpectrum(self, value):
        if self.cboxIMSpectrum.isChecked():
            cp.confpars.imageSpectrumIsOn = True
            #self.parentWidget.cboxIMSpectrum.setCheckState(2)
        else:
            cp.confpars.imageSpectrumIsOn = False
            #self.parentWidget.cboxIMSpectrum.setCheckState(0)


#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIWhatToDisplayForImage()
    ex.show()
    app.exec_()
#-----------------------------

