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
        self.setWindowTitle('Adjust Image Parameters')

        titFont = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        self.titImage    = QtGui.QLabel('Image')
        #self.titCSpad    = QtGui.QLabel('SCpad')
        #self.titWaveform = QtGui.QLabel('Waveform')

        self.titIMImageAmin      = QtGui.QLabel('Amin:')
        self.titIMImageAmax      = QtGui.QLabel('Amax:')

        self.titIMSpectrumAmin   = QtGui.QLabel('Amin:')
        self.titIMSpectrumAmax   = QtGui.QLabel('Amax:')
        self.titIMSpectrumNbins  = QtGui.QLabel('N bins:')
        self.titIMSpectrumAlim   = QtGui.QLabel('Slider range: 0 -')

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
        self.editIMAmpRange      = QtGui.QLineEdit(str(cp.confpars.imageAmplitudeRange))
        self.editIMAmpRange      .setMaximumWidth(50)

        self.titImage   .setFont (titFont) 
        #self.titCSpad   .setFont (titFont) 
        #self.titWaveform.setFont (titFont) 

        self.cboxIMImage    = QtGui.QCheckBox('Image',    self)
        self.cboxIMSpectrum = QtGui.QCheckBox('Spectrum', self)

        #self.cboxCSImage    = QtGui.QCheckBox('Image',    self)
        #self.cboxCSSpectrum = QtGui.QCheckBox('Spectrum', self)

        #self.cboxWFImage    = QtGui.QCheckBox('Image',    self)
        #self.cboxWFSpectrum = QtGui.QCheckBox('Spectrum', self)

        if cp.confpars.imageImageIsOn       : self.cboxIMImage   .setCheckState(2)
        if cp.confpars.imageSpectrumIsOn    : self.cboxIMSpectrum.setCheckState(2)
        #if cp.confpars.cspadImageIsOn       : self.cboxCSImage   .setCheckState(2)
        #if cp.confpars.cspadSpectrumIsOn    : self.cboxCSSpectrum.setCheckState(2)
        #if cp.confpars.waveformImageIsOn    : self.cboxWFImage   .setCheckState(1)
        #if cp.confpars.waveformSpectrumIsOn : self.cboxWFSpectrum.setCheckState(1)

        #self.cb.setFocusPolicy(QtCore.Qt.NoFocus) 
        #self.cb2.move(10, 50)
        #self.cb.move(10, 10)
        #self.cb.toggle() # set the check mark in the check box

        #lcd           = QtGui.QLCDNumber(self)
        self.sliderIMAmin  = QtGui.QSlider(QtCore.Qt.Horizontal, self)        
        self.sliderIMAmax  = QtGui.QSlider(QtCore.Qt.Horizontal, self)        
        self.setAmplitudeRange(cp.confpars.imageAmplitudeRange)
        self.sliderIMAmin.setValue(cp.confpars.imageSpectrumAmin)
        self.sliderIMAmax.setValue(cp.confpars.imageSpectrumAmax)
        
        self.butClose = QtGui.QPushButton("Close window")

        hboxIM01 = QtGui.QHBoxLayout()
        hboxIM01.addWidget(self.titImage)        
        hboxIM02 = QtGui.QHBoxLayout()
        #hboxIM02.addStretch(1)
        hboxIM02.addWidget(self.titIMSpectrumAlim) 
        hboxIM02.addWidget(self.editIMAmpRange) 
        hboxIM02.addWidget(self.sliderIMAmin)        
        hboxIM02.addWidget(self.sliderIMAmax)        

        gridIM = QtGui.QGridLayout()
        gridIM.addWidget(self.cboxIMImage,     0, 0)
        gridIM.addWidget(self.titIMImageAmin,  0, 1)
        gridIM.addWidget(self.editIMImageAmin, 0, 2)
        gridIM.addWidget(self.titIMImageAmax,  0, 3)
        gridIM.addWidget(self.editIMImageAmax, 0, 4)
        
        gridIM.addWidget(self.cboxIMSpectrum,      1, 0)
        gridIM.addWidget(self.titIMSpectrumAmin,   1, 1)
        gridIM.addWidget(self.editIMSpectrumAmin,  1, 2)
        gridIM.addWidget(self.titIMSpectrumAmax,   1, 3)
        gridIM.addWidget(self.editIMSpectrumAmax,  1, 4)
        gridIM.addWidget(self.titIMSpectrumNbins,  1, 5)
        gridIM.addWidget(self.editIMSpectrumNbins, 1, 6)
        
        #hboxWF01 = QtGui.QHBoxLayout()
        #hboxWF01.addWidget(self.titWaveform)        
        #hboxWF02 = QtGui.QHBoxLayout()
        #hboxWF02.addWidget(self.cboxWFImage)
        #hboxWF03 = QtGui.QHBoxLayout()
        #hboxWF03.addWidget(self.cboxWFSpectrum)

        #hboxCS01 = QtGui.QHBoxLayout()
        #hboxCS01.addWidget(self.titCSpad)        
        #hboxCS02 = QtGui.QHBoxLayout()
        #hboxCS02.addWidget(self.cboxCSImage)
        #hboxCS03 = QtGui.QHBoxLayout()
        #hboxCS03.addWidget(self.cboxCSSpectrum)

        hboxC = QtGui.QHBoxLayout()
        hboxC.addStretch(1)
        hboxC.addWidget(self.butClose)
        
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(hboxIM01)
        self.vbox.addLayout(gridIM) 
        self.vbox.addLayout(hboxIM02)
        #self.vbox.addStretch(1)     
        #self.vbox.addLayout(hboxWF01)
        #self.vbox.addLayout(hboxWF02)
        #self.vbox.addLayout(hboxWF03)
        #self.vbox.addStretch(1)     
        #self.vbox.addLayout(hboxCS01)
        #self.vbox.addLayout(hboxCS02)
        #self.vbox.addLayout(hboxCS03)
        self.vbox.addStretch(1)     

        if parent == None :
            self.vbox.addLayout(hboxC)
            self.setLayout(self.vbox)
            self.show()

        self.connect(self.butClose,            QtCore.SIGNAL('clicked()'),         self.processClose )

        self.connect(self.cboxIMImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxIMImage)
        self.connect(self.cboxIMSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxIMSpectrum)
        #self.connect(self.cboxCSImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSImage)
        #self.connect(self.cboxCSSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSSpectrum)
        #self.connect(self.cboxWFImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxWFImage)
        #self.connect(self.cboxWFSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxWFSpectrum)

        self.connect(self.sliderIMAmin,        QtCore.SIGNAL('valueChanged(int)'), self.processSliderIMAmin )
        self.connect(self.sliderIMAmax,        QtCore.SIGNAL('valueChanged(int)'), self.processSliderIMAmax )

        self.connect(self.editIMImageAmin,     QtCore.SIGNAL('textEdited(const QString&)'), self.processEditIMImageAmin )
        self.connect(self.editIMImageAmax,     QtCore.SIGNAL('textEdited(const QString&)'), self.processEditIMImageAmax )
        self.connect(self.editIMSpectrumAmin,  QtCore.SIGNAL('textEdited(const QString&)'), self.processEditIMSpectrumAmin )
        self.connect(self.editIMSpectrumAmax,  QtCore.SIGNAL('textEdited(const QString&)'), self.processEditIMSpectrumAmax )
        self.connect(self.editIMSpectrumNbins, QtCore.SIGNAL('textEdited(const QString&)'), self.processEditIMSpectrumNbins )
        self.connect(self.editIMAmpRange,      QtCore.SIGNAL('textEdited(const QString&)'), self.processEditIMAmplitudeRange )

        cp.confpars.wtdIMWindowIsOpen = True


    def getVBoxForLayout(self):
        return self.vbox

    def setParentWidget(self,parent):
        self.parentWidget = parent

    def setAmplitudeRange(self, amplitudeRange):
        self.sliderIMAmin.setRange(0,amplitudeRange)
        self.sliderIMAmax.setRange(0,amplitudeRange)
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
        self.setAmplitudeRange(cp.confpars.imageAmplitudeRange)

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
            self.parentWidget.cboxIMImage   .setCheckState(2)
        else:
            cp.confpars.imageImageIsOn = False
            self.parentWidget.cboxIMImage   .setCheckState(0)


    def processCBoxIMSpectrum(self, value):
        if self.cboxIMSpectrum.isChecked():
            cp.confpars.imageSpectrumIsOn = True
            self.parentWidget.cboxIMSpectrum.setCheckState(2)
        else:
            cp.confpars.imageSpectrumIsOn = False
            self.parentWidget.cboxIMSpectrum.setCheckState(0)


    #def processCBoxCSImage(self, value):
    #    if self.cboxCSImage.isChecked():
    #        cp.confpars.cspadImageIsOn = True
    #    else:
    #        cp.confpars.cspadImageIsOn = False


    #def processCBoxCSSpectrum(self, value):
    #    if self.cboxCSSpectrum.isChecked():
    #        cp.confpars.cspadSpectrumIsOn = True
    #    else:
    #        cp.confpars.cspadSpectrumIsOn = False


    #def processCBoxWFImage(self, value):
    #    if self.cboxWFImage.isChecked():
    #        cp.confpars.waveformImageIsOn = True
    #    else:
    #        cp.confpars.waveformImageIsOn = False


    #def processCBoxWFSpectrum(self, value):
    #    if self.cboxWFSpectrum.isChecked():
    #        cp.confpars.waveformSpectrumIsOn = True
    #    else:
    #        cp.confpars.waveformSpectrumIsOn = False


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
    ex  = GUIWhatToDisplayForImage()
    ex.show()
    app.exec_()
#-----------------------------

