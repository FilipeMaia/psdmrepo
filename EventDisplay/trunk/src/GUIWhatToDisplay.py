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
import GUIWhatToDisplayForImage    as wtdIM 
import GUIWhatToDisplayForCSpad    as wtdCS 
import GUIWhatToDisplayForWaveform as wtdWF

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

        self.setGeometry(370, 350, 500, 500)
        self.setWindowTitle('What to display?')

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        
        titFont   = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)

        self.titImage    = QtGui.QLabel('Image')
        self.titCSpad    = QtGui.QLabel('SCpad')
        self.titWaveform = QtGui.QLabel('Waveform')

        self.titImage   .setFont (titFont) 
        self.titCSpad   .setFont (titFont) 
        self.titWaveform.setFont (titFont) 
        
        self.titCSImage    = QtGui.QLabel('Images:')
        self.titCSSpectra  = QtGui.QLabel('Spectra:')
        self.titCSImageSpec= QtGui.QLabel('Image & Spectrum:')

        self.cboxIMImage       = QtGui.QCheckBox('Image',            self)
        self.cboxIMSpectrum    = QtGui.QCheckBox('Spectrum',         self)
        self.cboxIMImageSpec   = QtGui.QCheckBox('Image and Spectrum', self)
        
        self.cboxCSImageQuad   = QtGui.QCheckBox('Quad',          self)
        self.cboxCSImageDet    = QtGui.QCheckBox('Detector',      self)
        self.cboxCSImage       = QtGui.QCheckBox('8 of 2x1',      self)
        self.cboxCSImageOfPair = QtGui.QCheckBox('1 of 2x1',      self)
        self.cboxCSSpectrum    = QtGui.QCheckBox('16 ASICs',      self)
        self.cboxCSSpectrum08  = QtGui.QCheckBox('8 of 2x1',      self)

        self.cboxWFWaveform    = QtGui.QCheckBox('Waveform',      self)

        if cp.confpars.imageImageIsOn       : self.cboxIMImage       .setCheckState(2)
        if cp.confpars.imageSpectrumIsOn    : self.cboxIMSpectrum    .setCheckState(2)
        if cp.confpars.imageImageSpecIsOn   : self.cboxIMImageSpec   .setCheckState(2)

        if cp.confpars.cspadImageOfPairIsOn : self.cboxCSImageOfPair .setCheckState(2)
        if cp.confpars.cspadImageIsOn       : self.cboxCSImage       .setCheckState(2)
        if cp.confpars.cspadImageQuadIsOn   : self.cboxCSImageQuad   .setCheckState(2)
        if cp.confpars.cspadImageDetIsOn    : self.cboxCSImageDet    .setCheckState(2)
        if cp.confpars.cspadSpectrumIsOn    : self.cboxCSSpectrum    .setCheckState(2)
        if cp.confpars.cspadSpectrum08IsOn  : self.cboxCSSpectrum08  .setCheckState(2)
        if cp.confpars.waveformWaveformIsOn : self.cboxWFWaveform    .setCheckState(2)

        self.butClose      = QtGui.QPushButton('Quit')

        self.showToolTips()

        gridCS = QtGui.QGridLayout()
        gridCS.addWidget(self. titCSpad,        0, 0)

        gridCS.addWidget(self. titCSImage,      1, 0)
        gridCS.addWidget(self.cboxCSImage,      1, 1)
        gridCS.addWidget(self.cboxCSImageQuad,  1, 2)
        gridCS.addWidget(self.cboxCSImageDet,   1, 3)

        gridCS.addWidget(self. titCSSpectra,    2, 0)
        gridCS.addWidget(self.cboxCSSpectrum08, 2, 1)
        gridCS.addWidget(self.cboxCSSpectrum,   2, 2)

        gridCS.addWidget(self. titCSImageSpec,  3, 0)
        gridCS.addWidget(self.cboxCSImageOfPair,3, 1)

        
        gridIM = QtGui.QGridLayout()
        gridIM.addWidget(self.titImage,         0, 0)
        gridIM.addWidget(self.cboxIMImage,      1, 0)
        gridIM.addWidget(self.cboxIMSpectrum,   1, 1)
        gridIM.addWidget(self.cboxIMImageSpec,  1, 2, 1, 2)
        
        gridWF = QtGui.QGridLayout()
        gridWF.addWidget(self.titWaveform,      0, 0)
        gridWF.addWidget(self.cboxWFWaveform,   1, 0)

        self.tabBar   = QtGui.QTabBar()
        self.indTabCS = self.tabBar.addTab('CSpad')
        self.indTabIM = self.tabBar.addTab('Image')
        self.indTabWF = self.tabBar.addTab('Waveform')

        self.tabBar.setTabTextColor(self.indTabCS,QtGui.QColor('red'))
        self.tabBar.setTabTextColor(self.indTabIM,QtGui.QColor('blue'))
        self.tabBar.setTabTextColor(self.indTabWF,QtGui.QColor('green'))
        
        self.hboxT = QtGui.QHBoxLayout()
        self.hboxT.addWidget(self.tabBar) 

        self.guiTab = wtdCS.GUIWhatToDisplayForCSpad()          
        self.guiTab.setMinimumHeight(200)

        self.hboxD = QtGui.QHBoxLayout()
        self.hboxD.addWidget(self.guiTab)
        
        self.hboxC = QtGui.QHBoxLayout()
        self.hboxC.addStretch(1)
        self.hboxC.addWidget(self.butClose)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(gridCS) 
        self.vbox.addStretch(1)     
        self.vbox.addLayout(gridIM) 
        self.vbox.addStretch(1)     
        self.vbox.addLayout(gridWF)
        #self.vbox.addLayout(self.hboxC)
        self.vbox.addStretch(1)     
        self.vbox.addLayout(self.hboxT)
        self.vbox.addLayout(self.hboxD)
        self.vbox.addStretch(1)
        self.vbox.addLayout(self.hboxC)

        self.setLayout(self.vbox)

        self.connect(self.butClose,            QtCore.SIGNAL('clicked()'),         self.processClose )
        self.connect(self.cboxIMImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxIMImage)
        self.connect(self.cboxIMImageSpec,     QtCore.SIGNAL('stateChanged(int)'), self.processCBoxIMImageSpec)
        self.connect(self.cboxIMSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxIMSpectrum)
        self.connect(self.cboxCSImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSImage)
        self.connect(self.cboxCSImageQuad,     QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSImageQuad)
        self.connect(self.cboxCSImageDet,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSImageDet)
        self.connect(self.cboxCSImageOfPair,   QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSImageOfPair)
        self.connect(self.cboxCSSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSSpectrum)
        self.connect(self.cboxCSSpectrum08,    QtCore.SIGNAL('stateChanged(int)'), self.processCBoxCSSpectrum08)
        self.connect(self.cboxWFWaveform,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxWFWaveform)
        self.connect(self.tabBar,              QtCore.SIGNAL('currentChanged(int)'),self.processTabBar)


    def showToolTips(self):
        self.butClose    .setToolTip('Close this window') 
        #self.butIMOptions.setToolTip('Adjust amplitude and other plot\nparameters for Image type plots.')
        #self.butCSOptions.setToolTip('Adjust amplitude and other plot\nparameters for CSpad type plots.')
        #self.butWFOptions.setToolTip('Adjust amplitude and other plot\nparameters for Waveform type plots.')


    def closeEvent(self, event):
        print 'closeEvent'
        self.processClose()


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def processClose(self):
        print 'Close window'
        cp.confpars.wtdWindowIsOpen = False
        self.close()


    def processTabBar(self):
        indTab = self.tabBar.currentIndex()
        print 'TabBar index=',indTab

        minSize = self.hboxD.minimumSize()
        self.guiTab.close()

        if indTab == self.indTabCS :
            self.guiTab = wtdCS.GUIWhatToDisplayForCSpad()
        if indTab == self.indTabIM :
            self.guiTab = wtdIM.GUIWhatToDisplayForImage()
        if indTab == self.indTabWF :
            self.guiTab = wtdWF.GUIWhatToDisplayForWaveform()

        self.guiTab.setMinimumHeight(200)
        self.hboxD.addWidget(self.guiTab)


    def processCBoxIMImage(self, value):
        if self.cboxIMImage.isChecked():
            cp.confpars.imageImageIsOn = True
            #if cp.confpars.wtdIMWindowIsOpen : self.guiwtdIM.cboxIMImage.setCheckState(2)
        else:
            cp.confpars.imageImageIsOn = False
            #if cp.confpars.wtdIMWindowIsOpen : self.guiwtdIM.cboxIMImage.setCheckState(0)


    def processCBoxIMSpectrum(self, value):
        if self.cboxIMSpectrum.isChecked():
            cp.confpars.imageSpectrumIsOn = True
            #if cp.confpars.wtdIMWindowIsOpen : self.guiwtdIM.cboxIMSpectrum.setCheckState(2)
        else:
            cp.confpars.imageSpectrumIsOn = False
            #if cp.confpars.wtdIMWindowIsOpen : self.guiwtdIM.cboxIMSpectrum.setCheckState(0)


    def processCBoxIMImageSpec(self, value):
        if self.cboxIMImageSpec.isChecked():
            cp.confpars.imageImageSpecIsOn = True
        else:
            cp.confpars.imageImageSpecIsOn = False


    def processCBoxCSImage(self, value):
        if self.cboxCSImage.isChecked():
            cp.confpars.cspadImageIsOn = True
            #if cp.confpars.wtdCSWindowIsOpen : self.guiwtdCS.cboxCSImage.setCheckState(2)
        else:
            cp.confpars.cspadImageIsOn = False
            #if cp.confpars.wtdCSWindowIsOpen : self.guiwtdCS.cboxCSImage.setCheckState(0)


    def processCBoxCSImageOfPair(self, value):
        print 'processCBoxCSImageOfPair'
        if self.cboxCSImageOfPair.isChecked():
            cp.confpars.cspadImageOfPairIsOn = True
        else:
            cp.confpars.cspadImageOfPairIsOn = False


    def processCBoxCSImageQuad(self, value):
        if self.cboxCSImageQuad.isChecked():
            cp.confpars.cspadImageQuadIsOn = True
        else:
            cp.confpars.cspadImageQuadIsOn = False


    def processCBoxCSImageDet(self, value):
        if self.cboxCSImageDet.isChecked():
            cp.confpars.cspadImageDetIsOn = True
        else:
            cp.confpars.cspadImageDetIsOn = False


    def processCBoxCSSpectrum(self, value):
        if self.cboxCSSpectrum.isChecked():
            cp.confpars.cspadSpectrumIsOn = True
            #if cp.confpars.wtdCSWindowIsOpen : self.guiwtdCS.cboxCSSpectrum.setCheckState(2)
        else:
            cp.confpars.cspadSpectrumIsOn = False
            #if cp.confpars.wtdCSWindowIsOpen : self.guiwtdCS.cboxCSSpectrum.setCheckState(0)


    def processCBoxCSSpectrum08(self, value):
        if self.cboxCSSpectrum08.isChecked():
            cp.confpars.cspadSpectrum08IsOn = True
        else:
            cp.confpars.cspadSpectrum08IsOn = False


    def processCBoxWFWaveform(self, value):
        if self.cboxWFWaveform.isChecked():
            cp.confpars.waveformWaveformIsOn = True
        else:
            cp.confpars.waveformWaveformIsOn = False


#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIWhatToDisplay()
    ex.show()
    app.exec_()
#-----------------------------

