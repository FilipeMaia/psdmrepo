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
import ConfigParameters               as cp
import GUIWhatToDisplayForImage       as wtdIM 
import GUIWhatToDisplayForCSpad       as wtdCS 
import GUIWhatToDisplayForWaveform    as wtdWF
import GUIWhatToDisplayForProjections as wtdPR
import GUICorrelation                 as wtdCO

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

        self.setGeometry(370, 350, 500, 600)
        self.setWindowTitle('What to display?')

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        
        titFont   = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)

        self.titImage    = QtGui.QLabel('Image')
        self.titCSpad    = QtGui.QLabel('CSpad')
        self.titWaveform = QtGui.QLabel('Other')

        self.titImage   .setFont (titFont) 
        self.titCSpad   .setFont (titFont) 
        self.titWaveform.setFont (titFont) 
        
        self.titCSImage       = QtGui.QLabel('Images:')
        self.titCSSpectra     = QtGui.QLabel('Spectra:')
        self.titCSImageSpec   = QtGui.QLabel('Image & Spectrum:')
        self.titCSProjections = QtGui.QLabel('Projections:')

        self.cboxIMImage       = QtGui.QCheckBox('Image',              self)
        self.cboxIMSpectrum    = QtGui.QCheckBox('Spectrum',           self)
        self.cboxIMImageSpec   = QtGui.QCheckBox('Image and Spectrum', self)
        self.cboxIMProjX       = QtGui.QCheckBox('X',                  self)
        self.cboxIMProjY       = QtGui.QCheckBox('Y',                  self)
        self.cboxIMProjR       = QtGui.QCheckBox('R',                  self)
        self.cboxIMProjPhi     = QtGui.QCheckBox(u'\u03C6',            self) # Phi in Greek
        
        self.cboxCSImageQuad   = QtGui.QCheckBox('Quad',               self)
        self.cboxCSImageDet    = QtGui.QCheckBox('Detector',           self)
        self.cboxCSImage       = QtGui.QCheckBox('8 of 2x1',           self)
        self.cboxCSImageOfPair = QtGui.QCheckBox('1 of 2x1',           self)
        self.cboxCSSpectrum    = QtGui.QCheckBox('16 ASICs',           self)
        self.cboxCSSpectrum08  = QtGui.QCheckBox('8 of 2x1',           self)
        self.cboxCSProjX       = QtGui.QCheckBox('X',                  self)
        self.cboxCSProjY       = QtGui.QCheckBox('Y',                  self)
        self.cboxCSProjR       = QtGui.QCheckBox('R',                  self)
        self.cboxCSProjPhi     = QtGui.QCheckBox(u'\u03C6',            self) # Phi in Greek

        self.cboxWFWaveform    = QtGui.QCheckBox('Waveform',           self)
        self.cboxCO            = QtGui.QCheckBox('Correlations',       self)
       #self.cboxED            = QtGui.QCheckBox('Par vs event No.',   self)

        if cp.confpars.imageImageIsOn       : self.cboxIMImage       .setCheckState(2)
        if cp.confpars.imageSpectrumIsOn    : self.cboxIMSpectrum    .setCheckState(2)
        if cp.confpars.imageImageSpecIsOn   : self.cboxIMImageSpec   .setCheckState(2)
        if cp.confpars.imageProjXIsOn       : self.cboxIMProjX       .setCheckState(2)
        if cp.confpars.imageProjYIsOn       : self.cboxIMProjY       .setCheckState(2)
        if cp.confpars.imageProjRIsOn       : self.cboxIMProjR       .setCheckState(2)
        if cp.confpars.imageProjPhiIsOn     : self.cboxIMProjPhi     .setCheckState(2)

        if cp.confpars.cspadImageOfPairIsOn : self.cboxCSImageOfPair .setCheckState(2)
        if cp.confpars.cspadImageIsOn       : self.cboxCSImage       .setCheckState(2)
        if cp.confpars.cspadImageQuadIsOn   : self.cboxCSImageQuad   .setCheckState(2)
        if cp.confpars.cspadImageDetIsOn    : self.cboxCSImageDet    .setCheckState(2)
        if cp.confpars.cspadSpectrumIsOn    : self.cboxCSSpectrum    .setCheckState(2)
        if cp.confpars.cspadSpectrum08IsOn  : self.cboxCSSpectrum08  .setCheckState(2)
        if cp.confpars.cspadProjXIsOn       : self.cboxCSProjX       .setCheckState(2)
        if cp.confpars.cspadProjYIsOn       : self.cboxCSProjY       .setCheckState(2)
        if cp.confpars.cspadProjRIsOn       : self.cboxCSProjR       .setCheckState(2)
        if cp.confpars.cspadProjPhiIsOn     : self.cboxCSProjPhi     .setCheckState(2)

        if cp.confpars.waveformWaveformIsOn : self.cboxWFWaveform    .setCheckState(2)
        if cp.confpars.correlationsIsOn     : self.cboxCO            .setCheckState(2)
       #if cp.confpars.perEventDistIsOn     : self.cboxED            .setCheckState(2)

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

        gridCS.addWidget(self. titCSProjections,4, 0)
        gridCS.addWidget(self.cboxCSProjX,      4, 1)
        gridCS.addWidget(self.cboxCSProjY,      4, 2)
        gridCS.addWidget(self.cboxCSProjR,      4, 3)
        gridCS.addWidget(self.cboxCSProjPhi,    4, 4)
        
        gridIM = QtGui.QGridLayout()
        gridIM.addWidget(self.titImage,         0, 0)
        gridIM.addWidget(self.cboxIMImage,      1, 0)
        gridIM.addWidget(self.cboxIMSpectrum,   1, 1)
        gridIM.addWidget(self.cboxIMImageSpec,  1, 2, 1, 2)
        gridIM.addWidget(self.cboxIMProjX,      2, 0)
        gridIM.addWidget(self.cboxIMProjY,      2, 1)
        gridIM.addWidget(self.cboxIMProjR,      2, 2)
        gridIM.addWidget(self.cboxIMProjPhi,    2, 3)
        
        gridWF = QtGui.QGridLayout()
        gridWF.addWidget(self.titWaveform,      0, 0)
        gridWF.addWidget(self.cboxWFWaveform,   1, 0)
        gridWF.addWidget(self.cboxCO,           1, 1)
       #gridWF.addWidget(self.cboxED,           1, 2)

        self.tabBar   = QtGui.QTabBar()
        self.indTabCS = self.tabBar.addTab('CSpad')
        self.indTabIM = self.tabBar.addTab('Image')
        self.indTabWF = self.tabBar.addTab('Waveform')
        self.indTabPR = self.tabBar.addTab('Proj.')
        self.indTabCO = self.tabBar.addTab('Corr.')
       #self.indTabED = self.tabBar.addTab('1D vs ev.')

        self.tabBar.setTabTextColor(self.indTabCS,QtGui.QColor('red'))
        self.tabBar.setTabTextColor(self.indTabIM,QtGui.QColor('blue'))
        self.tabBar.setTabTextColor(self.indTabWF,QtGui.QColor('green'))
        self.tabBar.setTabTextColor(self.indTabPR,QtGui.QColor('magenta'))
        self.tabBar.setTabTextColor(self.indTabCO,QtGui.QColor('black'))
       #self.tabBar.setTabTextColor(self.indTabED,QtGui.QColor('white'))
        
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
  
        self.connect(self.butClose,            QtCore.SIGNAL('clicked()'),           self.processClose )
        self.connect(self.cboxIMImage,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMImage)
        self.connect(self.cboxIMImageSpec,     QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMImageSpec)
        self.connect(self.cboxIMSpectrum,      QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMSpectrum)
        self.connect(self.cboxIMProjX,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMProjX)
        self.connect(self.cboxIMProjY,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMProjY)
        self.connect(self.cboxIMProjR,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMProjR)
        self.connect(self.cboxIMProjPhi,       QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMProjPhi)

        self.connect(self.cboxCSImage,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxCSImage)
        self.connect(self.cboxCSImageQuad,     QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxCSImageQuad)
        self.connect(self.cboxCSImageDet,      QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxCSImageDet)
        self.connect(self.cboxCSImageOfPair,   QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxCSImageOfPair)
        self.connect(self.cboxCSSpectrum,      QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxCSSpectrum)
        self.connect(self.cboxCSSpectrum08,    QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxCSSpectrum08)
        self.connect(self.cboxCSProjX,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxCSProjX)
        self.connect(self.cboxCSProjY,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxCSProjY)
        self.connect(self.cboxCSProjR,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxCSProjR)
        self.connect(self.cboxCSProjPhi,       QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxCSProjPhi)

        self.connect(self.cboxWFWaveform,      QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxWFWaveform)
        self.connect(self.cboxCO,              QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxCO)
       #self.connect(self.cboxED,              QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxED)

        self.connect(self.tabBar,              QtCore.SIGNAL('currentChanged(int)'), self.processTabBar)


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
        if indTab == self.indTabPR :
            self.guiTab = wtdPR.GUIWhatToDisplayForProjections()
        if indTab == self.indTabCO :
            self.guiTab = wtdCO.GUICorrelation()
        #if indTab == self.indTabED :
        #    self.guiTab = QtGui.QLabel('Sorry, this GUI is not implemented yet.\n' + 
        #                               'Will select 1-D parameters for plot vs event number.')

        self.guiTab.setMinimumHeight(200)
        self.hboxD.addWidget(self.guiTab)
        

    #def processCBoxED(self, value):
    #    if self.cboxED.isChecked():
    #        self.tabBar.setCurrentIndex(self.indTabED)
    #        cp.confpars.perEventDistIsOn = True
    #    else:
    #        cp.confpars.perEventDistIsOn = False


    def processCBoxCO(self, value):
        if self.cboxCO.isChecked():
            self.tabBar.setCurrentIndex(self.indTabCO)
            cp.confpars.correlationsIsOn = True
        else:
            cp.confpars.correlationsIsOn = False


    def processCBoxIMProjX(self, value):
        if self.cboxIMProjX.isChecked():
            self.tabBar.setCurrentIndex(self.indTabPR)
            self.guiTab.tabBar.setCurrentIndex(0) # 0 stands for X
            cp.confpars.imageProjXIsOn = True
        else:
            cp.confpars.imageProjXIsOn = False


    def processCBoxIMProjY(self, value):
        if self.cboxIMProjY.isChecked():
            self.tabBar.setCurrentIndex(self.indTabPR)
            self.guiTab.tabBar.setCurrentIndex(1) # 1 stands for Y
            cp.confpars.imageProjYIsOn = True
        else:
            cp.confpars.imageProjYIsOn = False


    def processCBoxIMProjR(self, value):
        if self.cboxIMProjR.isChecked():
            self.tabBar.setCurrentIndex(self.indTabPR)
            self.guiTab.tabBar.setCurrentIndex(2) # 2 stands for R
            cp.confpars.imageProjRIsOn = True
        else:
            cp.confpars.imageProjRIsOn = False


    def processCBoxIMProjPhi(self, value):
        if self.cboxIMProjPhi.isChecked():
            self.tabBar.setCurrentIndex(self.indTabPR)
            self.guiTab.tabBar.setCurrentIndex(3) # 3 stands for Phi
            cp.confpars.imageProjPhiIsOn = True
        else:
            cp.confpars.imageProjPhiIsOn = False


    def processCBoxCSProjX(self, value):
        if self.cboxCSProjX.isChecked():
            self.tabBar.setCurrentIndex(self.indTabPR)
            self.guiTab.tabBar.setCurrentIndex(0) # 0 stands for X
            cp.confpars.cspadProjXIsOn = True
        else:
            cp.confpars.cspadProjXIsOn = False


    def processCBoxCSProjY(self, value):
        if self.cboxCSProjY.isChecked():
            self.tabBar.setCurrentIndex(self.indTabPR)
            self.guiTab.tabBar.setCurrentIndex(1) # 1 stands for Y
            cp.confpars.cspadProjYIsOn = True
        else:
            cp.confpars.cspadProjYIsOn = False


    def processCBoxCSProjR(self, value):
        if self.cboxCSProjR.isChecked():
            self.tabBar.setCurrentIndex(self.indTabPR)
            self.guiTab.tabBar.setCurrentIndex(2) # 2 stands for R
            cp.confpars.cspadProjRIsOn = True
        else:
            cp.confpars.cspadProjRIsOn = False


    def processCBoxCSProjPhi(self, value):
        if self.cboxCSProjPhi.isChecked():
            self.tabBar.setCurrentIndex(self.indTabPR)
            self.guiTab.tabBar.setCurrentIndex(3) # 3 stands for Phi
            cp.confpars.cspadProjPhiIsOn = True
        else:
            cp.confpars.cspadProjPhiIsOn = False


    def processCBoxIMImage(self, value):
        if self.cboxIMImage.isChecked():
            self.tabBar.setCurrentIndex(self.indTabIM)
            cp.confpars.imageImageIsOn = True
        else:
            cp.confpars.imageImageIsOn = False


    def processCBoxIMSpectrum(self, value):
        if self.cboxIMSpectrum.isChecked():
            self.tabBar.setCurrentIndex(self.indTabIM)
            cp.confpars.imageSpectrumIsOn = True
        else:
            cp.confpars.imageSpectrumIsOn = False


    def processCBoxIMImageSpec(self, value):
        if self.cboxIMImageSpec.isChecked():
            self.tabBar.setCurrentIndex(self.indTabIM)
            cp.confpars.imageImageSpecIsOn = True
        else:
            cp.confpars.imageImageSpecIsOn = False


    def processCBoxCSImage(self, value):
        if self.cboxCSImage.isChecked():
            self.tabBar.setCurrentIndex(self.indTabCS)
            cp.confpars.cspadImageIsOn = True
        else:
            cp.confpars.cspadImageIsOn = False


    def processCBoxCSImageOfPair(self, value):
        if self.cboxCSImageOfPair.isChecked():
            self.tabBar.setCurrentIndex(self.indTabCS)
            cp.confpars.cspadImageOfPairIsOn = True
        else:
            cp.confpars.cspadImageOfPairIsOn = False


    def processCBoxCSImageQuad(self, value):
        if self.cboxCSImageQuad.isChecked():
            self.tabBar.setCurrentIndex(self.indTabCS)
            cp.confpars.cspadImageQuadIsOn = True
        else:
            cp.confpars.cspadImageQuadIsOn = False


    def processCBoxCSImageDet(self, value):
        if self.cboxCSImageDet.isChecked():
            self.tabBar.setCurrentIndex(self.indTabCS)
            cp.confpars.cspadImageDetIsOn = True
        else:
            cp.confpars.cspadImageDetIsOn = False


    def processCBoxCSSpectrum(self, value):
        if self.cboxCSSpectrum.isChecked():
            self.tabBar.setCurrentIndex(self.indTabCS)
            cp.confpars.cspadSpectrumIsOn = True
        else:
            cp.confpars.cspadSpectrumIsOn = False


    def processCBoxCSSpectrum08(self, value):
        if self.cboxCSSpectrum08.isChecked():
            self.tabBar.setCurrentIndex(self.indTabCS)
            cp.confpars.cspadSpectrum08IsOn = True
        else:
            cp.confpars.cspadSpectrum08IsOn = False


    def processCBoxWFWaveform(self, value):
        if self.cboxWFWaveform.isChecked():
            self.tabBar.setCurrentIndex(self.indTabWF)
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

