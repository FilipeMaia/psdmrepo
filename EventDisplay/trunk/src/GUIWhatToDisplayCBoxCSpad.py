#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplayCBoxCSpad...
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
class GUIWhatToDisplayCBoxCSpad ( QtGui.QWidget ) :
    """Provides GUI to select information for rendering."""

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
        self.setWindowTitle('What to display for CSpad?')

        self.parent = parent

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        
        titFont   = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)

        self.titCSpad    = QtGui.QLabel('CSpad')
        self.titCSpad   .setFont (titFont) 

        self.titCSImage       = QtGui.QLabel('Images:')
        self.titCSSpectra     = QtGui.QLabel('Spectra:')
        self.titCSImageSpec   = QtGui.QLabel('Image & Spectrum:')
        self.titCSProjections = QtGui.QLabel('Projections:')

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
        

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(gridCS) 
        self.setLayout(self.vbox)
  
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


    def showToolTips(self):
        #self.butClose    .setToolTip('Close this window') 
        #self.butIMOptions.setToolTip('Adjust amplitude and other plot\nparameters for Image type plots.')
        #self.butCSOptions.setToolTip('Adjust amplitude and other plot\nparameters for CSpad type plots.')
        #self.butWFOptions.setToolTip('Adjust amplitude and other plot\nparameters for Waveform type plots.')
        pass


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def closeEvent(self, event):
        #print 'closeEvent'
        pass


    def processClose(self):
        #print 'Close button'
        self.close()


    def setActiveTabBarForIndex(self,ind,subind=None):
        self.parent.tabBar.setCurrentIndex(ind)               # 0,1,2,3 stands for CSpad, Image, Waveform, Proj., Corr. 
        if subind != None :
            self.parent.guiTab.tabBar.setCurrentIndex(subind) # 0,1,2,3 stands for X,Y,R,Phi


    def processCBoxCSProjX(self, value):
        if self.cboxCSProjX.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabPR,0) # 0 stands for X
            cp.confpars.cspadProjXIsOn = True
        else:
            cp.confpars.cspadProjXIsOn = False


    def processCBoxCSProjY(self, value):
        if self.cboxCSProjY.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabPR,1) # 1 stands for Y
            cp.confpars.cspadProjYIsOn = True
        else:
            cp.confpars.cspadProjYIsOn = False


    def processCBoxCSProjR(self, value):
        if self.cboxCSProjR.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabPR,2) # 2 stands for R
            cp.confpars.cspadProjRIsOn = True
        else:
            cp.confpars.cspadProjRIsOn = False


    def processCBoxCSProjPhi(self, value):
        if self.cboxCSProjPhi.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabPR,3) # 3 stands for Phi
            cp.confpars.cspadProjPhiIsOn = True
        else:
            cp.confpars.cspadProjPhiIsOn = False


    def processCBoxCSImage(self, value):
        if self.cboxCSImage.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabCS)
            cp.confpars.cspadImageIsOn = True
        else:
            cp.confpars.cspadImageIsOn = False


    def processCBoxCSImageOfPair(self, value):
        if self.cboxCSImageOfPair.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabCS)
            cp.confpars.cspadImageOfPairIsOn = True
        else:
            cp.confpars.cspadImageOfPairIsOn = False


    def processCBoxCSImageQuad(self, value):
        if self.cboxCSImageQuad.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabCS)
            cp.confpars.cspadImageQuadIsOn = True
        else:
            cp.confpars.cspadImageQuadIsOn = False


    def processCBoxCSImageDet(self, value):
        if self.cboxCSImageDet.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabCS)
            cp.confpars.cspadImageDetIsOn = True
        else:
            cp.confpars.cspadImageDetIsOn = False


    def processCBoxCSSpectrum(self, value):
        if self.cboxCSSpectrum.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabCS)
            cp.confpars.cspadSpectrumIsOn = True
        else:
            cp.confpars.cspadSpectrumIsOn = False


    def processCBoxCSSpectrum08(self, value):
        if self.cboxCSSpectrum08.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabCS)
            cp.confpars.cspadSpectrum08IsOn = True
        else:
            cp.confpars.cspadSpectrum08IsOn = False


#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIWhatToDisplayCBoxCSpad()
    ex.show()
    app.exec_()
#-----------------------------

