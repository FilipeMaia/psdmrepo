#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplayCBoxImage...
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
class GUIWhatToDisplayCBoxImage ( QtGui.QWidget ) :
    """Provides GUI to select information for rendering."""

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(370, 350, 500, 80)
        self.setWindowTitle('What to display?')

        self.parent = parent

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        
        titFont   = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)

        self.titImage    = QtGui.QLabel('Image')
        self.titImage   .setFont (titFont) 
        
        self.cboxIMImage       = QtGui.QCheckBox('Image',              self)
        self.cboxIMSpectrum    = QtGui.QCheckBox('Spectrum',           self)
        self.cboxIMImageSpec   = QtGui.QCheckBox('Image and Spectrum', self)
        self.cboxIMProjX       = QtGui.QCheckBox('X',                  self)
        self.cboxIMProjY       = QtGui.QCheckBox('Y',                  self)
        self.cboxIMProjR       = QtGui.QCheckBox('R',                  self)
        self.cboxIMProjPhi     = QtGui.QCheckBox(u'\u03C6',            self) # Phi in Greek
        
        if cp.confpars.imageImageIsOn       : self.cboxIMImage       .setCheckState(2)
        if cp.confpars.imageSpectrumIsOn    : self.cboxIMSpectrum    .setCheckState(2)
        if cp.confpars.imageImageSpecIsOn   : self.cboxIMImageSpec   .setCheckState(2)
        if cp.confpars.imageProjXIsOn       : self.cboxIMProjX       .setCheckState(2)
        if cp.confpars.imageProjYIsOn       : self.cboxIMProjY       .setCheckState(2)
        if cp.confpars.imageProjRIsOn       : self.cboxIMProjR       .setCheckState(2)
        if cp.confpars.imageProjPhiIsOn     : self.cboxIMProjPhi     .setCheckState(2)

        self.showToolTips()

        gridIM = QtGui.QGridLayout()
        gridIM.addWidget(self.titImage,         0, 0)
        gridIM.addWidget(self.cboxIMImage,      1, 0)
        gridIM.addWidget(self.cboxIMSpectrum,   1, 1)
        gridIM.addWidget(self.cboxIMImageSpec,  1, 2, 1, 2)
        gridIM.addWidget(self.cboxIMProjX,      2, 0)
        gridIM.addWidget(self.cboxIMProjY,      2, 1)
        gridIM.addWidget(self.cboxIMProjR,      2, 2)
        gridIM.addWidget(self.cboxIMProjPhi,    2, 3)
        
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(gridIM) 
        self.vbox.addStretch(1)     

        self.setLayout(self.vbox)
  
        self.connect(self.cboxIMImage,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMImage)
        self.connect(self.cboxIMImageSpec,     QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMImageSpec)
        self.connect(self.cboxIMSpectrum,      QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMSpectrum)
        self.connect(self.cboxIMProjX,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMProjX)
        self.connect(self.cboxIMProjY,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMProjY)
        self.connect(self.cboxIMProjR,         QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMProjR)
        self.connect(self.cboxIMProjPhi,       QtCore.SIGNAL('stateChanged(int)'),   self.processCBoxIMProjPhi)


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


    def processCBoxIMProjX(self, value):
        if self.cboxIMProjX.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabPR,0) # 0 stands for X
            cp.confpars.imageProjXIsOn = True
        else:
            cp.confpars.imageProjXIsOn = False


    def processCBoxIMProjY(self, value):
        if self.cboxIMProjY.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabPR,1) # 1 stands for Y
            cp.confpars.imageProjYIsOn = True
        else:
            cp.confpars.imageProjYIsOn = False


    def processCBoxIMProjR(self, value):
        if self.cboxIMProjR.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabPR,2)  # 2 stands for R
            cp.confpars.imageProjRIsOn = True
        else:
            cp.confpars.imageProjRIsOn = False


    def processCBoxIMProjPhi(self, value):
        if self.cboxIMProjPhi.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabPR,3) # 3 stands for Phi
            cp.confpars.imageProjPhiIsOn = True
        else:
            cp.confpars.imageProjPhiIsOn = False


    def processCBoxIMImage(self, value):
        if self.cboxIMImage.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabIM)
            cp.confpars.imageImageIsOn = True
        else:
            cp.confpars.imageImageIsOn = False


    def processCBoxIMSpectrum(self, value):
        if self.cboxIMSpectrum.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabIM)
            cp.confpars.imageSpectrumIsOn = True
        else:
            cp.confpars.imageSpectrumIsOn = False


    def processCBoxIMImageSpec(self, value):
        if self.cboxIMImageSpec.isChecked():
            self.setActiveTabBarForIndex(self.parent.indTabIM)
            cp.confpars.imageImageSpecIsOn = True
        else:
            cp.confpars.imageImageSpecIsOn = False


#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIWhatToDisplayCBoxImage()
    ex.show()
    app.exec_()
#-----------------------------

