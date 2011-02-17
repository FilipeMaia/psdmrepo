#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplayForWaveform...
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

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp

#---------------------
#  Class definition --
#---------------------
class GUIWhatToDisplayForWaveform ( QtGui.QWidget ) :
    """Provides GUI to select information for rendering."""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Adjust Waveform Parameters')

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(True)

        titFont12 = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        titFont10 = QtGui.QFont("Sans Serif", 10, QtGui.QFont.Bold)

        self.titWaveform         = QtGui.QLabel('Waveform')
        self.titWFWaveform       = QtGui.QLabel('Waveform plot')
        self.titWFSpectrum       = QtGui.QLabel('Spectrum')

        self.titWaveform   .setFont (titFont12) 
        self.titWFWaveform .setFont (titFont10)   
        self.titWFSpectrum .setFont (titFont10)

        self.titWFWaveformAmin   = QtGui.QLabel('Amin:')
        self.titWFWaveformAmax   = QtGui.QLabel('Amax:')

        self.titWFSpectrumAmin   = QtGui.QLabel('Amin:')
        self.titWFSpectrumAmax   = QtGui.QLabel('Amax:')
        self.titWFSpectrumNbins  = QtGui.QLabel('N bins:')
        self.titWFSpectrumAlim   = QtGui.QLabel('Slider range:')

        self.editWFWaveformAmin  = QtGui.QLineEdit(str(cp.confpars.waveformWaveformAmin))
        self.editWFWaveformAmax  = QtGui.QLineEdit(str(cp.confpars.waveformWaveformAmax))
        self.editWFWaveformAmin  .setMaximumWidth(45)
        self.editWFWaveformAmax  .setMaximumWidth(45)

        self.editWFSpectrumAmin  = QtGui.QLineEdit(str(cp.confpars.waveformSpectrumAmin))
        self.editWFSpectrumAmax  = QtGui.QLineEdit(str(cp.confpars.waveformSpectrumAmax))
        self.editWFSpectrumNbins = QtGui.QLineEdit(str(cp.confpars.waveformSpectrumNbins))
        self.editWFSpectrumAmin  .setMaximumWidth(45)
        self.editWFSpectrumAmax  .setMaximumWidth(45)
        self.editWFSpectrumNbins .setMaximumWidth(45)

        #self.butClose = QtGui.QPushButton("Close window")

        hboxWF01 = QtGui.QHBoxLayout()
        hboxWF01.addWidget(self.titWaveform)        

        gridWF = QtGui.QGridLayout()
        gridWF.addWidget(self.titWFWaveform,       0, 0)
        gridWF.addWidget(self.titWFWaveformAmin,   0, 1)
        gridWF.addWidget(self.editWFWaveformAmin,  0, 2)
        gridWF.addWidget(self.titWFWaveformAmax,   0, 3)
        gridWF.addWidget(self.editWFWaveformAmax,  0, 4)
        
        gridWF.addWidget(self.titWFSpectrum,       1, 0)
        gridWF.addWidget(self.titWFSpectrumAmin,   1, 1)
        gridWF.addWidget(self.editWFSpectrumAmin,  1, 2)
        gridWF.addWidget(self.titWFSpectrumAmax,   1, 3)
        gridWF.addWidget(self.editWFSpectrumAmax,  1, 4)
        gridWF.addWidget(self.titWFSpectrumNbins,  1, 5)
        gridWF.addWidget(self.editWFSpectrumNbins, 1, 6)
        
        #hboxC = QtGui.QHBoxLayout()
        #hboxC.addStretch(1)
        #hboxC.addWidget(self.butClose)
        
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(hboxWF01)
        self.vbox.addLayout(gridWF) 
        self.vbox.addStretch(1)     

        if parent == None :
            #self.vbox.addLayout(hboxC)
            self.setLayout(self.vbox)
            self.show()

        #self.connect(self.butClose,            QtCore.SIGNAL('clicked()'),         self.processClose )
        #self.connect(self.cboxWFImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxWFImage)
        #self.connect(self.cboxWFSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxWFSpectrum)

        self.connect(self.editWFWaveformAmin,  QtCore.SIGNAL('editingFinished ()'), self.processEditWFWaveformAmin )
        self.connect(self.editWFWaveformAmax,  QtCore.SIGNAL('editingFinished ()'), self.processEditWFWaveformAmax )
        self.connect(self.editWFSpectrumAmin,  QtCore.SIGNAL('editingFinished ()'), self.processEditWFSpectrumAmin )
        self.connect(self.editWFSpectrumAmax,  QtCore.SIGNAL('editingFinished ()'), self.processEditWFSpectrumAmax )
        self.connect(self.editWFSpectrumNbins, QtCore.SIGNAL('editingFinished ()'), self.processEditWFSpectrumNbins )

        cp.confpars.wtdWFWindowIsOpen = True

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

    def closeEvent(self, event):
        print 'closeEvent'
        self.processClose()

    def processClose(self):
        print 'Close window'
        cp.confpars.wtdWFWindowIsOpen = False
        self.close()

    def processEditWFWaveformAmin(self):
        print 'EditWFWaveformAmin'
        cp.confpars.waveformWaveformAmin = int(self.editWFWaveformAmin.displayText())        

    def processEditWFWaveformAmax(self):
        print 'EditWFWaveformAmax'
        cp.confpars.waveformWaveformAmax = int(self.editWFWaveformAmax.displayText())        

    def processEditWFSpectrumAmin(self):
        print 'EditWFSpectrumAmin'
        cp.confpars.waveformSpectrumAmin = int(self.editWFSpectrumAmin.displayText())        

    def processEditWFSpectrumAmax(self):
        print 'EditWFSpectrumAmax'
        cp.confpars.waveformSpectrumAmax  = int(self.editWFSpectrumAmax.displayText())        
        
    def processEditWFSpectrumNbins(self):
        print 'EditWFSpectrumNbins'
        cp.confpars.waveformSpectrumNbins = int(self.editWFSpectrumNbins.displayText())        


    #def processCBoxIMImage(self, value):
    #    if self.cboxIMImage.isChecked():
    #        cp.confpars.imageImageIsOn = True
    #        #self.parentWidget.cboxIMImage   .setCheckState(2)
    #    else:
    #        cp.confpars.imageImageIsOn = False
    #        #self.parentWidget.cboxIMImage   .setCheckState(0)

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIWhatToDisplayForWaveform()
    ex.show()
    app.exec_()
#-----------------------------

