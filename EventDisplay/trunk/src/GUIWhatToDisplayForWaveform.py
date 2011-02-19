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

        self.palette = QtGui.QPalette()

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

        titFont12 = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        titFont10 = QtGui.QFont("Sans Serif", 10, QtGui.QFont.Bold)

        self.titWaveform         = QtGui.QLabel('Waveform')

        self.titWaveform   .setFont (titFont12) 

        self.titWFWaveformAmin   = QtGui.QLabel('Amin:')
        self.titWFWaveformAmax   = QtGui.QLabel('Amax:')
        self.titWFWaveformTmin   = QtGui.QLabel('Tmin:')
        self.titWFWaveformTmax   = QtGui.QLabel('Tmax:')

        self.editWFWaveformAmin  = QtGui.QLineEdit(str(cp.confpars.waveformWaveformAmin))
        self.editWFWaveformAmax  = QtGui.QLineEdit(str(cp.confpars.waveformWaveformAmax))
        self.editWFWaveformAmin  .setMaximumWidth(45)
        self.editWFWaveformAmax  .setMaximumWidth(45)

        self.editWFWaveformTmin  = QtGui.QLineEdit(str(cp.confpars.waveformWaveformTmin))
        self.editWFWaveformTmax  = QtGui.QLineEdit(str(cp.confpars.waveformWaveformTmax))
        self.editWFWaveformTmin  .setMaximumWidth(45)
        self.editWFWaveformTmax  .setMaximumWidth(45)

        self.setEditFieldsReadOnly(cp.confpars.waveformAutoRangeIsOn)

        self.radioAuto   = QtGui.QRadioButton("Auto range of parameters")
        self.radioManual = QtGui.QRadioButton("Manual range control:")
        self.radioGroup  = QtGui.QButtonGroup()
        self.radioGroup.addButton(self.radioAuto)
        self.radioGroup.addButton(self.radioManual)
        if cp.confpars.waveformAutoRangeIsOn : self.radioAuto  .setChecked(True)
        else :                                 self.radioManual.setChecked(True)


        hboxWF01 = QtGui.QHBoxLayout()
        hboxWF01.addWidget(self.titWaveform)        

        gridWF = QtGui.QGridLayout()
        gridWF.addWidget(self.radioAuto,           0, 0)
        gridWF.addWidget(self.radioManual,         1, 0)
        gridWF.addWidget(self.titWFWaveformAmin,   1, 1)
        gridWF.addWidget(self.editWFWaveformAmin,  1, 2)
        gridWF.addWidget(self.titWFWaveformAmax,   1, 3)
        gridWF.addWidget(self.editWFWaveformAmax,  1, 4)
        gridWF.addWidget(self.titWFWaveformTmin,   2, 1)
        gridWF.addWidget(self.editWFWaveformTmin,  2, 2)
        gridWF.addWidget(self.titWFWaveformTmax,   2, 3)
        gridWF.addWidget(self.editWFWaveformTmax,  2, 4)
        
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
        self.connect(self.editWFWaveformTmin,  QtCore.SIGNAL('editingFinished ()'), self.processEditWFWaveformTmin )
        self.connect(self.editWFWaveformTmax,  QtCore.SIGNAL('editingFinished ()'), self.processEditWFWaveformTmax )
        self.connect(self.radioAuto,           QtCore.SIGNAL('clicked()'),          self.processRadioAuto    )
        self.connect(self.radioManual,         QtCore.SIGNAL('clicked()'),          self.processRadioManual  )
 
        cp.confpars.wtdWFWindowIsOpen = True

        self.showToolTips()

    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters for waveforms.')
        self.radioAuto  .setToolTip('Select between Auto and Manual range control.')
        self.radioManual.setToolTip('Select between Auto and Manual range control.')
        self.editWFWaveformAmin.setToolTip('This field can be edited for Manual control only.')
        self.editWFWaveformAmax.setToolTip('This field can be edited for Manual control only.')
        self.editWFWaveformTmin.setToolTip('This field can be edited for Manual control only.')
        self.editWFWaveformTmax.setToolTip('This field can be edited for Manual control only.')


    def setEditFieldsReadOnly(self, isReadOnly=False):

        if isReadOnly == True : self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('grey'))
        else :                  self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('white'))

        self.editWFWaveformAmin.setPalette(self.palette)
        self.editWFWaveformAmax.setPalette(self.palette)
        self.editWFWaveformTmin.setPalette(self.palette)
        self.editWFWaveformTmax.setPalette(self.palette)

        self.editWFWaveformAmin.setReadOnly(isReadOnly)
        self.editWFWaveformAmax.setReadOnly(isReadOnly)
        self.editWFWaveformTmin.setReadOnly(isReadOnly)
        self.editWFWaveformTmax.setReadOnly(isReadOnly)

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

    def processEditWFWaveformTmin(self):
        print 'EditWFWaveformTmin'
        cp.confpars.waveformWaveformTmin = int(self.editWFWaveformTmin.displayText())        

    def processEditWFWaveformTmax(self):
        print 'EditWFWaveformTmax'
        cp.confpars.waveformWaveformTmax = int(self.editWFWaveformTmax.displayText())        

    def processRadioAuto(self):
        #print 'RadioAuto'
        cp.confpars.waveformAutoRangeIsOn = True
        self.setEditFieldsReadOnly(cp.confpars.waveformAutoRangeIsOn)
                      
    def processRadioManual(self):
        #print 'RadioManual'
        cp.confpars.waveformAutoRangeIsOn = False
        self.setEditFieldsReadOnly(cp.confpars.waveformAutoRangeIsOn)

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

