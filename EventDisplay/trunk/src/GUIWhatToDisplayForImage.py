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
import PrintHDF5        as printh5
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

        self.palette_grey  = QtGui.QPalette()
        self.palette_white = QtGui.QPalette()
        self.palette_grey  .setColor(QtGui.QPalette.Base,QtGui.QColor('grey'))
        self.palette_white .setColor(QtGui.QPalette.Base,QtGui.QColor('white'))

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
        self.titIMSpectrumAlim   = QtGui.QLabel('Slider range:')

        self.radioBinWidth = QtGui.QRadioButton("Bin width:")
        self.radioNBins    = QtGui.QRadioButton("N bins:")
        self.radioGroupBin = QtGui.QButtonGroup()
        self.radioGroupBin.addButton(self.radioBinWidth)
        self.radioGroupBin.addButton(self.radioNBins)

        if cp.confpars.imageBinWidthIsOn  : self.radioBinWidth.setChecked(True)
        else :                              self.radioNBins.   setChecked(True)

        self.editIMImageAmin        = QtGui.QLineEdit(str(cp.confpars.imageImageAmin))
        self.editIMImageAmax        = QtGui.QLineEdit(str(cp.confpars.imageImageAmax))

        self.editIMImageAmin        .setMaximumWidth(45)
        self.editIMImageAmax        .setMaximumWidth(45)

        self.editIMSpectrumAmin     = QtGui.QLineEdit(str(cp.confpars.imageSpectrumAmin))
        self.editIMSpectrumAmax     = QtGui.QLineEdit(str(cp.confpars.imageSpectrumAmax))
        self.editIMSpectrumNBins    = QtGui.QLineEdit(str(cp.confpars.imageSpectrumNbins))
        self.editIMSpectrumBinWidth = QtGui.QLineEdit(str(cp.confpars.imageSpectrumBinWidth))

        self.editIMSpectrumAmin     .setMaximumWidth(45)
        self.editIMSpectrumAmax     .setMaximumWidth(45)
        self.editIMSpectrumNBins    .setMaximumWidth(45)
        self.editIMSpectrumBinWidth .setMaximumWidth(45)

        self.editIMAmpRaMin         = QtGui.QLineEdit(str(cp.confpars.imageAmplitudeRaMin))
        self.editIMAmpRange         = QtGui.QLineEdit(str(cp.confpars.imageAmplitudeRange))
        self.editIMAmpRaMin         .setMaximumWidth(50)
        self.editIMAmpRange         .setMaximumWidth(50)

        self.titIMDataset  = QtGui.QLabel('Dataset:')
        self.butSelDataSet = QtGui.QPushButton(cp.confpars.imageDataset)
        self.butSelDataSet.setMaximumWidth(350)
        self.setButSelDataSetTextAlignment()

        self.popupMenuForDataSet = QtGui.QMenu()
        self.fillPopupMenuForDataSet()

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
       #gridIM.addWidget(self.cboxIMImage,            0, 0)
        gridIM.addWidget(self.titIMDataset,           0, 0)
        gridIM.addWidget(self.butSelDataSet,          0, 1, 1, 6)
        gridIM.addWidget(self.titIMImage,             1, 0)
        gridIM.addWidget(self.titIMImageAmin,         1, 1)
        gridIM.addWidget(self.editIMImageAmin,        1, 2)
        gridIM.addWidget(self.titIMImageAmax,         1, 3)
        gridIM.addWidget(self.editIMImageAmax,        1, 4)
        
       #gridIM.addWidget(self.cboxIMSpectrum,         2, 0)
        gridIM.addWidget(self.titIMSpectrum,          2, 0)
        gridIM.addWidget(self.titIMSpectrumAmin,      2, 1)
        gridIM.addWidget(self.editIMSpectrumAmin,     2, 2)
        gridIM.addWidget(self.titIMSpectrumAmax,      2, 3)
        gridIM.addWidget(self.editIMSpectrumAmax,     2, 4)

        gridIM.addWidget(self.radioNBins,             2, 5)
        gridIM.addWidget(self.radioBinWidth,          3, 5)
        gridIM.addWidget(self.editIMSpectrumNBins,    2, 6)
        gridIM.addWidget(self.editIMSpectrumBinWidth, 3, 6)

        
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

        self.connect(self.butClose,               QtCore.SIGNAL('clicked()'),         self.processClose )

        #self.connect(self.cboxIMImage,            QtCore.SIGNAL('stateChanged(int)'), self.processCBoxIMImage)
        #self.connect(self.cboxIMSpectrum,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxIMSpectrum)

        self.connect(self.sliderIMAmin,           QtCore.SIGNAL('valueChanged(int)'), self.processSliderIMAmin )
        self.connect(self.sliderIMAmax,           QtCore.SIGNAL('valueChanged(int)'), self.processSliderIMAmax )

        self.connect(self.editIMImageAmin,        QtCore.SIGNAL('editingFinished ()'), self.processEditIMImageAmin )
        self.connect(self.editIMImageAmax,        QtCore.SIGNAL('editingFinished ()'), self.processEditIMImageAmax )
        self.connect(self.editIMSpectrumAmin,     QtCore.SIGNAL('editingFinished ()'), self.processEditIMSpectrumAmin )
        self.connect(self.editIMSpectrumAmax,     QtCore.SIGNAL('editingFinished ()'), self.processEditIMSpectrumAmax )
        self.connect(self.editIMSpectrumNBins,    QtCore.SIGNAL('editingFinished ()'), self.processEditIMSpectrumNBins )
        self.connect(self.editIMSpectrumBinWidth, QtCore.SIGNAL('editingFinished ()'), self.processEditIMSpectrumBinWidth )

        self.connect(self.editIMAmpRange,         QtCore.SIGNAL('editingFinished ()'), self.processEditIMAmplitudeRange )
        self.connect(self.editIMAmpRaMin,         QtCore.SIGNAL('editingFinished ()'), self.processEditIMAmplitudeRaMin )

        self.connect(self.radioBinWidth,          QtCore.SIGNAL('clicked()'),          self.processRadioBinWidth )
        self.connect(self.radioNBins,             QtCore.SIGNAL('clicked()'),          self.processRadioNBins    )
        self.connect(self.butSelDataSet,          QtCore.SIGNAL('clicked()'),          self.processMenuForDataSet )
  
        cp.confpars.wtdIMWindowIsOpen = True

        self.setBinning()
        self.setBinWidthReadOnly(not cp.confpars.imageBinWidthIsOn)



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
        cp.confpars.wtdIMWindowIsOpen = False
        self.close()

    def setButSelDataSetTextAlignment(self):
        if self.butSelDataSet.text() == 'None' or self.butSelDataSet.text() == 'All' :
            self.butSelDataSet.setStyleSheet('Text-align:center')
        else :
            self.butSelDataSet.setStyleSheet('Text-align:right')

    def fillPopupMenuForDataSet(self):
        print 'fillPopupMenuForDataSet'
        self.popupMenuForDataSet.addAction('None')
        for dsname in cp.confpars.list_of_checked_item_names :
            item_last_name   = printh5.get_item_last_name(dsname)           
            cspadIsInTheName = printh5.CSpadIsInTheName(dsname)
            if item_last_name == 'image' or cspadIsInTheName:

                self.popupMenuForDataSet.addAction(dsname)
        self.popupMenuForDataSet.addAction('All')

    def processMenuForDataSet(self):
        print 'MenuForDataSet'
        actionSelected = self.popupMenuForDataSet.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_ds = actionSelected.text()
        self.butSelDataSet.setText( selected_ds )
        self.setButSelDataSetTextAlignment()
        cp.confpars.imageDataset = str(selected_ds)






    def setBinWidthReadOnly(self, isReadOnly=False):
        if isReadOnly == True :
            self.editIMSpectrumNBins    .setPalette(self.palette_white)
            self.editIMSpectrumBinWidth .setPalette(self.palette_grey )
        else :
            self.editIMSpectrumNBins    .setPalette(self.palette_grey )
            self.editIMSpectrumBinWidth .setPalette(self.palette_white)

        self.editIMSpectrumNBins        .setReadOnly(not isReadOnly)
        self.editIMSpectrumBinWidth     .setReadOnly(isReadOnly)

    def setBinning(self) :
        if cp.confpars.imageBinWidthIsOn :
            cp.confpars.imageSpectrumNbins = (cp.confpars.imageSpectrumAmax - cp.confpars.imageSpectrumAmin) / cp.confpars.imageSpectrumBinWidth
            self.editIMSpectrumNBins.setText( str(cp.confpars.imageSpectrumNbins) )            
        else :
            cp.confpars.imageSpectrumBinWidth = (cp.confpars.imageSpectrumAmax - cp.confpars.imageSpectrumAmin) / cp.confpars.imageSpectrumNbins
            self.editIMSpectrumBinWidth.setText( str(cp.confpars.imageSpectrumBinWidth) )  

    def processRadioBinWidth(self):
        cp.confpars.imageBinWidthIsOn = True
        self.setBinWidthReadOnly(False)

    def processRadioNBins(self):
        cp.confpars.imageBinWidthIsOn = False
        self.setBinWidthReadOnly(True)

    def processEditIMSpectrumNBins   (self):
        cp.confpars.imageSpectrumNbins = int(self.editIMSpectrumNBins.displayText())
        self.setBinning()

    def processEditIMSpectrumBinWidth(self):
        cp.confpars.imageSpectrumBinWidth = int(self.editIMSpectrumBinWidth.displayText())
        self.setBinning()

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
        self.setBinning()

    def processEditIMSpectrumAmax(self):
        print 'EditIMSpectrumAmax'
        cp.confpars.imageSpectrumAmax  = int(self.editIMSpectrumAmax.displayText())        
        self.setBinning()

    def setAmplitudeRange(self,    amplitudeRaMin, amplitudeRange):
        self.sliderIMAmin.setRange(amplitudeRaMin, amplitudeRange)
        self.sliderIMAmax.setRange(amplitudeRaMin, amplitudeRange)
        #self.sliderIMAmin.setTickInterval(0.02*amplitudeRange)
        #self.sliderIMAmax.setTickInterval(0.02*amplitudeRange)
        
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

