#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplayForCSpadWindow...
#
#------------------------------------------------------------------------

"""Generates GUI to select information for rendaring in the HDF5Explorer.

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
import GlobalMethods        as gm

import GUISelectQuadAndPair as guiquadpair


#---------------------
#  Class definition --
#---------------------
class GUIWhatToDisplayForCSpadWindow ( QtGui.QWidget ) :
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

    def __init__(self, parent=None, window=0):
        QtGui.QWidget.__init__(self, parent)

        self.window = window
        cp.confpars.fillCSPadConfigParsNamedFromWin(self.window)

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Adjust CSpad Parameters')

        self.palette_white = QtGui.QPalette()
        self.palette_grey  = QtGui.QPalette()
        self.palette_red   = QtGui.QPalette()
        self.palette_white .setColor(QtGui.QPalette.Base,QtGui.QColor('white'))
        self.palette_grey  .setColor(QtGui.QPalette.Base,QtGui.QColor('grey'))
        self.palette_red   .setColor(QtGui.QPalette.Base,QtGui.QColor('red'))

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

        #titFont12 = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        titFont10 = QtGui.QFont("Sans Serif", 10, QtGui.QFont.Bold)

        self.titCSImage    = QtGui.QLabel('Image')
        self.titCSSpectrum = QtGui.QLabel('Spectrum')
        self.titCSImage    .setFont (titFont10)   
        self.titCSSpectrum .setFont (titFont10)

        self.cboxImALimits = QtGui.QCheckBox('A min/max:',self)
        self.cboxSpALimits = QtGui.QCheckBox('A min/max:',self)

        self.editCSImageAmin        = QtGui.QLineEdit(str(cp.confpars.cspadWindowParameters[self.window][1]))
        self.editCSImageAmax        = QtGui.QLineEdit(str(cp.confpars.cspadWindowParameters[self.window][2]))
        self.editCSSpectrumAmin     = QtGui.QLineEdit(str(cp.confpars.cspadWindowParameters[self.window][3]))
        self.editCSSpectrumAmax     = QtGui.QLineEdit(str(cp.confpars.cspadWindowParameters[self.window][4]))
        self.editCSSpectrumNBins    = QtGui.QLineEdit(str(cp.confpars.cspadWindowParameters[self.window][5]))
        self.editCSSpectrumBinWidth = QtGui.QLineEdit(str(cp.confpars.cspadWindowParameters[self.window][6]))

        self.editCSImageAmin        .setMaximumWidth(50)
        self.editCSImageAmax        .setMaximumWidth(50)
        self.editCSSpectrumAmin     .setMaximumWidth(50)
        self.editCSSpectrumAmax     .setMaximumWidth(50)
        self.editCSSpectrumNBins    .setMaximumWidth(50)
        self.editCSSpectrumBinWidth .setMaximumWidth(50)

        self.editCSImageAmin        .setValidator(QtGui.QIntValidator(-32000,32000,self))
        self.editCSImageAmax        .setValidator(QtGui.QIntValidator(-32000,32000,self))
        self.editCSSpectrumAmin     .setValidator(QtGui.QIntValidator(-32000,32000,self))
        self.editCSSpectrumAmax     .setValidator(QtGui.QIntValidator(-32000,32000,self))
        self.editCSSpectrumNBins    .setValidator(QtGui.QIntValidator(1,32000,self))
        self.editCSSpectrumBinWidth .setValidator(QtGui.QIntValidator(1,32000,self))
        
        self.radioNBins    = QtGui.QRadioButton("N bins:")
        self.radioBinWidth = QtGui.QRadioButton("Bin width:")
        self.radioGroup    = QtGui.QButtonGroup()
        self.radioGroup.addButton(self.radioNBins)
        self.radioGroup.addButton(self.radioBinWidth)

        if cp.confpars.cspadWindowParameters[self.window][9] : # cspadBinWidthIsOn :
            self.radioBinWidth.setChecked(True)
            self.processRadioBinWidth()
        else :
            self.radioNBins.setChecked(True)
            self.processRadioNBins()

        #self.char_expand = u'\u25BE' # down-head triangle
        #self.butMenuNWin = QtGui.QPushButton(str(cp.confpars.cspadNWindows) + self.char_expand)
        #self.butMenuNWin.setMaximumWidth(30)

        self.titCSDataset  = QtGui.QLabel('Dataset:')
        self.butSelDataSet = QtGui.QPushButton(cp.confpars.cspadWindowParameters[self.window][0])
        self.setButSelDataSetTextAlignment()

        self.popupMenuForDataSet = QtGui.QMenu()
        self.fillPopupMenuForDataSet()

        self.wquadpair = guiquadpair.GUISelectQuadAndPair(None,self.window)

        gridCS = QtGui.QGridLayout()

        gridCS.addWidget(self.titCSDataset,           0, 0)
        gridCS.addWidget(self.butSelDataSet,          0, 1, 1, 6)
 
        gridCS.addWidget(self.titCSImage,             1, 0)
        gridCS.addWidget(self.cboxImALimits,          1, 1)
        gridCS.addWidget(self.editCSImageAmin,        1, 2)
        gridCS.addWidget(self.editCSImageAmax,        1, 3)
        
        gridCS.addWidget(self.titCSSpectrum,          2, 0)
        gridCS.addWidget(self.cboxSpALimits,          2, 1)
        gridCS.addWidget(self.editCSSpectrumAmin,     2, 2)
        gridCS.addWidget(self.editCSSpectrumAmax,     2, 3)
        gridCS.addWidget(self.radioNBins,             2, 5)
        gridCS.addWidget(self.editCSSpectrumNBins,    2, 6)

        gridCS.addWidget(self.radioBinWidth,          3, 5)
        gridCS.addWidget(self.editCSSpectrumBinWidth, 3, 6)
        gridCS.addWidget(self.wquadpair,              3, 0, 1, 2)
    
        #hboxC = QtGui.QHBoxLayout()
        #hboxC.addStretch(1)
        #hboxC.addWidget(self.butClose)
        
        self.vbox = QtGui.QVBoxLayout()
        #self.vbox.addLayout(hboxCS01)
        self.vbox.addLayout(gridCS) 
        #self.vbox.addLayout(hboxCS02)

        if parent == None :
            self.setLayout(self.vbox)
            self.show()

        #self.connect(self.butClose,               QtCore.SIGNAL('clicked()'),          self.processClose )
        self.connect(self.editCSImageAmin,        QtCore.SIGNAL('editingFinished ()'), self.processEditCSImageAmin )
        self.connect(self.editCSImageAmax,        QtCore.SIGNAL('editingFinished ()'), self.processEditCSImageAmax )
        self.connect(self.editCSSpectrumAmin,     QtCore.SIGNAL('editingFinished ()'), self.processEditCSSpectrumAmin )
        self.connect(self.editCSSpectrumAmax,     QtCore.SIGNAL('editingFinished ()'), self.processEditCSSpectrumAmax )
        self.connect(self.editCSSpectrumNBins,    QtCore.SIGNAL('editingFinished ()'), self.processEditCSSpectrumNBins )
        self.connect(self.editCSSpectrumBinWidth, QtCore.SIGNAL('editingFinished ()'), self.processEditCSSpectrumBinWidth )
        self.connect(self.radioNBins,             QtCore.SIGNAL('clicked()'),          self.processRadioNBins    )
        self.connect(self.radioBinWidth,          QtCore.SIGNAL('clicked()'),          self.processRadioBinWidth )
        self.connect(self.cboxImALimits,          QtCore.SIGNAL('stateChanged(int)'),  self.processCboxImALimits)
        self.connect(self.cboxSpALimits,          QtCore.SIGNAL('stateChanged(int)'),  self.processCboxSpALimits)
        self.connect(self.butSelDataSet,          QtCore.SIGNAL('clicked()'),          self.processMenuForDataSet )
 
        self.setBinning()
        self.setCboxStatus()
        #cp.confpars.wtdCSWindowIsOpen = True


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
        #print 'closeEvent'
        #cp.confpars.wtdCSWindowIsOpen = False
        pass
        self.wquadpair.close()

    def processClose(self):
        #print 'Close button'
        self.close()


    def setCboxStatus(self):
        if cp.confpars.cspadWindowParameters[self.window][7] : self.cboxImALimits.setCheckState(2)
        if cp.confpars.cspadWindowParameters[self.window][8] : self.cboxSpALimits.setCheckState(2)
        self.setImEditFieldsStatus()
        self.setSpEditFieldsStatus()


    def processCboxImALimits(self):
        if self.cboxImALimits.isChecked():
            cp.confpars.cspadWindowParameters[self.window][7] = True
        else:
            cp.confpars.cspadWindowParameters[self.window][7] = False
        self.setImEditFieldsStatus()


    def processCboxSpALimits(self):
        if self.cboxSpALimits.isChecked():
            cp.confpars.cspadWindowParameters[self.window][8] = True
        else:
            cp.confpars.cspadWindowParameters[self.window][8] = False
        self.setSpEditFieldsStatus()


    def setImEditFieldsStatus(self):
        if self.cboxImALimits.isChecked(): self.palette = self.palette_white
        else :                             self.palette = self.palette_grey                            
        self.editCSImageAmin.setPalette(self.palette)
        self.editCSImageAmax.setPalette(self.palette)
        self.editCSImageAmin.setReadOnly(not self.cboxImALimits.isChecked())
        self.editCSImageAmax.setReadOnly(not self.cboxImALimits.isChecked())


    def setSpEditFieldsStatus(self):
        if self.cboxSpALimits.isChecked(): self.palette = self.palette_white
        else :                             self.palette = self.palette_grey                            
        self.editCSSpectrumAmin.setPalette(self.palette)
        self.editCSSpectrumAmax.setPalette(self.palette)
        self.editCSSpectrumAmin.setReadOnly(not self.cboxSpALimits.isChecked())
        self.editCSSpectrumAmax.setReadOnly(not self.cboxSpALimits.isChecked())


    def setButSelDataSetTextAlignment(self):
        if self.butSelDataSet.text() == 'None' or self.butSelDataSet.text() == 'All' :
            self.butSelDataSet.setStyleSheet('Text-align:center')
        else :
            self.butSelDataSet.setStyleSheet('Text-align:right')


    def fillPopupMenuForDataSet(self):
        #print 'fillPopupMenuForDataSet'
        self.popupMenuForDataSet.addAction('All')
        for dsname in cp.confpars.list_of_checked_item_names :
            #item_last_name   = gm.get_item_last_name(dsname)           
            cspadIsInTheName = gm.CSpadIsInTheName(dsname)
            #imageIsInTheName = gm.ImageIsInTheName(dsname)
            #if item_last_name == 'image' or cspadIsInTheName:
            if cspadIsInTheName :
                self.popupMenuForDataSet.addAction(dsname)
                print 'fillPopupMenuForDataSet: Add ds:', dsname
        #self.popupMenuForDataSet.addAction('All')


    def processMenuForDataSet(self):
        #print 'MenuForDataSet'
        actionSelected = self.popupMenuForDataSet.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_ds = actionSelected.text()
        self.butSelDataSet.setText( selected_ds )
        self.setButSelDataSetTextAlignment()
        cp.confpars.cspadWindowParameters[self.window][0] = str(selected_ds)


    def processEditCSImageAmin(self):
        #print 'EditCSImageAmin'
        cp.confpars.fillCSPadConfigParsNamedFromWin(self.window)
        cp.confpars.cspadImageAmin = int(self.editCSImageAmin.displayText())        
        if cp.confpars.cspadImageAmin >= cp.confpars.cspadImageAmax - cp.confpars.cspadSpectrumBinWidth:
            cp.confpars.cspadImageAmax = cp.confpars.cspadImageAmin + cp.confpars.cspadSpectrumBinWidth
            self.editCSImageAmax.setText( str(cp.confpars.cspadImageAmax))
        cp.confpars.fillCSPadConfigParsWinFromNamed(self.window)


    def processEditCSImageAmax(self):
        #print 'EditCSImageAmax'
        cp.confpars.fillCSPadConfigParsNamedFromWin(self.window)
        cp.confpars.cspadImageAmax = int(self.editCSImageAmax.displayText())        
        if cp.confpars.cspadImageAmax <= cp.confpars.cspadImageAmin + cp.confpars.cspadSpectrumBinWidth:
            cp.confpars.cspadImageAmin = cp.confpars.cspadImageAmax - cp.confpars.cspadSpectrumBinWidth
            self.editCSImageAmin.setText( str(cp.confpars.cspadImageAmin))
        cp.confpars.fillCSPadConfigParsWinFromNamed(self.window)


    def processEditCSSpectrumAmin(self):
        #print 'EditCSSpectrumAmin'
        cp.confpars.fillCSPadConfigParsNamedFromWin(self.window)
        cp.confpars.cspadSpectrumAmin = int(self.editCSSpectrumAmin.displayText())        
        if cp.confpars.cspadSpectrumAmin >= cp.confpars.cspadSpectrumAmax - cp.confpars.cspadSpectrumBinWidth:
            cp.confpars.cspadSpectrumAmax = cp.confpars.cspadSpectrumAmin + cp.confpars.cspadSpectrumBinWidth
            self.editCSSpectrumAmax.setText( str(cp.confpars.cspadSpectrumAmax))
        self.setBinning()
        cp.confpars.fillCSPadConfigParsWinFromNamed(self.window)


    def processEditCSSpectrumAmax(self):
        #print 'EditCSSpectrumAmax'
        cp.confpars.fillCSPadConfigParsNamedFromWin(self.window)
        cp.confpars.cspadSpectrumAmax  = int(self.editCSSpectrumAmax.displayText())        
        if cp.confpars.cspadSpectrumAmax <= cp.confpars.cspadSpectrumAmin + cp.confpars.cspadSpectrumBinWidth :
            cp.confpars.cspadSpectrumAmin = cp.confpars.cspadSpectrumAmax - cp.confpars.cspadSpectrumBinWidth
            self.editCSSpectrumAmin.setText( str(cp.confpars.cspadSpectrumAmin))
        self.setBinning()
        cp.confpars.fillCSPadConfigParsWinFromNamed(self.window)


    def processEditCSSpectrumNBins(self):
        #print 'EditCSSpectrumNBins'
        cp.confpars.fillCSPadConfigParsNamedFromWin(self.window)
        cp.confpars.cspadSpectrumNbins = int(self.editCSSpectrumNBins.displayText())        
        if  cp.confpars.cspadSpectrumNbins < 1 :
            cp.confpars.cspadSpectrumNbins = 1
            self.editCSSpectrumNBins.setText( str(cp.confpars.cspadSpectrumNbins) )
        cp.confpars.cspadWindowParameters[self.window][5] = cp.confpars.cspadSpectrumNbins
        self.setBinWidth()
        cp.confpars.fillCSPadConfigParsWinFromNamed(self.window)


    def processEditCSSpectrumBinWidth(self) : #,txt):
        #print 'EditCSSpectrumBinWidth'
        cp.confpars.fillCSPadConfigParsNamedFromWin(self.window)
        cp.confpars.cspadSpectrumBinWidth = int(self.editCSSpectrumBinWidth.displayText())        
        if  cp.confpars.cspadSpectrumBinWidth < 1 :
            cp.confpars.cspadSpectrumBinWidth = 1
            self.editCSSpectrumBinWidth.setText( str(cp.confpars.cspadSpectrumBinWidth) )
        cp.confpars.cspadWindowParameters[self.window][6] = cp.confpars.cspadSpectrumBinWidth        
        self.setNbins()
        self.setAmplitudeRange()
        cp.confpars.fillCSPadConfigParsWinFromNamed(self.window)


    def setNbins(self):
        cp.confpars.cspadSpectrumNbins = int((cp.confpars.cspadSpectrumAmax-cp.confpars.cspadSpectrumAmin)\
                                            / cp.confpars.cspadSpectrumBinWidth )
        if cp.confpars.cspadSpectrumNbins < 1 : cp.confpars.cspadSpectrumNbins = 1
        self.editCSSpectrumNBins.setText( str(cp.confpars.cspadSpectrumNbins) )


    def setBinWidth(self):
        cp.confpars.cspadSpectrumBinWidth = int((cp.confpars.cspadSpectrumAmax-cp.confpars.cspadSpectrumAmin)\
                                               / cp.confpars.cspadSpectrumNbins )
        if cp.confpars.cspadSpectrumBinWidth < 1 : cp.confpars.cspadSpectrumBinWidth = 1
        self.editCSSpectrumBinWidth.setText( str(cp.confpars.cspadSpectrumBinWidth) )


    def setBinning(self):
        if cp.confpars.cspadBinWidthIsOn : self.setNbins()    
        else :                             self.setBinWidth()


    def processRadioNBins(self):
        #print 'RadioNBins'
        cp.confpars.cspadBinWidthIsOn = cp.confpars.cspadWindowParameters[self.window][9] = False
        self.editCSSpectrumNBins   .setReadOnly(False)
        self.editCSSpectrumBinWidth.setReadOnly(True)
        self.editCSSpectrumNBins   .setPalette(self.palette_white)
        self.editCSSpectrumBinWidth.setPalette(self.palette_grey)


    def processRadioBinWidth(self):
        #print 'RadioBinWidth'
        cp.confpars.cspadBinWidthIsOn = cp.confpars.cspadWindowParameters[self.window][9] = True
        self.editCSSpectrumNBins   .setReadOnly(True)
        self.editCSSpectrumBinWidth.setReadOnly(False)
        self.editCSSpectrumNBins   .setPalette(self.palette_grey)
        self.editCSSpectrumBinWidth.setPalette(self.palette_white)


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


#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIWhatToDisplayForCSpadWindow()
    ex.show()
    app.exec_()
#-----------------------------

