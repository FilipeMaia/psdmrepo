#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplayForImageWindow...
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
import ConfigParameters as cp
import GlobalMethods    as gm
#---------------------
#  Class definition --
#---------------------
class GUIWhatToDisplayForImageWindow ( QtGui.QWidget ) :
    """Provides GUI to select information for rendering."""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, window=0):
        QtGui.QWidget.__init__(self, parent)

        self.window = window

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
        self.frame.setVisible(False)

        #titFont12 = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        titFont10 = QtGui.QFont("Sans Serif", 10, QtGui.QFont.Bold)

        self.titIMImage          = QtGui.QLabel('Image plot')
        self.titIMSpectrum       = QtGui.QLabel('Spectrum')
        self.titIMOffset         = QtGui.QLabel('Const. offset')

        self.titIMImage    .setFont (titFont10)   
        self.titIMSpectrum .setFont (titFont10)

        self.cboxImALimits   = QtGui.QCheckBox('A min/max:',self)
        self.cboxSpALimits   = QtGui.QCheckBox('A min/max:',self)

        self.radioBinWidth = QtGui.QRadioButton("Bin width:")
        self.radioNBins    = QtGui.QRadioButton("N bins:")
        self.radioGroupBin = QtGui.QButtonGroup()
        self.radioGroupBin.addButton(self.radioBinWidth)
        self.radioGroupBin.addButton(self.radioNBins)

        if cp.confpars.imageWindowParameters[self.window][9]  : self.radioBinWidth.setChecked(True)
        else :                                                  self.radioNBins.   setChecked(True)

        self.editIMImageAmin        = QtGui.QLineEdit(str(cp.confpars.imageWindowParameters[self.window][1]))
        self.editIMImageAmax        = QtGui.QLineEdit(str(cp.confpars.imageWindowParameters[self.window][2]))
        self.editIMSpectrumAmin     = QtGui.QLineEdit(str(cp.confpars.imageWindowParameters[self.window][3]))
        self.editIMSpectrumAmax     = QtGui.QLineEdit(str(cp.confpars.imageWindowParameters[self.window][4]))
        self.editIMSpectrumNBins    = QtGui.QLineEdit(str(cp.confpars.imageWindowParameters[self.window][5]))
        self.editIMSpectrumBinWidth = QtGui.QLineEdit(str(cp.confpars.imageWindowParameters[self.window][6]))
        self.editIMOffset           = QtGui.QLineEdit(str(cp.confpars.imageWindowParameters[self.window][10]))

        self.editIMImageAmin        .setValidator(QtGui.QIntValidator(-100000, 100000, self))
        self.editIMImageAmax        .setValidator(QtGui.QIntValidator(-100000, 100000, self))
        self.editIMSpectrumAmin     .setValidator(QtGui.QIntValidator(-100000, 100000, self))
        self.editIMSpectrumAmax     .setValidator(QtGui.QIntValidator(-100000, 100000, self))
        self.editIMSpectrumNBins    .setValidator(QtGui.QIntValidator(1, 10000, self))
        self.editIMSpectrumBinWidth .setValidator(QtGui.QIntValidator(1, 10000, self))

        self.editIMImageAmin        .setMaximumWidth(45)
        self.editIMImageAmax        .setMaximumWidth(45)
        self.editIMSpectrumAmin     .setMaximumWidth(45)
        self.editIMSpectrumAmax     .setMaximumWidth(45)
        self.editIMSpectrumNBins    .setMaximumWidth(45)
        self.editIMSpectrumBinWidth .setMaximumWidth(45)
        self.editIMOffset           .setMaximumWidth(65)
        
        self.titIMDataset  = QtGui.QLabel('Dataset:')
        self.butSelDataSet = QtGui.QPushButton(cp.confpars.imageWindowParameters[self.window][0])
        self.butSelDataSet.setMaximumWidth(350)
        self.setButSelDataSetTextAlignment()

        self.popupMenuForDataSet = QtGui.QMenu()
        self.fillPopupMenuForDataSet()

        #self.butClose = QtGui.QPushButton("Close window")

        gridIM = QtGui.QGridLayout()
        gridIM.addWidget(self.titIMDataset,           0, 0)
        gridIM.addWidget(self.butSelDataSet,          0, 1, 1, 6)
        gridIM.addWidget(self.titIMImage,             1, 0)
        gridIM.addWidget(self.cboxImALimits,          1, 1)
        gridIM.addWidget(self.editIMImageAmin,        1, 2)
        gridIM.addWidget(self.editIMImageAmax,        1, 3)
        
        gridIM.addWidget(self.titIMSpectrum,          2, 0)
        gridIM.addWidget(self.cboxSpALimits,          2, 1)
        gridIM.addWidget(self.editIMSpectrumAmin,     2, 2)
        gridIM.addWidget(self.editIMSpectrumAmax,     2, 3)

        gridIM.addWidget(self.radioNBins,             2, 5)
        gridIM.addWidget(self.radioBinWidth,          3, 5)
        gridIM.addWidget(self.editIMSpectrumNBins,    2, 6)
        gridIM.addWidget(self.editIMSpectrumBinWidth, 3, 6)

        gridIM.addWidget(self.titIMOffset,            3, 0)
        gridIM.addWidget(self.editIMOffset,           3, 1)

        
        #hboxC = QtGui.QHBoxLayout()
        #hboxC.addStretch(1)
        #hboxC.addWidget(self.butClose)
        
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(gridIM) 

        self.vbox.addStretch(1)     

        if parent == None :
            self.setLayout(self.vbox)
            self.show()

        self.connect(self.editIMImageAmin,        QtCore.SIGNAL('editingFinished ()'), self.processEditIMImageAmin )
        self.connect(self.editIMImageAmax,        QtCore.SIGNAL('editingFinished ()'), self.processEditIMImageAmax )
        self.connect(self.editIMSpectrumAmin,     QtCore.SIGNAL('editingFinished ()'), self.processEditIMSpectrumAmin )
        self.connect(self.editIMSpectrumAmax,     QtCore.SIGNAL('editingFinished ()'), self.processEditIMSpectrumAmax )
        self.connect(self.editIMSpectrumNBins,    QtCore.SIGNAL('editingFinished ()'), self.processEditIMSpectrumNBins )
        self.connect(self.editIMSpectrumBinWidth, QtCore.SIGNAL('editingFinished ()'), self.processEditIMSpectrumBinWidth )
        self.connect(self.editIMOffset,           QtCore.SIGNAL('editingFinished ()'), self.processEditIMOffset )

        self.connect(self.cboxImALimits,          QtCore.SIGNAL('stateChanged(int)'),  self.processCboxImALimits)
        self.connect(self.cboxSpALimits,          QtCore.SIGNAL('stateChanged(int)'),  self.processCboxSpALimits)

        self.connect(self.radioBinWidth,          QtCore.SIGNAL('clicked()'),          self.processRadioBinWidth )
        self.connect(self.radioNBins,             QtCore.SIGNAL('clicked()'),          self.processRadioNBins    )
        self.connect(self.butSelDataSet,          QtCore.SIGNAL('clicked()'),          self.processMenuForDataSet )
        #self.connect(self.butClose,               QtCore.SIGNAL('clicked()'),          self.processClose )

        #cp.confpars.wtdIMWindowIsOpen = True

        self.setBinning()
        self.setBinWidthReadOnly(not cp.confpars.imageWindowParameters[self.window][9])

        self.setCboxStatus()

    #-------------------
    # Private methods --
    #-------------------


    def setCboxStatus(self):
        if cp.confpars.imageWindowParameters[self.window][7] : self.cboxImALimits.setCheckState(2)
        if cp.confpars.imageWindowParameters[self.window][8] : self.cboxSpALimits.setCheckState(2)

        self.setImEditFieldsStatus()
        self.setSpEditFieldsStatus()

    def processCboxImALimits(self):
        if self.cboxImALimits.isChecked():
            cp.confpars.imageWindowParameters[self.window][7] = True
        else:
            cp.confpars.imageWindowParameters[self.window][7] = False
        self.setImEditFieldsStatus()

    def processCboxSpALimits(self):
        if self.cboxSpALimits.isChecked():
            cp.confpars.imageWindowParameters[self.window][8] = True
        else:
            cp.confpars.imageWindowParameters[self.window][8] = False
        self.setSpEditFieldsStatus()


    def setImEditFieldsStatus(self):
        if self.cboxImALimits.isChecked(): self.palette = self.palette_white
        else :                             self.palette = self.palette_grey                            
        self.editIMImageAmin.setPalette(self.palette)
        self.editIMImageAmax.setPalette(self.palette)
        self.editIMImageAmin.setReadOnly(not self.cboxImALimits.isChecked())
        self.editIMImageAmax.setReadOnly(not self.cboxImALimits.isChecked())


    def setSpEditFieldsStatus(self):
        if self.cboxSpALimits.isChecked(): self.palette = self.palette_white
        else :                             self.palette = self.palette_grey                            
        self.editIMSpectrumAmin.setPalette(self.palette)
        self.editIMSpectrumAmax.setPalette(self.palette)
        self.editIMSpectrumAmin.setReadOnly(not self.cboxSpALimits.isChecked())
        self.editIMSpectrumAmax.setReadOnly(not self.cboxSpALimits.isChecked())


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

  
    def getVBoxForLayout(self):
        return self.vbox


    def setParentWidget(self,parent):
        self.parentWidget = parent


    def closeEvent(self, event):
        #print 'closeEvent'
        pass


    def processClose(self):
        #print 'Close button'
        #cp.confpars.wtdIMWindowIsOpen = False
        self.close()

  
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
            #cspadIsInTheName = gm.CSpadIsInTheName(dsname)
            imageIsInTheName = gm.ImageIsInTheName(dsname)
            #if item_last_name == 'image' or cspadIsInTheName:
            if imageIsInTheName :

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
        cp.confpars.imageWindowParameters[self.window][0] = str(selected_ds)


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
        Amin     = cp.confpars.imageWindowParameters[self.window][3]
        Amax     = cp.confpars.imageWindowParameters[self.window][4]
        Nbins    = cp.confpars.imageWindowParameters[self.window][5]
        BinWidth = cp.confpars.imageWindowParameters[self.window][6]

        if cp.confpars.imageWindowParameters[self.window][9] :
            Nbins = (Amax - Amin) / BinWidth
            self.editIMSpectrumNBins.setText( str(Nbins) )
            cp.confpars.imageWindowParameters[self.window][5] = Nbins
        else :
            BinWidth = (Amax - Amin) / Nbins
            if BinWidth<1 : BinWidth=1
            self.editIMSpectrumBinWidth.setText( str(BinWidth) )  
            cp.confpars.imageWindowParameters[self.window][6] = BinWidth 

    def processRadioBinWidth(self):
        cp.confpars.imageWindowParameters[self.window][9] = True
        self.setBinWidthReadOnly(False)
        self.setBinning()
        
    def processRadioNBins(self):
        cp.confpars.imageWindowParameters[self.window][9] = False
        self.setBinning()
        self.setBinWidthReadOnly(True)

    def processEditIMSpectrumNBins   (self):
        cp.confpars.imageWindowParameters[self.window][5] = int(self.editIMSpectrumNBins.displayText())
        self.setBinning()

    def processEditIMSpectrumBinWidth(self):
        cp.confpars.imageWindowParameters[self.window][6] = int(self.editIMSpectrumBinWidth.displayText())
        self.setBinning()

    def processEditIMImageAmin(self):
        #print 'EditIMImageAmin'
        cp.confpars.imageWindowParameters[self.window][1] = int(self.editIMImageAmin.displayText())        

    def processEditIMImageAmax(self):
        #print 'EditIMImageAmax'
        cp.confpars.imageWindowParameters[self.window][2] = int(self.editIMImageAmax.displayText())        

    def processEditIMSpectrumAmin(self):
        #print 'EditIMSpectrumAmin'
        cp.confpars.imageWindowParameters[self.window][3] = int(self.editIMSpectrumAmin.displayText())        
        self.setBinning()

    def processEditIMSpectrumAmax(self):
        #print 'EditIMSpectrumAmax'
        cp.confpars.imageWindowParameters[self.window][4] = int(self.editIMSpectrumAmax.displayText())        
        self.setBinning()

    def processEditIMOffset(self):
        #print 'EditIMOffset'
        cp.confpars.imageWindowParameters[self.window][10] = int(self.editIMOffset.displayText())        

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIWhatToDisplayForImageWindow()
    ex.show()
    app.exec_()
#-----------------------------

