#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUICorrelationWindow...
#
#------------------------------------------------------------------------

"""GUI manipulates with parameters for each correlation plot.

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
import PrintHDF5        as printh5
import GlobalMethods    as gm

#---------------------
#  Class definition --
#---------------------
class GUICorrelationWindow ( QtGui.QWidget ) :
    """GUI manipulates with parameters for each correlation plot"""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, window=0):
        QtGui.QWidget.__init__(self, parent)

        print 'GUICorrelationWindow for plot', window

        self.window = window

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Adjust correlation parameters')

        self.palette = QtGui.QPalette()
        self.styleSheetRed   = "background-color: rgb(255, 0, 0); color: rgb(255, 255, 255)"
        self.styleSheetGreen = "background-color: rgb(0, 255, 0); color: rgb(255, 255, 255)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"
        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"

        titFont12 = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        titFont10 = QtGui.QFont("Sans Serif", 10, QtGui.QFont.Bold)

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

        #self.titYminmax= QtGui.QLabel('Ylims:')
        #self.titXminmax= QtGui.QLabel('Xlims:')

        self.cboxYlimits   = QtGui.QCheckBox('Ylims:',self)
        self.cboxXlimits   = QtGui.QCheckBox('Xlims:',self)

        if cp.confpars.correlationWindowParameters[self.window][9]  : self.cboxYlimits.setCheckState(2)
        if cp.confpars.correlationWindowParameters[self.window][10] : self.cboxXlimits.setCheckState(2)

        self.char_expand = u'\u25BE' # down-head triangle
        height = 20
        width  = 60

        self.editCorrelationYmin  = QtGui.QLineEdit(str(cp.confpars.correlationWindowParameters[self.window][3]))
        self.editCorrelationYmax  = QtGui.QLineEdit(str(cp.confpars.correlationWindowParameters[self.window][4]))
        self.editCorrelationXmin  = QtGui.QLineEdit(str(cp.confpars.correlationWindowParameters[self.window][5]))
        self.editCorrelationXmax  = QtGui.QLineEdit(str(cp.confpars.correlationWindowParameters[self.window][6]))
        self.editCorrelationYNBins= QtGui.QLineEdit(str(cp.confpars.correlationWindowParameters[self.window][12]))
        self.editCorrelationXNBins= QtGui.QLineEdit(str(cp.confpars.correlationWindowParameters[self.window][13]))

        self.editCorrelationYmin  .setMaximumWidth(width)
        self.editCorrelationYmax  .setMaximumWidth(width)
        self.editCorrelationXmin  .setMaximumWidth(width)
        self.editCorrelationXmax  .setMaximumWidth(width)
        self.editCorrelationYNBins.setMaximumWidth(width)
        self.editCorrelationXNBins.setMaximumWidth(width)


        self.editCorrelationYmin  .setMaximumHeight(height)
        self.editCorrelationYmax  .setMaximumHeight(height)
        self.editCorrelationXmin  .setMaximumHeight(height)
        self.editCorrelationXmax  .setMaximumHeight(height)
        self.editCorrelationYNBins.setMaximumHeight(height)
        self.editCorrelationXNBins.setMaximumHeight(height)

        self.editCorrelationYmin  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))
        self.editCorrelationYmax  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))
        self.editCorrelationXmin  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))
        self.editCorrelationXmax  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))
        self.editCorrelationYNBins.setValidator(QtGui.QIntValidator(1, 1000, self))
        self.editCorrelationXNBins.setValidator(QtGui.QIntValidator(1, 1000, self))
        
        self.titYNBins     = QtGui.QLabel('Y Nbins:')
        self.titXNBins     = QtGui.QLabel('X Nbins:')
        self.titVs         = QtGui.QLabel('Versus:')

        self.radioLogZ     = QtGui.QRadioButton('log Z')
        self.radioLinZ     = QtGui.QRadioButton('lin Z')

        self.radioGroupZScale = QtGui.QButtonGroup()
        self.radioGroupZScale.addButton(self.radioLogZ)
        self.radioGroupZScale.addButton(self.radioLinZ)

        self.setZScaleRadioButtons()


        self.radioVsIndex  = QtGui.QRadioButton('Index')
        self.radioVsTime   = QtGui.QRadioButton('Time' )
        self.radioVsXPar   = QtGui.QRadioButton('X-par')
        self.radioYHist    = QtGui.QRadioButton('Y-histo.')

        self.radioGroup    = QtGui.QButtonGroup()
        self.radioGroup.addButton(self.radioVsIndex)
        self.radioGroup.addButton(self.radioVsTime )
        self.radioGroup.addButton(self.radioVsXPar )
        self.radioGroup.addButton(self.radioYHist  )

        self.titCorrXDataSet = QtGui.QLabel('X-par:')
        self.titCorrYDataSet = QtGui.QLabel('Y-par:')
        #self.titCorrXDataSet.setFont (titFont10)   
        #self.titCorrYDataSet.setFont (titFont10)   
        self.butCorrXDataSet = QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][1])
        self.butCorrYDataSet = QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][0])
        self.butCorrXParName = QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][8])
        self.butCorrYParName = QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][7])
        self.butCorrXParIndex= QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][15])
        self.butCorrYParIndex= QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][14])

        self.butCorrXDataSet  .setMaximumHeight(height)
        self.butCorrYDataSet  .setMaximumHeight(height)
        self.butCorrXParName  .setMaximumHeight(height)
        self.butCorrYParName  .setMaximumHeight(height)
        self.butCorrXParIndex .setMaximumHeight(height)
        self.butCorrYParIndex .setMaximumHeight(height)

        self.butCorrXDataSet.setMaximumWidth(295)
        self.butCorrYDataSet.setMaximumWidth(295)
        

        self.setButCorrXDataSetTextAlignment()
        self.setButCorrYDataSetTextAlignment()

        self.popupMenuForDataSet = QtGui.QMenu()
        self.fillPopupMenuForDataSet()

        self.popupMenuForXParName = QtGui.QMenu()
        self.fillPopupMenuForXParName()

        self.popupMenuForYParName = QtGui.QMenu()
        self.fillPopupMenuForYParName()

        self.popupMenuForXParIndex = QtGui.QMenu()
        self.fillPopupMenuForXParIndex()

        self.popupMenuForYParIndex = QtGui.QMenu()
        self.fillPopupMenuForYParIndex()

        grid = QtGui.QGridLayout()

        grid.addWidget(self.titCorrYDataSet,      0, 0)
        grid.addWidget(self.butCorrYDataSet,      0, 1, 1, 4)
        grid.addWidget(self.butCorrYParName,      0, 5)
        grid.addWidget(self.butCorrYParIndex,     0, 6)

        grid.addWidget(self.titVs,                1, 0)
        grid.addWidget(self.radioVsIndex,         1, 1)
        grid.addWidget(self.radioVsTime,          1, 2)
        grid.addWidget(self.radioVsXPar,          1, 3)
        grid.addWidget(self.radioYHist,           1, 6)

        grid.addWidget(self.titCorrXDataSet,      2, 0)
        grid.addWidget(self.butCorrXDataSet,      2, 1, 1, 4)
        grid.addWidget(self.butCorrXParName,      2, 5)
        grid.addWidget(self.butCorrXParIndex,     2, 6)
        
        grid.addWidget(self.cboxYlimits,          3, 0)
        grid.addWidget(self.editCorrelationYmin,  3, 1)
        grid.addWidget(self.editCorrelationYmax,  3, 2)
        grid.addWidget(self.titYNBins,            3, 3)
        grid.addWidget(self.editCorrelationYNBins,3, 5)
        grid.addWidget(self.radioLinZ,            3, 6)

        grid.addWidget(self.cboxXlimits,          4, 0)
        grid.addWidget(self.editCorrelationXmin,  4, 1)
        grid.addWidget(self.editCorrelationXmax,  4, 2)
        grid.addWidget(self.titXNBins,            4, 3)
        grid.addWidget(self.editCorrelationXNBins,4, 5)
        grid.addWidget(self.radioLogZ,            4, 6)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(grid) 
        self.vbox.addStretch(1)     

        self.setButStatus()
        self.setEditFieldsStatus()

        if parent == None :
            self.setLayout(self.vbox)
            self.show()

        self.connect(self.radioVsIndex,          QtCore.SIGNAL('clicked()'),          self.processRadioVsIndex )
        self.connect(self.radioVsTime,           QtCore.SIGNAL('clicked()'),          self.processRadioVsTime )
        self.connect(self.radioVsXPar,           QtCore.SIGNAL('clicked()'),          self.processRadioVsXPar )
        self.connect(self.radioYHist,            QtCore.SIGNAL('clicked()'),          self.processRadioYHist )

        self.connect(self.butCorrXDataSet,       QtCore.SIGNAL('clicked()'),          self.processMenuForXDataSet )
        self.connect(self.butCorrYDataSet,       QtCore.SIGNAL('clicked()'),          self.processMenuForYDataSet )
        self.connect(self.butCorrXParName,       QtCore.SIGNAL('clicked()'),          self.processMenuForXParName )
        self.connect(self.butCorrYParName,       QtCore.SIGNAL('clicked()'),          self.processMenuForYParName )
        self.connect(self.butCorrXParIndex,      QtCore.SIGNAL('clicked()'),          self.processMenuForXParIndex )
        self.connect(self.butCorrYParIndex,      QtCore.SIGNAL('clicked()'),          self.processMenuForYParIndex )

        self.connect(self.editCorrelationYmin,   QtCore.SIGNAL('editingFinished ()'), self.processEditCorrelationYmin )
        self.connect(self.editCorrelationYmax,   QtCore.SIGNAL('editingFinished ()'), self.processEditCorrelationYmax )
        self.connect(self.editCorrelationXmin,   QtCore.SIGNAL('editingFinished ()'), self.processEditCorrelationXmin )
        self.connect(self.editCorrelationXmax,   QtCore.SIGNAL('editingFinished ()'), self.processEditCorrelationXmax )

        self.connect(self.cboxYlimits,           QtCore.SIGNAL('stateChanged(int)'),  self.processCboxYlimits)
        self.connect(self.cboxXlimits,           QtCore.SIGNAL('stateChanged(int)'),  self.processCboxXlimits)

        self.connect(self.editCorrelationYNBins, QtCore.SIGNAL('editingFinished ()'), self.processEditCorrelationYNBins )
        self.connect(self.editCorrelationXNBins, QtCore.SIGNAL('editingFinished ()'), self.processEditCorrelationXNBins )
        self.connect(self.radioLogZ,             QtCore.SIGNAL('clicked()'),          self.processRadioLogZ )
        self.connect(self.radioLinZ,             QtCore.SIGNAL('clicked()'),          self.processRadioLinZ )
  
        cp.confpars.selectionWindowIsOpen = True

        self.showToolTips()

        self.setMinimumHeight(250)
        self.setMaximumHeight(300)

    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters for waveforms.')
        self.radioVsIndex      .setToolTip('Select X axis parameter between Index/Time/Parameter')
        self.radioVsTime       .setToolTip('Select X axis parameter between Index/Time/Parameter')
        self.radioVsXPar       .setToolTip('Select X axis parameter between Index/Time/Parameter')
        self.radioYHist        .setToolTip('Select 1-D histogram for Y-parameter')
        self.butCorrYDataSet   .setToolTip('Select the dataset for the Y axis parameter.\n' +\
                                           'The list of datasets is formed from the checked items in the HDF5 tree GUI')
        self.butCorrYParName   .setToolTip('Select the parameter name in the dataset.\n' +\
                                           'The dataset should be selected first in the left box')


    def resizeEvent(self, e):
        self.frame.setGeometry(self.rect())


    def getVBoxForLayout(self):
        return self.vbox


    def setParentWidget(self,parent):
        self.parentWidget = parent


    def closeEvent(self, event):
        cp.confpars.correlationWindowIsOpen = False


    def processClose(self):
        self.close()


    def setButStatus(self):
        if   cp.confpars.correlationWindowParameters[self.window][2] == 0 :
            self.radioVsIndex.setChecked(True)
            self.processRadioVsIndex()

        elif cp.confpars.correlationWindowParameters[self.window][2] == 1 :
            self.radioVsTime.setChecked(True)
            self.processRadioVsTime()

        elif cp.confpars.correlationWindowParameters[self.window][2] == 2 :
            self.radioVsXPar.setChecked(True)
            self.processRadioVsXPar()

        elif cp.confpars.correlationWindowParameters[self.window][2] == 3 :
            self.radioYHist.setChecked(True)
            self.processRadioYHist()

        self.setButCorrYDataSetTextAlignment()


    def processRadioLogZ(self): 
        cp.confpars.correlationWindowParameters[self.window][11] = True 
        #self.radioLogZ.setChecked(True)


    def processRadioLinZ(self): 
        cp.confpars.correlationWindowParameters[self.window][11] = False 
        #self.radioLinZ.setChecked(True)


    def processRadioVsIndex(self):
        cp.confpars.correlationWindowParameters[self.window][2] = 0
        dsname = 'Index'
        self.butCorrXDataSet.setText(dsname)
        cp.confpars.correlationWindowParameters[self.window][1] = str(dsname)

        self.butCorrXParName.setText('None')
        cp.confpars.correlationWindowParameters[self.window][8] = 'None'

        self.setButCorrXDataSetTextAlignment()
        self.setEditFieldsStatusForMode()
        
            
    def processRadioVsTime(self):
        cp.confpars.correlationWindowParameters[self.window][2] = 1
        dsname = 'Time'
        self.butCorrXDataSet.setText(dsname)
        cp.confpars.correlationWindowParameters[self.window][1] = str(dsname)

        self.butCorrXParName.setText('None')
        cp.confpars.correlationWindowParameters[self.window][8] = 'None'

        self.setButCorrXDataSetTextAlignment()
        self.setEditFieldsStatusForMode()
        

    def processRadioYHist(self):
        cp.confpars.correlationWindowParameters[self.window][2] = 3
        dsname = 'Is-not-used'
        self.butCorrXDataSet.setText(dsname)
        cp.confpars.correlationWindowParameters[self.window][1] = str(dsname)

        self.butCorrXParName.setText('None')
        cp.confpars.correlationWindowParameters[self.window][8] = 'None'

        self.setButCorrXDataSetTextAlignment()
        self.setEditFieldsStatusForMode()  


    def processRadioVsXPar(self):
        cp.confpars.correlationWindowParameters[self.window][2] = 2

        dsname = cp.confpars.correlationWindowParameters[self.window][1]

        if     dsname == 'None' \
            or dsname == 'Time' \
            or dsname == 'Index' \
            or dsname == 'Is-not-used' :

            self.butCorrXDataSet.setText('Select-X-parameter')
            self.butCorrXParName.setText('None')
            cp.confpars.correlationWindowParameters[self.window][8] = 'None'

        else :
            self.butCorrXDataSet.setText(dsname)

        self.setButCorrXDataSetTextAlignment()
        #self.setButCorrXParNameTextAlignment()
        self.setEditFieldsStatusForMode()


    def setZScaleRadioButtons(self):
            if cp.confpars.correlationWindowParameters[self.window][11] :
                self.radioLogZ.setChecked(True)
            else :
                self.radioLinZ.setChecked(True)

        
    def setEditFieldsStatusForMode(self):

        self.cboxXlimits.setCheckable(True)
        #self.processRadioLinZ(self)
        self.radioLogZ.setCheckable(False)
  
        if   cp.confpars.correlationWindowParameters[self.window][2] == 0 :
            self.editCorrelationYNBins.setStyleSheet(self.styleSheetGrey)
            self.editCorrelationXNBins.setStyleSheet(self.styleSheetGrey)
            self.editCorrelationYNBins.setReadOnly(True)
            self.editCorrelationXNBins.setReadOnly(True)

        elif cp.confpars.correlationWindowParameters[self.window][2] == 1 :
            self.editCorrelationYNBins.setStyleSheet(self.styleSheetGrey)
            self.editCorrelationXNBins.setStyleSheet(self.styleSheetGrey)
            self.editCorrelationYNBins.setReadOnly(True)
            self.editCorrelationXNBins.setReadOnly(True)

        elif cp.confpars.correlationWindowParameters[self.window][2] == 2 :
            self.editCorrelationYNBins.setStyleSheet(self.styleSheetWhite)
            self.editCorrelationXNBins.setStyleSheet(self.styleSheetWhite)
            self.editCorrelationYNBins.setReadOnly(False)
            self.editCorrelationXNBins.setReadOnly(False)
            self.radioLogZ.setCheckable(True)
            self.setZScaleRadioButtons()

        elif cp.confpars.correlationWindowParameters[self.window][2] == 3 :
            self.editCorrelationYNBins.setStyleSheet(self.styleSheetWhite)
            self.editCorrelationXNBins.setStyleSheet(self.styleSheetGrey)
            self.editCorrelationYNBins.setReadOnly(False)
            self.editCorrelationXNBins.setReadOnly(True)
            self.cboxXlimits.setCheckState(0)
            self.cboxXlimits.setCheckable(False)


    def setEditFieldsStatus(self):

        if self.cboxYlimits.isChecked() :
            self.editCorrelationYmin.setStyleSheet(self.styleSheetWhite)
            self.editCorrelationYmax.setStyleSheet(self.styleSheetWhite)
            self.editCorrelationYmin.setReadOnly(False)
            self.editCorrelationYmax.setReadOnly(False)
        else :
            self.editCorrelationYmin.setStyleSheet(self.styleSheetGrey)
            self.editCorrelationYmax.setStyleSheet(self.styleSheetGrey)
            self.editCorrelationYmin.setReadOnly(True)
            self.editCorrelationYmax.setReadOnly(True)

        if self.cboxXlimits.isChecked() :
            self.editCorrelationXmin.setStyleSheet(self.styleSheetWhite)
            self.editCorrelationXmax.setStyleSheet(self.styleSheetWhite)
            self.editCorrelationXmin.setReadOnly(False)
            self.editCorrelationXmax.setReadOnly(False)
        else :
            self.editCorrelationXmin.setStyleSheet(self.styleSheetGrey)
            self.editCorrelationXmax.setStyleSheet(self.styleSheetGrey)
            self.editCorrelationXmin.setReadOnly(True)
            self.editCorrelationXmax.setReadOnly(True)


    def processCboxYlimits(self):
        if self.cboxYlimits.isChecked():
            cp.confpars.correlationWindowParameters[self.window][9] = True
        else:
            cp.confpars.correlationWindowParameters[self.window][9] = False
        self.setEditFieldsStatus()


    def processCboxXlimits(self):
        if self.cboxXlimits.isChecked():
            cp.confpars.correlationWindowParameters[self.window][10] = True
        else:
            cp.confpars.correlationWindowParameters[self.window][10] = False
        self.setEditFieldsStatus()


    def processEditCorrelationYmin(self):
        cp.confpars.correlationWindowParameters[self.window][3] = int(self.editCorrelationYmin.displayText())        


    def processEditCorrelationYmax(self):
        cp.confpars.correlationWindowParameters[self.window][4] = int(self.editCorrelationYmax.displayText())        


    def processEditCorrelationXmin(self):
        cp.confpars.correlationWindowParameters[self.window][5] = int(self.editCorrelationXmin.displayText())        


    def processEditCorrelationXmax(self):
        cp.confpars.correlationWindowParameters[self.window][6] = int(self.editCorrelationXmax.displayText())        


    def processEditCorrelationYNBins(self):
        cp.confpars.correlationWindowParameters[self.window][12] = int(self.editCorrelationYNBins.displayText())


    def processEditCorrelationXNBins(self):
        cp.confpars.correlationWindowParameters[self.window][13] = int(self.editCorrelationXNBins.displayText())


    def setButCorrYDataSetTextAlignment(self):
        if  self.butCorrYDataSet.text() == 'None' :
            self.butCorrYDataSet.setStyleSheet('Text-align:center')
            self.butCorrYDataSet.setStyleSheet(self.styleSheetRed)
            self.butCorrYParName.setText('None')
            self.butCorrYParIndex.setText('None')
            cp.confpars.correlationWindowParameters[self.window][7] = 'None'
            cp.confpars.correlationWindowParameters[self.window][14] = 'None'
        else :
            self.butCorrYDataSet.setStyleSheet('Text-align:right;' + self.styleSheetWhite)

        self.setButCorrYParNameTextAlignment()


    def setButCorrXDataSetTextAlignment(self):
        if     self.butCorrXDataSet.text() == 'Time' \
            or self.butCorrXDataSet.text() == 'Index' \
            or self.butCorrXDataSet.text() == 'Is-not-used' :
            self.butCorrXDataSet.setStyleSheet('Text-align:center;' + self.styleSheetWhite)
            self.butCorrXParName.setStyleSheet(self.styleSheetWhite)
            self.butCorrXParName.setText('None')
            self.butCorrXParIndex.setText('None')

        elif   self.butCorrXDataSet.text() == 'None' \
            or self.butCorrXDataSet.text() == 'Select-X-parameter' :
            self.butCorrXDataSet.setStyleSheet('Text-align:center;' + self.styleSheetRed)
            self.butCorrXParName.setStyleSheet(self.styleSheetRed)
            self.butCorrXParName.setText('None')
            self.butCorrXParIndex.setText('None')
            cp.confpars.correlationWindowParameters[self.window][8] = 'None'
            cp.confpars.correlationWindowParameters[self.window][15] = 'None'
        else :
            self.butCorrXDataSet.setStyleSheet('Text-align:right;' + self.styleSheetWhite)


    def setButCorrYParNameTextAlignment(self):
        if self.butCorrYParName.text() == 'None' :
            self.butCorrYParName.setStyleSheet(self.styleSheetRed)
        else :
            self.butCorrYParName.setStyleSheet(self.styleSheetWhite)


    def setButCorrXParNameTextAlignment(self):
        if  self.butCorrXParName.text() == 'None' :
            self.butCorrXParName.setStyleSheet(self.styleSheetRed)
        else :
            self.butCorrXParName.setStyleSheet(self.styleSheetWhite)


    def fillPopupMenuForDataSet(self):
        print 'fillPopupMenuForDataSet'
        self.popupMenuForDataSet.addAction('None')
        for dsname in cp.confpars.list_of_checked_item_names :

            item_last_name   = gm.get_item_last_name(dsname)           
            if item_last_name == 'waveforms'    : continue
            if item_last_name == 'timestamps'   : continue
            if gm.ImageIsInTheName(dsname)      : continue
            if gm.CSpadIsInTheName(dsname)      : continue
 
            self.popupMenuForDataSet.addAction(dsname)


    def processMenuForYDataSet(self):
        print 'MenuForYDataSet'
        actionSelected = self.popupMenuForDataSet.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_dsname = actionSelected.text()
        self.butCorrYDataSet.setText( selected_dsname )
        cp.confpars.correlationWindowParameters[self.window][0] = str(selected_dsname)
        self.butCorrYParName.setText('None')
        self.butCorrYParIndex.setText('None')
        cp.confpars.correlationWindowParameters[self.window][7] = 'None'
        cp.confpars.correlationWindowParameters[self.window][14] = 'None'
        self.setButCorrYDataSetTextAlignment()



    def processMenuForXDataSet(self):
        print 'MenuForXDataSet'
        if cp.confpars.correlationWindowParameters[self.window][2] < 2 : return # for Index and Time

        actionSelected = self.popupMenuForDataSet.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_dsname = actionSelected.text()
        self.butCorrXDataSet.setText( selected_dsname )
        cp.confpars.correlationWindowParameters[self.window][1] = str(selected_dsname)

        self.setButCorrXDataSetTextAlignment()


    def fillPopupMenuForYParName(self):
        print 'fillPopupMenuForYParName'
        dsname = cp.confpars.correlationWindowParameters[self.window][0]
        print 'dsname=', dsname
        self.listOfDatasetParNames = printh5.getListOfDatasetParNames(dsname)
        del self.popupMenuForYParName
        self.popupMenuForYParName=QtGui.QMenu()
        for parName in self.listOfDatasetParNames :
            self.popupMenuForYParName.addAction(parName)

  
    def fillPopupMenuForXParName(self):
        print 'fillPopupMenuForXParName'
        dsname = cp.confpars.correlationWindowParameters[self.window][1]
        print 'dsname=', dsname
        self.listOfDatasetParNames = printh5.getListOfDatasetParNames(dsname)
        del self.popupMenuForXParName
        self.popupMenuForXParName=QtGui.QMenu()
        for parName in self.listOfDatasetParNames :
            self.popupMenuForXParName.addAction(parName)


    def processMenuForYParName(self):
        print 'MenuForYParName'
        self.fillPopupMenuForYParName()
        actionSelected = self.popupMenuForYParName.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_dsname = actionSelected.text()
        selected_ind    = self.listOfDatasetParNames.index(selected_dsname)
        print 'selected_ind = ', selected_ind
        print 'selected_dsname = ', selected_dsname
        self.butCorrYParName.setText( selected_dsname )
        cp.confpars.correlationWindowParameters[self.window][7] = str(selected_dsname)
        self.butCorrYParIndex.setText('None')
        cp.confpars.correlationWindowParameters[self.window][14] = 'None'

        self.setButCorrYParNameTextAlignment()
        self.cboxYlimits.setCheckState(0)


    def processMenuForXParName(self):
        print 'MenuForXParName'
        if cp.confpars.correlationWindowParameters[self.window][2] < 2 : return # for Index and Time

        self.fillPopupMenuForXParName()
        actionSelected = self.popupMenuForXParName.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_dsname = actionSelected.text()
        selected_ind    = self.listOfDatasetParNames.index(selected_dsname)
        print 'selected_ind = ', selected_ind
        print 'selected_dsname = ', selected_dsname
        self.butCorrXParName.setText( selected_dsname )
        cp.confpars.correlationWindowParameters[self.window][8]  = str(selected_dsname)
        self.butCorrXParIndex.setText('None')
        cp.confpars.correlationWindowParameters[self.window][15] = 'None'

        self.setButCorrXParNameTextAlignment()
        self.cboxXlimits.setCheckState(0)











    def fillPopupMenuForXParIndex(self):
        print 'fillPopupMenuForXParIndex'
        dsname  = cp.confpars.correlationWindowParameters[self.window][1]
        parname = cp.confpars.correlationWindowParameters[self.window][8]
        print 'dsname=', dsname, '   parname=', parname
        self.listOfDatasetParIndexes = printh5.getListOfDatasetParIndexes(dsname,parname)
        del self.popupMenuForXParIndex
        self.popupMenuForXParIndex=QtGui.QMenu()
        for parIndex in self.listOfDatasetParIndexes :
            self.popupMenuForXParIndex.addAction(parIndex)

    def fillPopupMenuForYParIndex(self):
        print 'fillPopupMenuForYParIndex'
        dsname  = cp.confpars.correlationWindowParameters[self.window][0]
        parname = cp.confpars.correlationWindowParameters[self.window][7]
        print 'dsname=', dsname, '   parname=', parname
        self.listOfDatasetParIndexes = printh5.getListOfDatasetParIndexes(dsname,parname)
        del self.popupMenuForYParIndex
        self.popupMenuForYParIndex=QtGui.QMenu()
        for parIndex in self.listOfDatasetParIndexes :
            self.popupMenuForYParIndex.addAction(parIndex)

    def processMenuForYParIndex(self):
        print 'MenuForYParIndex'
        self.fillPopupMenuForYParIndex()
        actionSelected = self.popupMenuForYParIndex.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected         = actionSelected.text()
        selected_ind     = self.listOfDatasetParIndexes.index(selected)
        print 'selected = ', selected
        self.butCorrYParIndex.setText( selected )
        cp.confpars.correlationWindowParameters[self.window][14] = str(selected)

    def processMenuForXParIndex(self):
        print 'MenuForXParIndex'
        self.fillPopupMenuForXParIndex()
        actionSelected = self.popupMenuForXParIndex.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected         = actionSelected.text()
        selected_ind     = self.listOfDatasetParIndexes.index(selected)
        print 'selected = ', selected
        self.butCorrXParIndex.setText( selected )
        cp.confpars.correlationWindowParameters[self.window][15] = str(selected)

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUICorrelationWindow()
    ex.show()
    app.exec_()
#-----------------------------

