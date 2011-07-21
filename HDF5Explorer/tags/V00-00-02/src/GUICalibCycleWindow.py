#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUICalibCycleWindow...
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
class GUICalibCycleWindow ( QtGui.QWidget ) :
    """GUI manipulates with parameters for each correlation plot"""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, window=0):
        QtGui.QWidget.__init__(self, parent)

        print 'GUICalibCycleWindow for plot', window

        self.window = window

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Adjust calibcycle parameters')

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

        self.cboxYlimits   = QtGui.QCheckBox('Ylims:',self)
        self.cboxXlimits   = QtGui.QCheckBox('Xlims:',self)

        if cp.confpars.calibcycleWindowParameters[self.window][9]  : self.cboxYlimits.setCheckState(2)
        if cp.confpars.calibcycleWindowParameters[self.window][10] : self.cboxXlimits.setCheckState(2)

        self.char_expand = u'\u25BE' # down-head triangle
        height = 20
        width  = 60

        self.editCalibCycleYmin  = QtGui.QLineEdit(str(cp.confpars.calibcycleWindowParameters[self.window][3]))
        self.editCalibCycleYmax  = QtGui.QLineEdit(str(cp.confpars.calibcycleWindowParameters[self.window][4]))
        self.editCalibCycleXmin  = QtGui.QLineEdit(str(cp.confpars.calibcycleWindowParameters[self.window][5]))
        self.editCalibCycleXmax  = QtGui.QLineEdit(str(cp.confpars.calibcycleWindowParameters[self.window][6]))
        self.editCalibCycleYNBins= QtGui.QLineEdit(str(cp.confpars.calibcycleWindowParameters[self.window][12]))
        self.editCalibCycleXNBins= QtGui.QLineEdit(str(cp.confpars.calibcycleWindowParameters[self.window][13]))

        self.editCalibCycleYmin  .setMaximumWidth(width)
        self.editCalibCycleYmax  .setMaximumWidth(width)
        self.editCalibCycleXmin  .setMaximumWidth(width)
        self.editCalibCycleXmax  .setMaximumWidth(width)
        self.editCalibCycleYNBins.setMaximumWidth(width)
        self.editCalibCycleXNBins.setMaximumWidth(width)


        self.editCalibCycleYmin  .setMaximumHeight(height)
        self.editCalibCycleYmax  .setMaximumHeight(height)
        self.editCalibCycleXmin  .setMaximumHeight(height)
        self.editCalibCycleXmax  .setMaximumHeight(height)
        self.editCalibCycleYNBins.setMaximumHeight(height)
        self.editCalibCycleXNBins.setMaximumHeight(height)

        self.editCalibCycleYmin  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))
        self.editCalibCycleYmax  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))
        self.editCalibCycleXmin  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))
        self.editCalibCycleXmax  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))
        self.editCalibCycleYNBins.setValidator(QtGui.QIntValidator(1, 1000, self))
        self.editCalibCycleXNBins.setValidator(QtGui.QIntValidator(1, 1000, self))
        
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

        self.titCalibCXDataSet = QtGui.QLabel('X-par:')
        self.titCalibCYDataSet = QtGui.QLabel('Y-par:')
        self.butCalibCXDataSet = QtGui.QPushButton(cp.confpars.calibcycleWindowParameters[self.window][1])
        self.butCalibCYDataSet = QtGui.QPushButton(cp.confpars.calibcycleWindowParameters[self.window][0])
        self.butCalibCXParName = QtGui.QPushButton(cp.confpars.calibcycleWindowParameters[self.window][8])
        self.butCalibCYParName = QtGui.QPushButton(cp.confpars.calibcycleWindowParameters[self.window][7])
        self.butCalibCXParIndex= QtGui.QPushButton(cp.confpars.calibcycleWindowParameters[self.window][15])
        self.butCalibCYParIndex= QtGui.QPushButton(cp.confpars.calibcycleWindowParameters[self.window][14])

        self.butCalibCXDataSet  .setMaximumHeight(height)
        self.butCalibCYDataSet  .setMaximumHeight(height)
        self.butCalibCXParName  .setMaximumHeight(height)
        self.butCalibCYParName  .setMaximumHeight(height)
        self.butCalibCXParIndex .setMaximumHeight(height)
        self.butCalibCYParIndex .setMaximumHeight(height)

        self.butCalibCXDataSet.setMaximumWidth(295)
        self.butCalibCYDataSet.setMaximumWidth(295)
        

        self.setButCalibCXDataSetTextAlignment()
        self.setButCalibCYDataSetTextAlignment()

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

        grid.addWidget(self.titCalibCYDataSet,      0, 0)
        grid.addWidget(self.butCalibCYDataSet,      0, 1, 1, 4)
        grid.addWidget(self.butCalibCYParName,      0, 5)
        grid.addWidget(self.butCalibCYParIndex,     0, 6)

        grid.addWidget(self.titVs,                  1, 0)
        grid.addWidget(self.radioVsIndex,           1, 1)
        grid.addWidget(self.radioVsTime,            1, 2)
        grid.addWidget(self.radioVsXPar,            1, 3)
        grid.addWidget(self.radioYHist,             1, 6)

        grid.addWidget(self.titCalibCXDataSet,      2, 0)
        grid.addWidget(self.butCalibCXDataSet,      2, 1, 1, 4)
        grid.addWidget(self.butCalibCXParName,      2, 5)
        grid.addWidget(self.butCalibCXParIndex,     2, 6)
        
        grid.addWidget(self.cboxYlimits,            3, 0)
        grid.addWidget(self.editCalibCycleYmin,     3, 1)
        grid.addWidget(self.editCalibCycleYmax,     3, 2)
        grid.addWidget(self.titYNBins,              3, 3)
        grid.addWidget(self.editCalibCycleYNBins,   3, 5)
        grid.addWidget(self.radioLinZ,              3, 6)

        grid.addWidget(self.cboxXlimits,            4, 0)
        grid.addWidget(self.editCalibCycleXmin,     4, 1)
        grid.addWidget(self.editCalibCycleXmax,     4, 2)
        grid.addWidget(self.titXNBins,              4, 3)
        grid.addWidget(self.editCalibCycleXNBins,   4, 5)
        grid.addWidget(self.radioLogZ,              4, 6)

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

        self.connect(self.butCalibCXDataSet,     QtCore.SIGNAL('clicked()'),          self.processMenuForXDataSet )
        self.connect(self.butCalibCYDataSet,     QtCore.SIGNAL('clicked()'),          self.processMenuForYDataSet )
        self.connect(self.butCalibCXParName,     QtCore.SIGNAL('clicked()'),          self.processMenuForXParName )
        self.connect(self.butCalibCYParName,     QtCore.SIGNAL('clicked()'),          self.processMenuForYParName )
        self.connect(self.butCalibCXParIndex,    QtCore.SIGNAL('clicked()'),          self.processMenuForXParIndex )
        self.connect(self.butCalibCYParIndex,    QtCore.SIGNAL('clicked()'),          self.processMenuForYParIndex )

        self.connect(self.editCalibCycleYmin,    QtCore.SIGNAL('editingFinished ()'), self.processEditCalibCycleYmin )
        self.connect(self.editCalibCycleYmax,    QtCore.SIGNAL('editingFinished ()'), self.processEditCalibCycleYmax )
        self.connect(self.editCalibCycleXmin,    QtCore.SIGNAL('editingFinished ()'), self.processEditCalibCycleXmin )
        self.connect(self.editCalibCycleXmax,    QtCore.SIGNAL('editingFinished ()'), self.processEditCalibCycleXmax )

        self.connect(self.cboxYlimits,           QtCore.SIGNAL('stateChanged(int)'),  self.processCboxYlimits)
        self.connect(self.cboxXlimits,           QtCore.SIGNAL('stateChanged(int)'),  self.processCboxXlimits)

        self.connect(self.editCalibCycleYNBins,  QtCore.SIGNAL('editingFinished ()'), self.processEditCalibCycleYNBins )
        self.connect(self.editCalibCycleXNBins,  QtCore.SIGNAL('editingFinished ()'), self.processEditCalibCycleXNBins )
        self.connect(self.radioLogZ,             QtCore.SIGNAL('clicked()'),          self.processRadioLogZ )
        self.connect(self.radioLinZ,             QtCore.SIGNAL('clicked()'),          self.processRadioLinZ )
  
        cp.confpars.calibcycleWindowIsOpen = True

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
        self.butCalibCYDataSet .setToolTip('Select the dataset for the Y axis parameter.\n' +\
                                           'The list of datasets is formed from the checked items in the HDF5 tree GUI')
        self.butCalibCYParName .setToolTip('Select the parameter name in the dataset.\n' +\
                                           'The dataset should be selected first in the left box')


    def resizeEvent(self, e):
        self.frame.setGeometry(self.rect())


    def getVBoxForLayout(self):
        return self.vbox


    def setParentWidget(self,parent):
        self.parentWidget = parent


    def closeEvent(self, event):
        cp.confpars.calibcycleWindowIsOpen = False


    def processClose(self):
        self.close()


    def setButStatus(self):
        if   cp.confpars.calibcycleWindowParameters[self.window][2] == 0 :
            self.radioVsIndex.setChecked(True)
            self.processRadioVsIndex()

        elif cp.confpars.calibcycleWindowParameters[self.window][2] == 1 :
            self.radioVsTime.setChecked(True)
            self.processRadioVsTime()

        elif cp.confpars.calibcycleWindowParameters[self.window][2] == 2 :
            self.radioVsXPar.setChecked(True)
            self.processRadioVsXPar()

        elif cp.confpars.calibcycleWindowParameters[self.window][2] == 3 :
            self.radioYHist.setChecked(True)
            self.processRadioYHist()

        self.setButCalibCYDataSetTextAlignment()


    def processRadioLogZ(self): 
        cp.confpars.calibcycleWindowParameters[self.window][11] = True 
        #self.radioLogZ.setChecked(True)


    def processRadioLinZ(self): 
        cp.confpars.calibcycleWindowParameters[self.window][11] = False 
        #self.radioLinZ.setChecked(True)


    def processRadioVsIndex(self):
        cp.confpars.calibcycleWindowParameters[self.window][2] = 0
        dsname = 'Index'
        self.butCalibCXDataSet.setText(dsname)
        cp.confpars.calibcycleWindowParameters[self.window][1] = str(dsname)

        self.butCalibCXParName.setText('None')
        cp.confpars.calibcycleWindowParameters[self.window][8] = 'None'

        self.setButCalibCXDataSetTextAlignment()
        self.setEditFieldsStatusForMode()
        
            
    def processRadioVsTime(self):
        cp.confpars.calibcycleWindowParameters[self.window][2] = 1
        dsname = 'Time'
        self.butCalibCXDataSet.setText(dsname)
        cp.confpars.calibcycleWindowParameters[self.window][1] = str(dsname)

        self.butCalibCXParName.setText('None')
        cp.confpars.calibcycleWindowParameters[self.window][8] = 'None'

        self.setButCalibCXDataSetTextAlignment()
        self.setEditFieldsStatusForMode()
        

    def processRadioYHist(self):
        cp.confpars.calibcycleWindowParameters[self.window][2] = 3
        dsname = 'Is-not-used'
        self.butCalibCXDataSet.setText(dsname)
        cp.confpars.calibcycleWindowParameters[self.window][1] = str(dsname)

        self.butCalibCXParName.setText('None')
        cp.confpars.calibcycleWindowParameters[self.window][8] = 'None'

        self.setButCalibCXDataSetTextAlignment()
        self.setEditFieldsStatusForMode()  


    def processRadioVsXPar(self):
        cp.confpars.calibcycleWindowParameters[self.window][2] = 2

        dsname = cp.confpars.calibcycleWindowParameters[self.window][1]

        if     dsname == 'None' \
            or dsname == 'Time' \
            or dsname == 'Index' \
            or dsname == 'Is-not-used' :

            self.butCalibCXDataSet.setText('Select-X-parameter')
            self.butCalibCXParName.setText('None')
            cp.confpars.calibcycleWindowParameters[self.window][8] = 'None'

        else :
            self.butCalibCXDataSet.setText(dsname)

        self.setButCalibCXDataSetTextAlignment()
        #self.setButCalibCXParNameTextAlignment()
        self.setEditFieldsStatusForMode()


    def setZScaleRadioButtons(self):
            if cp.confpars.calibcycleWindowParameters[self.window][11] :
                self.radioLogZ.setChecked(True)
            else :
                self.radioLinZ.setChecked(True)

        
    def setEditFieldsStatusForMode(self):

        self.cboxXlimits.setCheckable(True)
        #self.processRadioLinZ(self)
        self.radioLogZ.setCheckable(False)
  
        if   cp.confpars.calibcycleWindowParameters[self.window][2] == 0 :
            self.editCalibCycleYNBins.setStyleSheet(self.styleSheetGrey)
            self.editCalibCycleXNBins.setStyleSheet(self.styleSheetGrey)
            self.editCalibCycleYNBins.setReadOnly(True)
            self.editCalibCycleXNBins.setReadOnly(True)

        elif cp.confpars.calibcycleWindowParameters[self.window][2] == 1 :
            self.editCalibCycleYNBins.setStyleSheet(self.styleSheetGrey)
            self.editCalibCycleXNBins.setStyleSheet(self.styleSheetGrey)
            self.editCalibCycleYNBins.setReadOnly(True)
            self.editCalibCycleXNBins.setReadOnly(True)

        elif cp.confpars.calibcycleWindowParameters[self.window][2] == 2 :
            self.editCalibCycleYNBins.setStyleSheet(self.styleSheetWhite)
            self.editCalibCycleXNBins.setStyleSheet(self.styleSheetWhite)
            self.editCalibCycleYNBins.setReadOnly(False)
            self.editCalibCycleXNBins.setReadOnly(False)
            self.radioLogZ.setCheckable(True)
            self.setZScaleRadioButtons()

        elif cp.confpars.calibcycleWindowParameters[self.window][2] == 3 :
            self.editCalibCycleYNBins.setStyleSheet(self.styleSheetWhite)
            self.editCalibCycleXNBins.setStyleSheet(self.styleSheetGrey)
            self.editCalibCycleYNBins.setReadOnly(False)
            self.editCalibCycleXNBins.setReadOnly(True)
            self.cboxXlimits.setCheckState(0)
            self.cboxXlimits.setCheckable(False)


    def setEditFieldsStatus(self):

        if self.cboxYlimits.isChecked() :
            self.editCalibCycleYmin.setStyleSheet(self.styleSheetWhite)
            self.editCalibCycleYmax.setStyleSheet(self.styleSheetWhite)
            self.editCalibCycleYmin.setReadOnly(False)
            self.editCalibCycleYmax.setReadOnly(False)
        else :
            self.editCalibCycleYmin.setStyleSheet(self.styleSheetGrey)
            self.editCalibCycleYmax.setStyleSheet(self.styleSheetGrey)
            self.editCalibCycleYmin.setReadOnly(True)
            self.editCalibCycleYmax.setReadOnly(True)

        if self.cboxXlimits.isChecked() :
            self.editCalibCycleXmin.setStyleSheet(self.styleSheetWhite)
            self.editCalibCycleXmax.setStyleSheet(self.styleSheetWhite)
            self.editCalibCycleXmin.setReadOnly(False)
            self.editCalibCycleXmax.setReadOnly(False)
        else :
            self.editCalibCycleXmin.setStyleSheet(self.styleSheetGrey)
            self.editCalibCycleXmax.setStyleSheet(self.styleSheetGrey)
            self.editCalibCycleXmin.setReadOnly(True)
            self.editCalibCycleXmax.setReadOnly(True)


    def processCboxYlimits(self):
        if self.cboxYlimits.isChecked():
            cp.confpars.calibcycleWindowParameters[self.window][9] = True
        else:
            cp.confpars.calibcycleWindowParameters[self.window][9] = False
        self.setEditFieldsStatus()


    def processCboxXlimits(self):
        if self.cboxXlimits.isChecked():
            cp.confpars.calibcycleWindowParameters[self.window][10] = True
        else:
            cp.confpars.calibcycleWindowParameters[self.window][10] = False
        self.setEditFieldsStatus()


    def processEditCalibCycleYmin(self):
        cp.confpars.calibcycleWindowParameters[self.window][3] = int(self.editCalibCycleYmin.displayText())        


    def processEditCalibCycleYmax(self):
        cp.confpars.calibcycleWindowParameters[self.window][4] = int(self.editCalibCycleYmax.displayText())        


    def processEditCalibCycleXmin(self):
        cp.confpars.calibcycleWindowParameters[self.window][5] = int(self.editCalibCycleXmin.displayText())        


    def processEditCalibCycleXmax(self):
        cp.confpars.calibcycleWindowParameters[self.window][6] = int(self.editCalibCycleXmax.displayText())        


    def processEditCalibCycleYNBins(self):
        cp.confpars.calibcycleWindowParameters[self.window][12] = int(self.editCalibCycleYNBins.displayText())


    def processEditCalibCycleXNBins(self):
        cp.confpars.calibcycleWindowParameters[self.window][13] = int(self.editCalibCycleXNBins.displayText())


    def setButCalibCYDataSetTextAlignment(self):
        if  self.butCalibCYDataSet.text() == 'None' :
            self.butCalibCYDataSet.setStyleSheet('Text-align:center')
            self.butCalibCYDataSet.setStyleSheet(self.styleSheetRed)
            self.butCalibCYParName.setText('None')
            self.butCalibCYParIndex.setText('None')
            cp.confpars.calibcycleWindowParameters[self.window][7] = 'None'
            cp.confpars.calibcycleWindowParameters[self.window][14] = 'None'
        else :
            self.butCalibCYDataSet.setStyleSheet('Text-align:right;' + self.styleSheetWhite)

        self.setButCalibCYParNameTextAlignment()


    def setButCalibCXDataSetTextAlignment(self):
        if     self.butCalibCXDataSet.text() == 'Time' \
            or self.butCalibCXDataSet.text() == 'Index' \
            or self.butCalibCXDataSet.text() == 'Is-not-used' :
            self.butCalibCXDataSet.setStyleSheet('Text-align:center;' + self.styleSheetWhite)
            self.butCalibCXParName.setStyleSheet(self.styleSheetWhite)
            self.butCalibCXParName.setText('None')
            self.butCalibCXParIndex.setText('None')

        elif   self.butCalibCXDataSet.text() == 'None' \
            or self.butCalibCXDataSet.text() == 'Select-X-parameter' :
            self.butCalibCXDataSet.setStyleSheet('Text-align:center;' + self.styleSheetRed)
            self.butCalibCXParName.setStyleSheet(self.styleSheetRed)
            self.butCalibCXParName.setText('None')
            self.butCalibCXParIndex.setText('None')
            cp.confpars.calibcycleWindowParameters[self.window][8] = 'None'
            cp.confpars.calibcycleWindowParameters[self.window][15] = 'None'
        else :
            self.butCalibCXDataSet.setStyleSheet('Text-align:right;' + self.styleSheetWhite)


    def setButCalibCYParNameTextAlignment(self):
        if self.butCalibCYParName.text() == 'None' :
            self.butCalibCYParName.setStyleSheet(self.styleSheetRed)
        else :
            self.butCalibCYParName.setStyleSheet(self.styleSheetWhite)


    def setButCalibCXParNameTextAlignment(self):
        if  self.butCalibCXParName.text() == 'None' :
            self.butCalibCXParName.setStyleSheet(self.styleSheetRed)
        else :
            self.butCalibCXParName.setStyleSheet(self.styleSheetWhite)


    def fillPopupMenuForDataSet(self):
        print 'fillPopupMenuForDataSet'
        self.popupMenuForDataSet.addAction('None')
        for dsname in cp.confpars.list_of_checked_item_names :

            item_last_name   = gm.get_item_last_name(dsname)           
            if item_last_name == 'waveforms'        : continue
            if item_last_name == 'image'            : continue
            if item_last_name == 'timestamps'       : continue
            if not gm.CalibCycleIsInThePath(dsname) : continue
            if gm.CSpadIsInTheName(dsname)          : continue
 
            self.popupMenuForDataSet.addAction(dsname)


    def processMenuForYDataSet(self):
        print 'MenuForYDataSet'
        actionSelected = self.popupMenuForDataSet.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_dsname = actionSelected.text()
        self.butCalibCYDataSet.setText( selected_dsname )
        cp.confpars.calibcycleWindowParameters[self.window][0] = str(selected_dsname)
        self.butCalibCYParName.setText('None')
        self.butCalibCYParIndex.setText('None')
        cp.confpars.calibcycleWindowParameters[self.window][7] = 'None'
        cp.confpars.calibcycleWindowParameters[self.window][14] = 'None'
        self.setButCalibCYDataSetTextAlignment()



    def processMenuForXDataSet(self):
        print 'MenuForXDataSet'
        if cp.confpars.calibcycleWindowParameters[self.window][2] < 2 : return # for Index and Time

        actionSelected = self.popupMenuForDataSet.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_dsname = actionSelected.text()
        self.butCalibCXDataSet.setText( selected_dsname )
        cp.confpars.calibcycleWindowParameters[self.window][1] = str(selected_dsname)

        self.setButCalibCXDataSetTextAlignment()


    def fillPopupMenuForYParName(self):
        print 'fillPopupMenuForYParName'
        dsname = cp.confpars.calibcycleWindowParameters[self.window][0]
        print 'dsname=', dsname
        self.listOfDatasetParNames = printh5.getListOfDatasetParNames(dsname)
        del self.popupMenuForYParName
        self.popupMenuForYParName=QtGui.QMenu()
        for parName in self.listOfDatasetParNames :
            self.popupMenuForYParName.addAction(parName)

  
    def fillPopupMenuForXParName(self):
        print 'fillPopupMenuForXParName'
        dsname = cp.confpars.calibcycleWindowParameters[self.window][1]
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
        self.butCalibCYParName.setText( selected_dsname )
        cp.confpars.calibcycleWindowParameters[self.window][7] = str(selected_dsname)
        self.butCalibCYParIndex.setText('None')
        cp.confpars.calibcycleWindowParameters[self.window][14] = 'None'

        self.setButCalibCYParNameTextAlignment()
        self.cboxYlimits.setCheckState(0)


    def processMenuForXParName(self):
        print 'MenuForXParName'
        if cp.confpars.calibcycleWindowParameters[self.window][2] < 2 : return # for Index and Time

        self.fillPopupMenuForXParName()
        actionSelected = self.popupMenuForXParName.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_dsname = actionSelected.text()
        selected_ind    = self.listOfDatasetParNames.index(selected_dsname)
        print 'selected_ind = ', selected_ind
        print 'selected_dsname = ', selected_dsname
        self.butCalibCXParName.setText( selected_dsname )
        cp.confpars.calibcycleWindowParameters[self.window][8]  = str(selected_dsname)
        self.butCalibCXParIndex.setText('None')
        cp.confpars.calibcycleWindowParameters[self.window][15] = 'None'

        self.setButCalibCXParNameTextAlignment()
        self.cboxXlimits.setCheckState(0)







    def fillPopupMenuForXParIndex(self):
        print 'fillPopupMenuForXParIndex'
        dsname  = cp.confpars.calibcycleWindowParameters[self.window][1]
        parname = cp.confpars.calibcycleWindowParameters[self.window][8]
        print 'dsname=', dsname, '   parname=', parname
        self.listOfDatasetParIndexes = printh5.getListOfDatasetParIndexes(dsname,parname)
        del self.popupMenuForXParIndex
        self.popupMenuForXParIndex=QtGui.QMenu()
        for parIndex in self.listOfDatasetParIndexes :
            self.popupMenuForXParIndex.addAction(parIndex)

    def fillPopupMenuForYParIndex(self):
        print 'fillPopupMenuForYParIndex'
        dsname  = cp.confpars.calibcycleWindowParameters[self.window][0]
        parname = cp.confpars.calibcycleWindowParameters[self.window][7]
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
        self.butCalibCYParIndex.setText( selected )
        cp.confpars.calibcycleWindowParameters[self.window][14] = str(selected)

    def processMenuForXParIndex(self):
        print 'MenuForXParIndex'
        self.fillPopupMenuForXParIndex()
        actionSelected = self.popupMenuForXParIndex.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected         = actionSelected.text()
        selected_ind     = self.listOfDatasetParIndexes.index(selected)
        print 'selected = ', selected
        self.butCalibCXParIndex.setText( selected )
        cp.confpars.calibcycleWindowParameters[self.window][15] = str(selected)

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUICalibCycleWindow()
    ex.show()
    app.exec_()
#-----------------------------

