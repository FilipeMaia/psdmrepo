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


        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

        titFont12 = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        titFont10 = QtGui.QFont("Sans Serif", 10, QtGui.QFont.Bold)

        self.titYminmax= QtGui.QLabel('Ymin, Ymax:')

        self.char_expand = u'\u25BE' # down-head triangle
        height = 18
        width  = 80

        self.editCorrelationYmin  = QtGui.QLineEdit(str(cp.confpars.correlationWindowParameters[self.window][3]))
        self.editCorrelationYmax  = QtGui.QLineEdit(str(cp.confpars.correlationWindowParameters[self.window][4]))

        #self.editCorrelationYmin  .setMaximumWidth(width)
        #self.editCorrelationYmax  .setMaximumWidth(width)

        self.editCorrelationYmin  .setMaximumHeight(height)
        self.editCorrelationYmax  .setMaximumHeight(height)

        self.editCorrelationYmin  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))
        self.editCorrelationYmax  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))

        self.titVs         = QtGui.QLabel('Versus:')
        self.radioVsIndex  = QtGui.QRadioButton('Index')
        self.radioVsTime   = QtGui.QRadioButton('Time' )
        self.radioVsXPar   = QtGui.QRadioButton('X-par')

        self.radioGroup    = QtGui.QButtonGroup()
        self.radioGroup.addButton(self.radioVsIndex)
        self.radioGroup.addButton(self.radioVsTime )
        self.radioGroup.addButton(self.radioVsXPar )

        self.titCorrXDataSet = QtGui.QLabel('X-par:')
        self.titCorrYDataSet = QtGui.QLabel('Y-par:')
        self.butCorrXDataSet = QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][1])
        self.butCorrYDataSet = QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][0])
        self.butCorrXParName = QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][8])
        self.butCorrYParName = QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][7])

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

        grid = QtGui.QGridLayout()

        grid.addWidget(self.titCorrYDataSet,     0, 0)
        grid.addWidget(self.butCorrYDataSet,     0, 1, 1, 4)
        grid.addWidget(self.butCorrYParName,     0, 5)

        grid.addWidget(self.titVs,               2, 0)
        grid.addWidget(self.radioVsIndex,        2, 1)
        grid.addWidget(self.radioVsTime,         2, 2)
        grid.addWidget(self.radioVsXPar,         2, 3)

        grid.addWidget(self.titCorrXDataSet,     3, 0)
        grid.addWidget(self.butCorrXDataSet,     3, 1, 1, 4)
        grid.addWidget(self.butCorrXParName,     3, 5)
        
        grid.addWidget(self.titYminmax,          4, 0, 1, 2)
        grid.addWidget(self.editCorrelationYmin, 4, 2)
        grid.addWidget(self.editCorrelationYmax, 4, 3)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(grid) 
        self.vbox.addStretch(1)     

        self.setButStatus()

        if parent == None :
            self.setLayout(self.vbox)
            self.show()

        self.connect(self.radioVsIndex,        QtCore.SIGNAL('clicked()'),          self.processRadioVsIndex )
        self.connect(self.radioVsTime,         QtCore.SIGNAL('clicked()'),          self.processRadioVsTime )
        self.connect(self.radioVsXPar,         QtCore.SIGNAL('clicked()'),          self.processRadioVsXPar )
        self.connect(self.editCorrelationYmin, QtCore.SIGNAL('editingFinished ()'), self.processEditCorrelationYmin )
        self.connect(self.editCorrelationYmax, QtCore.SIGNAL('editingFinished ()'), self.processEditCorrelationYmax )
        self.connect(self.butCorrXDataSet,     QtCore.SIGNAL('clicked()'),          self.processMenuForXDataSet )
        self.connect(self.butCorrYDataSet,     QtCore.SIGNAL('clicked()'),          self.processMenuForYDataSet )
        self.connect(self.butCorrXParName,     QtCore.SIGNAL('clicked()'),          self.processMenuForXParName )
        self.connect(self.butCorrYParName,     QtCore.SIGNAL('clicked()'),          self.processMenuForYParName )
  
        cp.confpars.selectionWindowIsOpen = True

        self.showToolTips()

    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters for waveforms.')
        self.radioVsIndex      .setToolTip('Select X axis parameter between Index/Time/Parameter')
        self.radioVsTime       .setToolTip('Select X axis parameter between Index/Time/Parameter')
        self.radioVsXPar       .setToolTip('Select X axis parameter between Index/Time/Parameter')
        #self.editSelectionYmin.setToolTip('This field can be edited for Manual control only.')
        #self.editSelectionYmax.setToolTip('This field can be edited for Manual control only.')


    def setEditFieldsReadOnly(self, isReadOnly=False):

        if isReadOnly == True : self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('grey'))
        else :                  self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('white'))

        self.editCorrelationYmin.setPalette(self.palette)
        self.editCorrelationYmax.setPalette(self.palette)

        self.editCorrelationYmin.setReadOnly(isReadOnly)
        self.editCorrelationYmax.setReadOnly(isReadOnly)


    def resizeEvent(self, e):
        self.frame.setGeometry(self.rect())


    def getVBoxForLayout(self):
        return self.vbox


    def setParentWidget(self,parent):
        self.parentWidget = parent


    def closeEvent(self, event):
        self.processClose()


    def processClose(self):
        cp.confpars.correlationWindowIsOpen = False
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

        self.setButCorrYDataSetTextAlignment()


    def processRadioVsIndex(self):
        cp.confpars.correlationWindowParameters[self.window][2] = 0
        dsname = 'Index'
        self.butCorrXDataSet.setText(dsname)
        cp.confpars.correlationWindowParameters[self.window][1] = str(dsname)

        self.butCorrXParName.setText('None')
        cp.confpars.correlationWindowParameters[self.window][8] = 'None'

        self.setButCorrXDataSetTextAlignment()
  
            
    def processRadioVsTime(self):
        cp.confpars.correlationWindowParameters[self.window][2] = 1
        dsname = 'Time'
        self.butCorrXDataSet.setText(dsname)
        cp.confpars.correlationWindowParameters[self.window][1] = str(dsname)

        self.butCorrXParName.setText('None')
        cp.confpars.correlationWindowParameters[self.window][8] = 'None'

        self.setButCorrXDataSetTextAlignment()


    def processRadioVsXPar(self):
        cp.confpars.correlationWindowParameters[self.window][2] = 2

        dsname = cp.confpars.correlationWindowParameters[self.window][1]

        if     dsname == 'None' \
            or dsname == 'Time' \
            or dsname == 'Index' :

            self.butCorrXDataSet.setText('Select X parameter')
            self.butCorrXParName.setText('None')
            cp.confpars.correlationWindowParameters[self.window][8] = 'None'

        else :
            self.butCorrXDataSet.setText(dsname)

        self.setButCorrXDataSetTextAlignment()
        #self.setButCorrXParNameTextAlignment()


    def processEditCorrelationYmin(self):
        cp.confpars.correlationWindowParameters[self.window][3] = int(self.editCorrelationYmin.displayText())        


    def processEditCorrelationYmax(self):
        cp.confpars.correlationWindowParameters[self.window][4] = int(self.editCorrelationYmax.displayText())        


    def setButCorrYDataSetTextAlignment(self):
        if  self.butCorrYDataSet.text() == 'None' :
            self.butCorrYDataSet.setStyleSheet('Text-align:center')
            self.butCorrYDataSet.setStyleSheet(self.styleSheetRed)
            self.butCorrYParName.setText('None')
            cp.confpars.correlationWindowParameters[self.window][7] = 'None'
        else :
            self.butCorrYDataSet.setStyleSheet('Text-align:right;' + self.styleSheetWhite)

        self.setButCorrYParNameTextAlignment()


    def setButCorrXDataSetTextAlignment(self):
        if     self.butCorrXDataSet.text() == 'Time' \
            or self.butCorrXDataSet.text() == 'Index' :
            self.butCorrXDataSet.setStyleSheet('Text-align:center;' + self.styleSheetWhite)
            self.butCorrXParName.setStyleSheet(self.styleSheetWhite)
            self.butCorrXParName.setText('None')

        elif   self.butCorrXDataSet.text() == 'None' \
            or self.butCorrXDataSet.text() == 'Select X parameter' :
            self.butCorrXDataSet.setStyleSheet('Text-align:center;' + self.styleSheetRed)
            self.butCorrXParName.setStyleSheet(self.styleSheetRed)
            self.butCorrXParName.setText('None')
            cp.confpars.correlationWindowParameters[self.window][8] = 'None'
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
            self.popupMenuForDataSet.addAction(dsname)


    def processMenuForYDataSet(self):
        print 'MenuForYDataSet'
        actionSelected = self.popupMenuForDataSet.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_dsname = actionSelected.text()
        self.butCorrYDataSet.setText( selected_dsname )
        cp.confpars.correlationWindowParameters[self.window][0] = str(selected_dsname)
        self.butCorrYParName.setText('None')
        cp.confpars.correlationWindowParameters[self.window][7] = 'None'
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
        cp.confpars.correlationWindowParameters[self.window][9] = selected_ind

        self.setButCorrYParNameTextAlignment()

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
        cp.confpars.correlationWindowParameters[self.window][10] = selected_ind


        self.setButCorrXParNameTextAlignment()

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUICorrelationWindow()
    ex.show()
    app.exec_()
#-----------------------------

