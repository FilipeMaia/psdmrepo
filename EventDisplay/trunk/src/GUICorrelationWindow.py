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
        width  = 100

        self.editCorrelationYmin  = QtGui.QLineEdit(str(cp.confpars.correlationWindowParameters[self.window][3]))
        self.editCorrelationYmax  = QtGui.QLineEdit(str(cp.confpars.correlationWindowParameters[self.window][4]))

        self.editCorrelationYmin  .setMaximumWidth(width)
        self.editCorrelationYmax  .setMaximumWidth(width)

        self.editCorrelationYmin  .setMaximumHeight(height)
        self.editCorrelationYmax  .setMaximumHeight(height)

        self.editCorrelationYmin  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))
        self.editCorrelationYmax  .setValidator(QtGui.QIntValidator(-1000000, 1000000, self))

        self.titVs         = QtGui.QLabel('Versus:')
        self.radioVsIndex  = QtGui.QRadioButton("index")
        self.radioVsTime   = QtGui.QRadioButton("time" )
        self.radioVsXPar   = QtGui.QRadioButton("X-par")

        self.radioGroup    = QtGui.QButtonGroup()
        self.radioGroup.addButton(self.radioVsIndex)
        self.radioGroup.addButton(self.radioVsTime )
        self.radioGroup.addButton(self.radioVsXPar )

        if   cp.confpars.correlationWindowParameters[self.window][2] == 0 :
            self.radioVsIndex.setChecked(True)
        elif cp.confpars.correlationWindowParameters[self.window][2] == 1 :
            self.radioVsTime.setChecked(True)
        elif cp.confpars.correlationWindowParameters[self.window][2] == 2 :
            self.radioVsXPar.setChecked(True)

        self.titCorrYDataSet = QtGui.QLabel('Y-par:')
        self.butCorrYDataSet = QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][0])
        self.butCorrYDataSet.setMaximumWidth(350)

        self.titCorrXDataSet = QtGui.QLabel('X-par:')
        self.butCorrXDataSet = QtGui.QPushButton(cp.confpars.correlationWindowParameters[self.window][1])
        self.butCorrXDataSet.setMaximumWidth(350)

        self.setButCorrXDataSetTextAlignment()
        self.setButCorrYDataSetTextAlignment()

        self.popupMenuForDataSet = QtGui.QMenu()
        self.fillPopupMenuForDataSet()

        grid = QtGui.QGridLayout()

        grid.addWidget(self.titCorrYDataSet,     0, 0)
        grid.addWidget(self.butCorrYDataSet,     0, 1, 1, 5)

        grid.addWidget(self.titVs,               2, 0)
        grid.addWidget(self.radioVsIndex,        2, 1)
        grid.addWidget(self.radioVsTime,         2, 2)
        grid.addWidget(self.radioVsXPar,         2, 3)

        grid.addWidget(self.titCorrXDataSet,     3, 0)
        grid.addWidget(self.butCorrXDataSet,     3, 1, 1, 5)

        grid.addWidget(self.titYminmax,          4, 0)
        grid.addWidget(self.editCorrelationYmin, 4, 1)
        grid.addWidget(self.editCorrelationYmax, 4, 2)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(grid) 
        self.vbox.addStretch(1)     

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


    def processRadioVsIndex(self):
        cp.confpars.correlationWindowParameters[self.window][2] = 0
        dsname = 'Index'
        self.butCorrXDataSet.setText(dsname)
        cp.confpars.correlationWindowParameters[self.window][1] = str(dsname)


    def processRadioVsTime(self):
        cp.confpars.correlationWindowParameters[self.window][2] = 1
        dsname = 'Time'
        self.butCorrXDataSet.setText(dsname)
        cp.confpars.correlationWindowParameters[self.window][1] = str(dsname)


    def processRadioVsXPar(self):
        cp.confpars.correlationWindowParameters[self.window][2] = 2
        dsname = 'Select X parameter'
        self.butCorrXDataSet.setText(dsname)
        cp.confpars.correlationWindowParameters[self.window][1] = str(dsname)


    def processEditCorrelationYmin(self):
        cp.confpars.correlationWindowParameters[self.window][3] = int(self.editCorrelationYmin.displayText())        


    def processEditCorrelationYmax(self):
        cp.confpars.correlationWindowParameters[self.window][4] = int(self.editCorrelationYmax.displayText())        


    def setButCorrYDataSetTextAlignment(self):
        if  self.butCorrYDataSet.text() == 'None' :
            self.butCorrYDataSet.setStyleSheet('Text-align:center')
        else :
            self.butCorrYDataSet.setStyleSheet('Text-align:right')

    def setButCorrXDataSetTextAlignment(self):
        if  self.butCorrXDataSet.text() == 'None' :
            self.butCorrXDataSet.setStyleSheet('Text-align:center')
        else :
            self.butCorrXDataSet.setStyleSheet('Text-align:right')


    def fillPopupMenuForDataSet(self):
        print 'fillPopupMenuForDataSet'
        self.popupMenuForDataSet.addAction('None')
        for dsname in cp.confpars.list_of_checked_item_names :
            self.popupMenuForDataSet.addAction(dsname)


    def processMenuForYDataSet(self):
        print 'MenuForYDataSet'
        actionSelected = self.popupMenuForDataSet.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_ds = actionSelected.text()
        self.butCorrYDataSet.setText( selected_ds )
        self.setButCorrYDataSetTextAlignment()
        cp.confpars.correlationWindowParameters[self.window][0] = str(selected_ds)


    def processMenuForXDataSet(self):
        print 'MenuForXDataSet'
        actionSelected = self.popupMenuForDataSet.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_ds = actionSelected.text()
        self.butCorrXDataSet.setText( selected_ds )
        self.setButCorrXDataSetTextAlignment()
        cp.confpars.correlationWindowParameters[self.window][1] = str(selected_ds)

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUICorrelationWindow()
    ex.show()
    app.exec_()
#-----------------------------

