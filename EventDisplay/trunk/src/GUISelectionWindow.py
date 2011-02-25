#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISelectionWindow...
#
#------------------------------------------------------------------------

"""GUI manipulates with parameters for event selection in particular window of the image.

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
class GUISelectionWindow ( QtGui.QWidget ) :
    """GUI manipulates with parameters for event selection in particular window of the image."""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, window=0):
        QtGui.QWidget.__init__(self, parent)

        print 'GUISelectionWindow for region', window

        self.window = window

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Adjust selection parameters')

        self.palette = QtGui.QPalette()

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

        titFont12 = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        titFont10 = QtGui.QFont("Sans Serif", 10, QtGui.QFont.Bold)

        self.titXminmax= QtGui.QLabel('Xmin, Xmax:')
        self.titYminmax= QtGui.QLabel('Ymin, Ymax:')

        self.char_expand = u'\u25BE' # down-head triangle
        height = 22
        width  = 50

        self.editSelectionThr   = QtGui.QLineEdit(str(cp.confpars.selectionWindowParameters[self.window][0]))
        self.editSelectionXmin  = QtGui.QLineEdit(str(cp.confpars.selectionWindowParameters[self.window][2]))
        self.editSelectionXmax  = QtGui.QLineEdit(str(cp.confpars.selectionWindowParameters[self.window][3]))
        self.editSelectionYmin  = QtGui.QLineEdit(str(cp.confpars.selectionWindowParameters[self.window][4]))
        self.editSelectionYmax  = QtGui.QLineEdit(str(cp.confpars.selectionWindowParameters[self.window][5]))

        self.editSelectionThr   .setMaximumWidth(width)
        self.editSelectionXmin  .setMaximumWidth(width)
        self.editSelectionXmax  .setMaximumWidth(width)
        self.editSelectionYmin  .setMaximumWidth(width)
        self.editSelectionYmax  .setMaximumWidth(width)

        self.editSelectionThr   .setMaximumHeight(45)
        self.editSelectionXmin  .setMaximumHeight(height)
        self.editSelectionXmax  .setMaximumHeight(height)
        self.editSelectionYmin  .setMaximumHeight(height)
        self.editSelectionYmax  .setMaximumHeight(height)

        self.editSelectionThr   .setValidator(QtGui.QIntValidator(0,30000,self))
        self.editSelectionXmin  .setValidator(QtGui.QIntValidator(0, 2000,self))
        self.editSelectionXmax  .setValidator(QtGui.QIntValidator(0, 2000,self))
        self.editSelectionYmin  .setValidator(QtGui.QIntValidator(0, 2000,self))
        self.editSelectionYmax  .setValidator(QtGui.QIntValidator(0, 2000,self))

        self.titThreshold  = QtGui.QLabel('Threshold on min intensity (ADU) ')
        self.radioInWin    = QtGui.QRadioButton("in window")
        self.radioInBin    = QtGui.QRadioButton("in bin")
        self.radioGroup    = QtGui.QButtonGroup()
        self.radioGroup.addButton(self.radioInWin)
        self.radioGroup.addButton(self.radioInBin)

        if cp.confpars.selectionWindowParameters[self.window][1] : self.radioInBin.setChecked(True)
        else :                                                     self.radioInWin.setChecked(True)

        grid = QtGui.QGridLayout()
        grid.addWidget(self.titThreshold,        0, 0, 2, 2)
        grid.addWidget(self.editSelectionThr,    0, 2, 2, 1)
        grid.addWidget(self.radioInBin,          0, 3)
        grid.addWidget(self.radioInWin,          1, 3)

        grid.addWidget(self.titXminmax,          3, 0)
        grid.addWidget(self.editSelectionXmin,   3, 1)
        grid.addWidget(self.editSelectionXmax,   3, 2)

        grid.addWidget(self.titYminmax,          4, 0)
        grid.addWidget(self.editSelectionYmin,   4, 1)
        grid.addWidget(self.editSelectionYmax,   4, 2)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(grid) 
        self.vbox.addStretch(1)     

        if parent == None :
            self.setLayout(self.vbox)
            self.show()

        self.connect(self.radioInBin,         QtCore.SIGNAL('clicked()'),          self.processRadioInBin )
        self.connect(self.radioInWin,         QtCore.SIGNAL('clicked()'),          self.processRadioInWin )
        self.connect(self.editSelectionThr,   QtCore.SIGNAL('editingFinished ()'), self.processEditSelectionThr  )
        self.connect(self.editSelectionXmin,  QtCore.SIGNAL('editingFinished ()'), self.processEditSelectionXmin )
        self.connect(self.editSelectionXmax,  QtCore.SIGNAL('editingFinished ()'), self.processEditSelectionXmax )
        self.connect(self.editSelectionYmin,  QtCore.SIGNAL('editingFinished ()'), self.processEditSelectionYmin )
        self.connect(self.editSelectionYmax,  QtCore.SIGNAL('editingFinished ()'), self.processEditSelectionYmax )
 
        cp.confpars.selectionWindowIsOpen = True

        self.showToolTips()

    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters for waveforms.')
        self.radioInBin       .setToolTip('Select between threshold in bin or in entire window.')
        self.radioInWin       .setToolTip('Select between threshold in bin or in entire window.')
        #self.editSelectionXmin.setToolTip('This field can be edited for Manual control only.')
        #self.editSelectionXmax.setToolTip('This field can be edited for Manual control only.')
        #self.editSelectionYmin.setToolTip('This field can be edited for Manual control only.')
        #self.editSelectionYmax.setToolTip('This field can be edited for Manual control only.')


    def setEditFieldsReadOnly(self, isReadOnly=False):

        if isReadOnly == True : self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('grey'))
        else :                  self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('white'))

        self.editSelectionXmin.setPalette(self.palette)
        self.editSelectionXmax.setPalette(self.palette)
        self.editSelectionYmin.setPalette(self.palette)
        self.editSelectionYmax.setPalette(self.palette)

        self.editSelectionXmin.setReadOnly(isReadOnly)
        self.editSelectionXmax.setReadOnly(isReadOnly)
        self.editSelectionYmin.setReadOnly(isReadOnly)
        self.editSelectionYmax.setReadOnly(isReadOnly)


    def resizeEvent(self, e):
        self.frame.setGeometry(self.rect())


    def getVBoxForLayout(self):
        return self.vbox


    def setParentWidget(self,parent):
        self.parentWidget = parent


    def closeEvent(self, event):
        self.processClose()


    def processClose(self):
        cp.confpars.selectionWindowIsOpen = False
        self.close()


    def processRadioInBin(self):
        cp.confpars.selectionWindowParameters[self.window][1] = True


    def processRadioInWin(self):
        cp.confpars.selectionWindowParameters[self.window][1] = False


    def processEditSelectionThr(self):
        cp.confpars.selectionWindowParameters[self.window][0] = int(self.editSelectionThr.displayText())        


    def processEditSelectionXmin(self):
        cp.confpars.selectionWindowParameters[self.window][2] = int(self.editSelectionXmin.displayText())        


    def processEditSelectionXmax(self):
        cp.confpars.selectionWindowParameters[self.window][3] = int(self.editSelectionXmax.displayText())        


    def processEditSelectionYmin(self):
        cp.confpars.selectionWindowParameters[self.window][4] = int(self.editSelectionYmin.displayText())        


    def processEditSelectionYmax(self):
        cp.confpars.selectionWindowParameters[self.window][5] = int(self.editSelectionYmax.displayText())        

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUISelectionWindow()
    ex.show()
    app.exec_()
#-----------------------------

