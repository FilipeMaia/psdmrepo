#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgSpeButtons...
#
#------------------------------------------------------------------------

"""Plots for any 'image' record in the EventeDisplay project.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule
@version $Id: 
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
import os

from PyQt4 import QtGui, QtCore

from Logger                 import logger
from FileNameManager        import fnm

#---------------------
#  Class definition --
#---------------------

#class ImgSpeButtons (QtGui.QMainWindow) :
class ImgSpeButtons (QtGui.QWidget) :
    """A set of buttons for figure control."""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, widgimage=None):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('GUI of buttons')

        self.setFrame()
        self.widgimage = widgimage
        self.parent    = parent
        self.fig       = widgimage.fig

        self.styleSheetGrey  = "background-color: rgb(100, 100, 100); color: rgb(0, 0, 0)"
        self.styleSheetWhite = "background-color: rgb(230, 230, 230); color: rgb(0, 0, 0)"

        self.myXmin = None
        self.myXmax = None
        self.myYmin = None
        self.myYmax = None
        self.myZmin = None
        self.myZmax = None
        self.myNBins = 100
        self.myZoomIsOn = False
        self.myGridIsOn = False
        self.myLogIsOn  = False

        self.but_draw  = QtGui.QPushButton('&Draw')
        self.but_save  = QtGui.QPushButton('&Save')
        self.but_quit  = QtGui.QPushButton('&Quit')
        self.cbox_grid = QtGui.QCheckBox('&Grid')
        self.cbox_grid.setChecked(self.myGridIsOn)
        self.cbox_log  = QtGui.QCheckBox('&Log')
        self.cbox_log.setChecked(self.myLogIsOn)

        #self.cboxXIsOn = QtGui.QCheckBox("X min, max:")
        #self.cboxYIsOn = QtGui.QCheckBox("Y min, max:")
        self.cboxZIsOn = QtGui.QCheckBox("&A min, max:")

        #self.editXmin  = QtGui.QLineEdit(self.stringOrNone(self.myXmin))
        #self.editXmax  = QtGui.QLineEdit(self.stringOrNone(self.myXmax))
        #self.editYmin  = QtGui.QLineEdit(self.stringOrNone(self.myYmin))
        #self.editYmax  = QtGui.QLineEdit(self.stringOrNone(self.myYmax))
        self.editZmin  = QtGui.QLineEdit(self.stringOrNone(self.myZmin))
        self.editZmax  = QtGui.QLineEdit(self.stringOrNone(self.myZmax))
        self.editNBins = QtGui.QLineEdit(self.stringOrNone(self.myNBins))

        width = 60
        #self.editXmin.setMaximumWidth(width)
        #self.editXmax.setMaximumWidth(width)
        #self.editYmin.setMaximumWidth(width)
        #self.editYmax.setMaximumWidth(width)
        self.editZmin .setMaximumWidth(width)
        self.editZmax .setMaximumWidth(width)
        self.editNBins.setMaximumWidth(width)

        #self.editXmin.setValidator(QtGui.QIntValidator(0,100000,self))
        #self.editXmax.setValidator(QtGui.QIntValidator(0,100000,self)) 
        #self.editYmin.setValidator(QtGui.QIntValidator(0,100000,self))
        #self.editYmax.setValidator(QtGui.QIntValidator(0,100000,self)) 
        self.editZmin.setValidator(QtGui.QIntValidator(-100000,100000,self))
        self.editZmax.setValidator(QtGui.QIntValidator(-100000,100000,self))
        self.editNBins.setValidator(QtGui.QIntValidator(1,1000,self))

 
        self.connect(self.but_draw,  QtCore.SIGNAL('clicked()'),         self.on_but_draw)
        self.connect(self.but_save,  QtCore.SIGNAL('clicked()'),         self.on_but_save)
        self.connect(self.but_quit,  QtCore.SIGNAL('clicked()'),         self.on_but_quit)
        self.connect(self.cbox_grid, QtCore.SIGNAL('stateChanged(int)'), self.on_cbox_grid)
        self.connect(self.cbox_log,  QtCore.SIGNAL('stateChanged(int)'), self.on_cbox_log)
        self.connect(self.cboxZIsOn, QtCore.SIGNAL('stateChanged(int)'), self.on_cbox_z)
        self.connect(self.editZmin,  QtCore.SIGNAL('editingFinished ()'), self.on_edit_zmin)
        self.connect(self.editZmax,  QtCore.SIGNAL('editingFinished ()'), self.on_edit_zmax)
        self.connect(self.editNBins, QtCore.SIGNAL('editingFinished ()'), self.on_edit_nbins)

        #self.connect(self.cboxXIsOn, QtCore.SIGNAL('stateChanged(int)'), self.processCBoxes)
        #self.connect(self.cboxYIsOn, QtCore.SIGNAL('stateChanged(int)'), self.processCBoxes)
        #self.connect(self.editXmin, QtCore.SIGNAL('editingFinished ()'), self.processEditXmin)
        #self.connect(self.editXmax, QtCore.SIGNAL('editingFinished ()'), self.processEditXmax)
        #self.connect(self.editYmin, QtCore.SIGNAL('editingFinished ()'), self.processEditYmin)
        #self.connect(self.editYmax, QtCore.SIGNAL('editingFinished ()'), self.processEditYmax)

        # Layout with box sizers
        # 
        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.cboxZIsOn, 0, 1)
        self.grid.addWidget(self.editZmin,  0, 2)
        self.grid.addWidget(self.editZmax,  0, 3)
        self.grid.addWidget(self.editNBins, 0, 4)
        self.grid.addWidget(self.cbox_grid, 0, 5)
        self.grid.addWidget(self.cbox_log,  0, 6)
        self.grid.addWidget(self.but_draw,  0, 7)
        self.grid.addWidget(self.but_save,  0, 8)
        self.grid.addWidget(self.but_quit,  0, 9)

        #hboxX.addWidget(self.cboxXIsOn)
        #hboxX.addWidget(self.editXmin)
        #hboxX.addWidget(self.editXmax)
        #hboxY.addWidget(self.cboxYIsOn)
        #hboxY.addWidget(self.editYmin)
        #hboxY.addWidget(self.editYmax)
        #hboxZ.addWidget(self.cboxZIsOn)
        #hboxZ.addWidget(self.editZmin)
        #hboxZ.addWidget(self.editZmax)

        self.setLayout(self.grid)

        self.setEditFieldValues()


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def closeEvent(self, event): # is called for self.close() or when click on "x"
        #print 'Close application'
        self.parent.close()
           
    def on_but_quit(self):
        logger.debug('on_but_quit', __name__ )
        self.close()
     
    def setEditFieldValues(self) :
        #self.editXmin.setText( str(self.intOrNone(self.myXmin)) )
        #self.editXmax.setText( str(self.intOrNone(self.myXmax)) )

        #self.editYmin.setText( str(self.intOrNone(self.myYmin)) )
        #self.editYmax.setText( str(self.intOrNone(self.myYmax)) ) 

        self.editZmin.setText( str(self.intOrNone(self.myZmin)) )
        self.editZmax.setText( str(self.intOrNone(self.myZmax)) )

        self.setEditFieldColors()

       
    def setEditFieldColors(self) :
        
        #if self.cboxXIsOn.isChecked(): self.styleSheet = self.styleSheetWhite
        #else                         : self.styleSheet = self.styleSheetGrey
        #self.editXmin.setStyleSheet('Text-align:left;' + self.styleSheet)
        #self.editXmax.setStyleSheet('Text-align:left;' + self.styleSheet)

        #if self.cboxYIsOn.isChecked(): self.styleSheet = self.styleSheetWhite
        #else                         : self.styleSheet = self.styleSheetGrey
        #self.editYmin.setStyleSheet('Text-align:left;' + self.styleSheet)
        #self.editYmax.setStyleSheet('Text-align:left;' + self.styleSheet)

        if self.cboxZIsOn.isChecked():
            self.styleSheet = self.styleSheetWhite
            self.fig.myZmin = self.myZmin
            self.fig.myZmax = self.myZmax
        else :
            self.styleSheet = self.styleSheetGrey
            self.fig.myZmin = None
            self.fig.myZmax = None

        self.editZmin.setText(str(self.fig.myZmin))
        self.editZmax.setText(str(self.fig.myZmax))
            
        self.editZmin.setStyleSheet('Text-align:left;' + self.styleSheet)
        self.editZmax.setStyleSheet('Text-align:left;' + self.styleSheet)

        #self.editXmin.setReadOnly( not self.cboxXIsOn.isChecked() )
        #self.editXmax.setReadOnly( not self.cboxXIsOn.isChecked() )

        #self.editYmin.setReadOnly( not self.cboxYIsOn.isChecked() )
        #self.editYmax.setReadOnly( not self.cboxYIsOn.isChecked() )

        self.editZmin.setReadOnly( not self.cboxZIsOn.isChecked() )
        self.editZmax.setReadOnly( not self.cboxZIsOn.isChecked() )


    def on_cbox_z(self):
        self.setEditFieldColors()


    def stringOrNone(self,value):
        if value == None : return 'None'
        else             : return str(value)


    def intOrNone(self,value):
        if value == None : return None
        else             : return int(value)


    def on_edit_zmin(self):
        self.fig.myZmin = self.myZmin = self.editZmin.displayText()


    def on_edit_zmax(self):
        self.fig.myZmax = self.myZmax = self.editZmax.displayText()


    def on_edit_nbins(self):
        self.fig.myNBins = int(self.editNBins.displayText())
        logger.info('Set for spectrum the number of bins ='+str(self.fig.myNBins), __name__ )
        self.widgimage.processDraw()
 

    def on_but_draw(self):
        logger.debug('on_but_draw', __name__ )
        self.widgimage.processDraw()


    def on_but_save(self):
        logger.debug('on_but_save', __name__ )
        path = fnm.path_pedestals_plot()
        #dir, fname = os.path.split(path)
        path  = str( QtGui.QFileDialog.getSaveFileName(self,
                                                       caption='Select file to save the plot',
                                                       directory = path,
                                                       filter = '*.png, *.eps, *pdf, *.ps'
                                                       ) )
        #dname, fname = os.path.split(path)
        if path == '' :
            logger.debug('Saving is cancelled.', __name__ )
            return
        logger.info('Save plot in file: ' + path, __name__ )
        self.widgimage.saveFigure(path)


    def on_cbox_log(self):
        logger.info('Not implemented yet.', __name__ )


    def on_cbox_grid(self):
        logger.info('On/Off grid.', __name__ )
        self.fig.myGridIsOn = self.cbox_grid.isChecked()
        self.widgimage.processDraw()
        
#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    w = ImgSpeButtons(None)
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
