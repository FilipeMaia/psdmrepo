#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotImgSpeButtons...
#
#------------------------------------------------------------------------

"""Buttons for interactive plot of the image and spectrum for 2-d array

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
from GUIHelp                import *
from GUIELogPostingDialog   import *

#from FileNameManager        import fnm
#from ConfigParametersCorAna import confpars as cp

#---------------------
#  Class definition --
#---------------------

#class PlotImgSpeButtons (QtGui.QMainWindow) :
class PlotImgSpeButtons (QtGui.QWidget) :
    """Buttons for interactive plot of the image and spectrum for 2-d array."""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, widgimage=None, ofname='./fig.png'):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('GUI of buttons')

        self.setFrame()

        self.parent    = parent
        self.ofname    = ofname

        self.widgimage = widgimage
        self.fig       = widgimage.fig

        self.but_reset = QtGui.QPushButton('&Reset')
        self.but_help  = QtGui.QPushButton('&Help')
        self.but_save  = QtGui.QPushButton('&Save')
        self.but_elog  = QtGui.QPushButton(u'\u2192 &ELog')
        self.but_quit  = QtGui.QPushButton('&Close')
        self.cbox_grid = QtGui.QCheckBox('&Grid')
        self.cbox_log  = QtGui.QCheckBox('&Log')
        self.tit_nbins = QtGui.QLabel('N bins:')
        self.edi_nbins = QtGui.QLineEdit(self.stringOrNone(self.fig.myNBins))
        self.set_buttons()
        
        width = 60
        self.edi_nbins.setFixedWidth(width)
        self.but_reset.setFixedWidth(width)
        self.but_help .setFixedWidth(width)
        self.but_save .setFixedWidth(width)
        self.but_elog .setFixedWidth(width)
        self.but_quit .setFixedWidth(width)
        self.edi_nbins.setValidator(QtGui.QIntValidator(1,1000,self))
 
        self.but_help.setStyleSheet (cp.styleButtonGood) 
        self.but_save.setStyleSheet (cp.styleButtonGood) 
        self.but_quit.setStyleSheet (cp.styleButtonBad) 

        self.connect(self.but_help,  QtCore.SIGNAL('clicked()'),          self.on_but_help)
        self.connect(self.but_reset, QtCore.SIGNAL('clicked()'),          self.on_but_reset)
        self.connect(self.but_save,  QtCore.SIGNAL('clicked()'),          self.on_but_save)
        self.connect(self.but_elog,  QtCore.SIGNAL('clicked()'),          self.on_but_elog)
        self.connect(self.but_quit,  QtCore.SIGNAL('clicked()'),          self.on_but_quit)
        self.connect(self.cbox_grid, QtCore.SIGNAL('stateChanged(int)'),  self.on_cbox_grid)
        self.connect(self.cbox_log,  QtCore.SIGNAL('stateChanged(int)'),  self.on_cbox_log)
        self.connect(self.edi_nbins, QtCore.SIGNAL('editingFinished ()'), self.on_edit_nbins)

        #self.setGridLayout()        
        self.setHBoxLayout()        
        self.showToolTips()


    def setHBoxLayout(self):
        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.but_help)
        self.hbox.addWidget(self.tit_nbins)
        self.hbox.addWidget(self.edi_nbins)
        self.hbox.addWidget(self.cbox_grid)
        self.hbox.addWidget(self.cbox_log)
        self.hbox.addWidget(self.but_reset)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_save)
        self.hbox.addWidget(self.but_elog)
        self.hbox.addWidget(self.but_quit)
        self.setLayout(self.hbox)


    def setGridLayout(self):
        self.grid = QtGui.QGridLayout() 
        self.grid.addWidget(self.but_help,  0, 0)
        self.grid.addWidget(self.tit_nbins, 0, 1)
        self.grid.addWidget(self.edi_nbins, 0, 2)
        self.grid.addWidget(self.cbox_grid, 0, 3)
        self.grid.addWidget(self.cbox_log,  0, 4)
        self.grid.addWidget(self.but_reset, 0, 6)
        self.grid.addWidget(self.but_save,  0, 7)
        self.grid.addWidget(self.but_quit,  0, 8)
        self.setLayout(self.grid)


    def showToolTips(self):
        self.but_reset.setToolTip('Reset original view') 
        self.but_quit .setToolTip('Quit this GUI') 
        self.but_save .setToolTip('Save the figure in file') 
        self.but_elog .setToolTip('Send figure to ELog') 
        self.but_help .setToolTip('Click on this button\nand get help') 
        self.cbox_grid.setToolTip('On/Off grid') 
        self.cbox_log .setToolTip('Log/Linear scale') 
        self.edi_nbins.setToolTip('Edit the number of bins\nfor spectrum [1-1000]')


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
        try    : self.guihelp.close()
        except : pass

        try    : self.parent.close()
        except : pass
           

    def on_but_quit(self):
        logger.debug('on_but_quit', __name__ )
        self.close()


    def stringOrNone(self,value):
        if value == None : return 'None'
        else             : return str(value)


    def intOrNone(self,value):
        if value == None : return None
        else             : return int(value)

    def set_buttons(self) :
        self.cbox_grid.setChecked(self.fig.myGridIsOn)
        self.cbox_log .setChecked(self.fig.myLogIsOn)
        self.edi_nbins.setText(self.stringOrNone(self.fig.myNBins))


    def on_edit_nbins(self):
        self.fig.myNBins = int(self.edi_nbins.displayText())
        logger.info('Set for spectrum the number of bins ='+str(self.fig.myNBins), __name__ )
        self.widgimage.processDraw()
 

    def on_but_reset(self):
        logger.debug('on_but_reset', __name__ )
        self.widgimage.initParameters()
        self.set_buttons()
        self.widgimage.on_draw()


    def on_but_save(self):
        logger.debug('on_but_save', __name__ )
        path = self.ofname
        #dir, fname = os.path.split(path)
        path  = str( QtGui.QFileDialog.getSaveFileName(self,
                                                       caption='Select file to save the plot',
                                                       directory = path,
                                                       filter = '*.png, *.eps, *pdf, *.ps'
                                                       ) )
        if path == '' :
            logger.debug('Saving is cancelled.', __name__ )
            return
        logger.info('Save plot in file: ' + path, __name__ )
        self.widgimage.saveFigure(path)


    def on_but_elog(self):
        logger.info('Send message to ELog:', __name__ )
        path = self.ofname
        logger.info('1. Save plot in file: ' + path, __name__ )
        self.widgimage.saveFigure(path)
        logger.info('2. Submit message with plot to ELog', __name__ )
        wdialog = GUIELogPostingDialog (self, fname=path)
        resp=wdialog.exec_()
         

    def on_cbox_log(self):
        logger.info('Set log10 scale', __name__ )
        self.fig.myLogIsOn = self.cbox_log.isChecked()
        self.fig.myZmin    = None
        self.fig.myZmax    = None        
        self.widgimage.processDraw()


    def on_cbox_grid(self):
        logger.info('On/Off grid.', __name__ )
        self.fig.myGridIsOn = self.cbox_grid.isChecked()
        self.widgimage.processDraw()


    def on_but_help(self):
        logger.debug('on_but_help - is not implemented yet...', __name__ )
        try :
            self.guihelp.close()
            del self.guihelp
        except :
            self.guihelp = GUIHelp(None,self.help_message())
            self.guihelp.setFixedSize(620,160) 
            self.guihelp.move(self.parentWidget().pos().__add__(QtCore.QPoint(250,60))) 
            self.guihelp.show()


    def help_message(self):
        msg  = 'Mouse control functions:'
        msg += '\nZoom-in image: left mouse click, move, and release in another image position.'
        msg += '\nMiddle mouse button click on image - restores full size image'
        msg += '\nLeft/right mouse click on histogram or color bar - sets min/max amplitude.' 
        msg += '\nMiddle mouse click on histogram or color bar - resets amplitude limits to default.'
        msg += '\n"Reset" button - resets all parameters to default values.'
        return msg


    def popup_confirmation_box(self):
        """Pop-up box for help"""
        msg = QtGui.QMessageBox(self, windowTitle='Help for interactive plot',
            text='This is a help',
            #standardButtons=QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
            standardButtons=QtGui.QMessageBox.Close)

        msg.setDefaultButton(msg.Close)
        clicked = msg.exec_()

        #if   clicked == QtGui.QMessageBox.Save :
        #    logger.debug('Saving is requested', __name__)
        #elif clicked == QtGui.QMessageBox.Discard :
        #    logger.debug('Discard is requested', __name__)
        #else :
        #    logger.debug('Cancel is requested', __name__)
        #return clicked

        
#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    w = PlotImgSpeButtons(None)
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
