#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIConfigFile...
#
#------------------------------------------------------------------------

"""GUI works with configuration parameters management"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from CalibManager.Frame     import Frame
from ConfigParametersForApp import cp
from Logger                 import logger

#---------------------
#  Class definition --
#---------------------
#class GUIConfigFile ( QtGui.QWidget ) :
class GUIConfigFile ( Frame ) :
    """GUI for configuration file parameters management"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        """Constructor"""

        #QtGui.QWidget.__init__(self, parent)
        Frame.__init__(self, parent, mlw=1)

        #self.parent = cp.guimain

        self.setGeometry(370, 350, 500,150)
        self.setWindowTitle('Configuration File')
        #self.setFrame()
        
        self.titFile     = QtGui.QLabel('File with configuration parameters:')
        self.titPars     = QtGui.QLabel('Operations on file:')
        self.butFile     = QtGui.QPushButton('File:')
        self.butRead     = QtGui.QPushButton('Read')
        self.butWrite    = QtGui.QPushButton('Save')
        self.butDefault  = QtGui.QPushButton('Reset default')
        self.butPrint    = QtGui.QPushButton('Print current')
        self.ediFile     = QtGui.QLineEdit( cp.fname_cp )        
        self.cbxSave     = QtGui.QCheckBox('&Save at exit')
        self.cbxSave.setChecked( cp.save_cp_at_exit.value() )
 
        grid = QtGui.QGridLayout()
        grid.addWidget(self.titFile,       0, 0, 1, 5)
        grid.addWidget(self.butFile,       1, 0)
        grid.addWidget(self.ediFile,       1, 1, 1, 4)
        grid.addWidget(self.titPars,       2, 0, 1, 3)
        grid.addWidget(self.cbxSave,       2, 4)
        grid.addWidget(self.butRead,       3, 1)
        grid.addWidget(self.butWrite,      3, 2)
        grid.addWidget(self.butDefault,    3, 3)
        grid.addWidget(self.butPrint,      3, 4)
        #self.setLayout(grid)

        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addLayout(grid)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)


        self.connect(self.ediFile,      QtCore.SIGNAL('editingFinished ()'), self.onEditFile     )
        self.connect(self.butRead,      QtCore.SIGNAL('clicked()'),          self.onRead         )
        self.connect(self.butWrite,     QtCore.SIGNAL('clicked()'),          self.onSave         )
        self.connect(self.butPrint,     QtCore.SIGNAL('clicked()'),          self.onPrint        )
        self.connect(self.butDefault,   QtCore.SIGNAL('clicked()'),          self.onDefault      )
        self.connect(self.butFile,      QtCore.SIGNAL('clicked()'),          self.onFile         ) 
        self.connect(self.cbxSave,      QtCore.SIGNAL('stateChanged(int)'),  self.onCbxSave      ) 
 
        self.showToolTips()
        self.setStyle()


    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        self.ediFile   .setToolTip('Type the file path name here,\nor better use "Browse" button.')
        self.butFile   .setToolTip('Select the file path name\nto read/write the configuration parameters.')
        self.butRead   .setToolTip('Read the configuration parameters from file.')
        self.butWrite  .setToolTip('Save (write) the configuration parameters in file.')
        self.butDefault.setToolTip('Reset the configuration parameters\nto their default values.')
        self.butPrint  .setToolTip('Print current values of the configuration parameters.')

#    def setFrame(self):
#        self.frame = QtGui.QFrame(self)
#        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
#        self.frame.setLineWidth(0)
#        self.frame.setMidLineWidth(1)
#        self.frame.setGeometry(self.rect())
#        #self.frame.setVisible(False)

    def setStyle(self):

        self.setMinimumSize(500,150)
        self.setMaximumSize(700,150)
        #width = 80
        #self.butFile .setFixedWidth(width)
        #self.edi_kin_win_size   .setAlignment(QtCore.Qt.AlignRight)

        self           .setStyleSheet(cp.styleBkgd)
        self.titFile   .setStyleSheet(cp.styleLabel)
        self.titPars   .setStyleSheet(cp.styleLabel)
        self.ediFile   .setStyleSheet(cp.styleEdit) 

        self.butFile   .setStyleSheet(cp.styleButton) 
        self.butRead   .setStyleSheet(cp.styleButton)
        self.butWrite  .setStyleSheet(cp.styleButton)
        self.butDefault.setStyleSheet(cp.styleButton)
        self.butPrint  .setStyleSheet(cp.styleButton)
        self.cbxSave   .setStyleSheet(cp.styleLabel)
        #self.butClose  .setStyleSheet(cp.styleButtonClose)

        self.butFile   .setFixedWidth(50)
 
    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        #self.frame.setGeometry(self.rect())
        pass

    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        #try    : del cp.guiconfigparameters 
        #except : pass

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def onRead(self):
        logger.debug('onRead', __name__)
        cp.readParametersFromFile( self.getFileNameFromEditField() )
        self.ediFile.setText( cp.fname_cp )
        #self.parent.ediFile.setText( cp.fname_cp )
        #self.refreshGUIWhatToDisplay()

    def onWrite(self):
        fname = self.getFileNameFromEditField()
        logger.info('onWrite - save all configuration parameters in file: ' + fname, __name__)
        cp.saveParametersInFile( fname )

    def onSave(self):
        fname = cp.fname_cp
        logger.info('onSave - save all configuration parameters in file: ' + fname, __name__)
        cp.saveParametersInFile( fname )

    def onDefault(self):
        logger.info('onDefault - Set default values of configuration parameters.', __name__)
        cp.setDefaultValues()
        self.ediFile.setText( cp.fname_cp )
        #self.refreshGUIWhatToDisplay()

    def onPrint(self):
        logger.info('onPrint', __name__)
        cp.printParameters()

    def onFile(self):
        logger.debug('onFile', __name__)
        self.path = self.getFileNameFromEditField()
        self.dname,self.fname = os.path.split(self.path)
        logger.info('dname : %s' % (self.dname), __name__)
        logger.info('fname : %s' % (self.fname), __name__)
        self.path = str( QtGui.QFileDialog.getOpenFileName(self,'Open file',self.dname) )
        self.dname,self.fname = os.path.split(self.path)

        if self.dname == '' or self.fname == '' :
            logger.info('Input directiry name or file name is empty... use default values', __name__)
        else :
            self.ediFile.setText(self.path)
            cp.fname_cp = self.path

    def onEditFile(self):
        logger.debug('onEditFile', __name__)
        self.path = self.getFileNameFromEditField()
        #cp.fname_cp.setValue(self.path)
        cp.fname_cp = self.path
        dname,fname = os.path.split(self.path)
        logger.info('Set dname : %s' % (dname), __name__)
        logger.info('Set fname : %s' % (fname), __name__)

    def getFileNameFromEditField(self):
        return str( self.ediFile.displayText() )

    def onCbxSave(self):
        #if self.cbx.hasFocus() :
        par = cp.save_cp_at_exit
        cbx = self.cbxSave
        tit = cbx.text()

        par.setValue( cbx.isChecked() )
        msg = 'check box ' + tit  + ' is set to: ' + str( par.value())
        logger.info(msg, __name__ )


#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIConfigFile ()
    widget.show()
    app.exec_()

#-----------------------------
