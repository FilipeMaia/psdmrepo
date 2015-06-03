#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIListOfTau...
#
#------------------------------------------------------------------------

"""GUI sets the list of tau indexes"""

#------------------------------
#  Module's version from CVS --
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
from Logger import logger
from ConfigParametersCorAna import confpars as cp
from FileNameManager        import fnm
import GlobalUtils          as     gu
from GUIFileBrowser         import *

#---------------------
#  Class definition --
#---------------------
class GUIListOfTau ( QtGui.QWidget ) :
    """GUI sets the list of tau indexes"""

    list_tau_options = ['auto', 'file']

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(20, 40, 390, 30)
        self.setWindowTitle('List of tau')
        self.setFrame()

        self.tit_tau_list_set  = QtGui.QLabel(u'List of \u03C4 indexes:')  # tau = u"\u03C4"
        self.rad_tau_list_auto = QtGui.QRadioButton('auto-generated (log-like)')
        self.rad_tau_list_file = QtGui.QRadioButton('from file')
        self.rad_tau_list_grp  = QtGui.QButtonGroup()
        self.rad_tau_list_grp.addButton(self.rad_tau_list_auto)
        self.rad_tau_list_grp.addButton(self.rad_tau_list_file)
        if cp.ana_tau_list_type.value() == self.list_tau_options[0] : self.rad_tau_list_auto.setChecked(True)
        if cp.ana_tau_list_type.value() == self.list_tau_options[1] : self.rad_tau_list_file.setChecked(True)

        self.but_file          = QtGui.QPushButton('File:')
        self.but_brow          = QtGui.QPushButton('View/Edit')
        self.but_reset         = QtGui.QPushButton('Reset')
        self.edi_tau_list_file = QtGui.QLineEdit( fnm.path_tau_list() )       
        self.edi_tau_list_file.setReadOnly( True )  

        self.grid = QtGui.QGridLayout()

        self.grid_row = 0
        self.grid.addWidget(self.tit_tau_list_set,      self.grid_row+1, 0, 1, 9)
        self.grid.addWidget(self.rad_tau_list_auto,     self.grid_row+2, 1, 1, 6)
        self.grid.addWidget(self.rad_tau_list_file,     self.grid_row+3, 1, 1, 6)
        self.grid.addWidget(self.but_brow,              self.grid_row+3, 5, 1, 2)
        self.grid.addWidget(self.but_reset,             self.grid_row+3, 7, 1, 2)
        self.grid.addWidget(self.but_file,              self.grid_row+4, 0, 1, 2)
        self.grid.addWidget(self.edi_tau_list_file,     self.grid_row+4, 2, 1, 7)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.grid)
        self.vbox.addStretch(1)

        self.setLayout(self.vbox)

        self.connect( self.rad_tau_list_auto, QtCore.SIGNAL('clicked()'), self.onTauRadioGrp )
        self.connect( self.rad_tau_list_file, QtCore.SIGNAL('clicked()'), self.onTauRadioGrp )
        self.connect( self.but_file,          QtCore.SIGNAL('clicked()'), self.onButFile   )
        self.connect( self.but_brow,          QtCore.SIGNAL('clicked()'), self.onButBrow   )
        self.connect( self.but_reset,         QtCore.SIGNAL('clicked()'), self.onButReset  )

        self.showToolTips()
        self.setStyle()
        self.setFieldsState()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit field'

        msg_rad_tau_list = 'Use this group of radio buttons\nto select the tau-list options'
        self.rad_tau_list_auto.setToolTip(msg_rad_tau_list)
        self.rad_tau_list_file.setToolTip(msg_rad_tau_list)
        self.but_reset        .setToolTip('Resets the file name to default')
        self.but_brow         .setToolTip('Opens browser for the file')
        self.but_file         .setToolTip('Change the file name')
        self.edi_tau_list_file.setToolTip('Click on "File:" button\nto change this field')



    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        width = 60
        self.                  setFixedWidth(390)
        self.                  setStyleSheet (cp.styleBkgd)

        self.tit_tau_list_set .setStyleSheet (cp.styleTitle)
        self.rad_tau_list_auto.setStyleSheet (cp.styleLabel)
        self.rad_tau_list_file.setStyleSheet (cp.styleLabel)

        self.but_file         .setStyleSheet (cp.styleButton)
        self.but_brow         .setStyleSheet (cp.styleButton)
        self.but_reset        .setStyleSheet (cp.styleButton)
        self.edi_tau_list_file.setStyleSheet (cp.styleEditInfo)
        self.edi_tau_list_file.setAlignment (QtCore.Qt.AlignRight)


    def setFieldsState(self):
        is_active = self.rad_tau_list_file.isChecked()
        self.but_file  .setEnabled( is_active )
        self.but_brow  .setEnabled( is_active )
        self.but_reset .setEnabled( is_active )


    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__)
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__)
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        try: # try to delete self object in the cp
            del cp.guilistoftau # GUIListOfTau

        except AttributeError:
            pass # silently ignore

        #try :
        #    cp.tau_listeditor.close()
        #    del cp.tau_listeditor
        #except :
        #    pass


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def on(self):
        logger.debug('on click - is not implemented yet', __name__)


    def onTauRadioGrp(self):
        if self.rad_tau_list_auto.isChecked() : cp.ana_tau_list_type.setValue(self.list_tau_options[0])
        if self.rad_tau_list_file.isChecked() : cp.ana_tau_list_type.setValue(self.list_tau_options[1])
        logger.info('onTauRadioGrp - set cp.ana_tau_list_type = ' + cp.ana_tau_list_type.value(), __name__)
        self.setFieldsState()

 
    def onButFile(self):
        logger.debug('onButFile', __name__)

        path = fnm.path_tau_list()
        #print 'path_tau_list()', path

        if path is None : dname, fname = cp.ana_tau_list_fname.value_def(), cp.ana_tau_list_dname.value_def()
        else            : dname, fname = os.path.split(path)

        path = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',path) )
        dname, fname = os.path.split(path)

        if dname == '' or fname == '' :
            logger.warning('Input directiry name or file name is empty... keep file name unchanged...', __name__)
            return

        self.edi_tau_list_file.setText(path)
        cp.ana_tau_list_fname.setValue(fname)
        cp.ana_tau_list_dname.setValue(dname)
        logger.info('selected file for tau_list: ' + str(cp.ana_tau_list_fname.value()), __name__ )


    def onButReset(self):
        cp.ana_tau_list_dname.setDefault()
        cp.ana_tau_list_fname.setDefault()
        self.edi_tau_list_file.setText(fnm.path_tau_list())


    def onButBrow (self):       
        logger.debug('onButBrow', __name__)
        try    :
            cp.guifilebrowser.close()
        except :
            cp.guifilebrowser = GUIFileBrowser(None, \
                                               [fnm.path_tau_list(), fnm.path_cora_proc_tau_in(), fnm.path_cora_merge_tau()], \
                                                fnm.path_tau_list(), is_editable=True)
            try    : cp.guifilebrowser.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            except : cp.guifilebrowser.move(QtCore.QPoint(300,120))
            cp.guifilebrowser.show()

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIListOfTau ()
    widget.show()
    app.exec_()

#-----------------------------
