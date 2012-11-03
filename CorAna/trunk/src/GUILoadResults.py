#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUILoadResults...
#
#------------------------------------------------------------------------

"""GUI Load Results"""

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
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersCorAna import confpars as cp
from Logger                 import logger
#from Overlay                import *

#---------------------
#  Class definition --
#---------------------
class GUILoadResults ( QtGui.QWidget ) :
    """GUI Load Results"""

    def __init__ ( self, parent=None ) :
        #super(GUILoadResults, self).__init__()
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 100)
        self.setWindowTitle('Load Results')
        self.setFrame()

        self.list_of_modes = ['NONE','mode-1','mode-2','mode-3'] 
        self.sect_fields     = []

        self.grid = QtGui.QGridLayout()
        self.grid_row = 1

        self.guiSection('Load Results', cp.res_load_mode, cp.res_fname )
        ##self.guiSection('Static  Phi Partition', cp.ana_stat_meth_phi, cp.ana_stat_part_phi )

        self.setLayout(self.grid)

        self.showToolTips()
        self.setStyle()

        #self.overlay = Overlay(self,'Load Results')
                
    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        msg = 'Use this GUI to set partitions.'
        self.setToolTip(msg)

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setMinimumWidth(400)
        self.setStyleSheet(cp.styleBkgd)

    def guiSection(self, title, mode, par) :
        tit      = QtGui.QLabel(title)
        tit_box  = QtGui.QLabel('Plot Mode')
        #tit_edi  = QtGui.QLabel('File')
        edi      = QtGui.QLineEdit( str(par.value()) )        
        but      = QtGui.QPushButton('Load')
        box      = QtGui.QComboBox( self ) 
        box.addItems(self.list_of_modes)
        box.setCurrentIndex( self.list_of_modes.index(mode.value()) )

        #edi.setReadOnly( True )  

        edi.setToolTip('Edit number in this field\nor click on "Browse"\nto select the file.')
        but.setToolTip('Click on this button\nand select the file.')
        box.setToolTip('Click on this box\nand select the partitioning mode.')

        self.grid.addWidget(tit,     self.grid_row,  0, 1, 9)
        #self.grid.addWidget(tit_edi, self.grid_row+1, 1)
        self.grid.addWidget(tit_box, self.grid_row+2, 0)
        self.grid.addWidget(edi,     self.grid_row+1, 1, 1, 9)
        self.grid.addWidget(box,     self.grid_row+2, 1, 1, 4)
        self.grid.addWidget(but,     self.grid_row+1, 0)
        self.grid_row += 3

        tit    .setStyleSheet (cp.styleTitle)
        tit_box.setStyleSheet (cp.styleLabel)
        #tit_edi.setStyleSheet (cp.styleLabel)
        edi    .setStyleSheet (cp.styleEdit) # cp.styleEditInfo
        box    .setStyleSheet (cp.styleButton) 
        but    .setStyleSheet (cp.styleButton) 
        edi    .setAlignment (QtCore.Qt.AlignRight)

        width = 80
        but    .setFixedWidth(width)

        self.connect(edi, QtCore.SIGNAL('editingFinished()'),        self.onEdit )
        self.connect(but, QtCore.SIGNAL('clicked()'),                self.onBut  )
        self.connect(box, QtCore.SIGNAL('currentIndexChanged(int)'), self.onBox  )
                                 #  0        1        2    3    4    5     6    7
        self.sect_fields.append( (tit, tit_box, 'tit_edi', box, edi, but, mode, par ) )


    def setParent(self,parent) :
        self.parent = parent


    def resizeEvent(self, e):
        logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

        #self.overlay.resize(e.size())
        e.accept()

    def moveEvent(self, e):
        logger.debug('moveEvent', __name__) 
#        cp.posGUIMain = (self.pos().x(),self.pos().y())

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        try    : del cp.guiloadresults # GUILoadResults
        except : pass # silently ignore

    def onClose(self):
        logger.info('onClose', __name__)
        self.close()

    def onApply(self):
        logger.info('onApply - is already applied...', __name__)

    def onShow(self):
        logger.info('onShow - is not implemented yet...', __name__)


    def onEdit(self):
        logger.debug('onEdit', __name__)
        for fields in self.sect_fields :
            edi = fields[4]
            par = fields[7]
            if edi.isModified() :            
                edi.setModified(False)
                par.setValue( str(edi.displayText()) )
                logger.info('Set parameter = ' + str( par.value()), __name__ )

        
    def onBut(self):
        logger.debug('onBut', __name__)
        for fields in self.sect_fields :
            but = fields[5]
            if but.hasFocus() :
                tit = fields[0]
                edi = fields[4]
                par = fields[7]
                #fname = par.value()
                dir   = cp.dir_results.value() # './'
                logger.info('Section: ' + str(tit.text()) + ' - browser for file', __name__ )
                path  = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',dir) )
                dname, fname = os.path.split(path)

                if dname == '' or fname == '' :
                    logger.warning('Input directiry name or file name is empty... keep file name unchanged...', __name__)
                    return

                edi.setText (path)
                par.setValue(path)
                logger.info('selected the file name: ' + str(par.value()), __name__ )


    def onBox(self):
        for fields in self.sect_fields :
            box = fields[3]
            if box.hasFocus() :
                tit    = fields[0]
                mode = fields[6]
                mode_selected = box.currentText()
                mode.setValue( mode_selected ) 
                logger.info('onBox for ' + str(tit.text()) + ' - selected mode: ' + mode_selected, __name__)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUILoadResults ()
    widget.show()
    app.exec_()

#-----------------------------
