#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIAnaPartitions...
#
#------------------------------------------------------------------------

"""GUI sets parameters for analysis"""

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

from ConfigParametersCorAna import confpars as cp
from Logger                 import logger

#---------------------
#  Class definition --
#---------------------
class GUIAnaPartitions ( QtGui.QWidget ) :
    """GUI sets parameters for analysis"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 370, 30)
        self.setWindowTitle('Analysis parameters')
        self.setFrame()

        self.list_ana_types  = ['Static', 'Dynamic']
        self.list_of_methods = ['evenly-spaced','non-evenly-spaced'] 
        self.sect_fields     = []

        self.grid = QtGui.QGridLayout()
        self.grid_row = 1

        if cp.ana_type.value() == self.list_ana_types[0] : 

            self.guiSection('Static  Q   Partition', cp.ana_stat_meth_q  , cp.ana_stat_part_q   )
            self.guiSection('Static  Phi Partition', cp.ana_stat_meth_phi, cp.ana_stat_part_phi )
        else : 
            self.guiSection('Dynamic Q   Partition', cp.ana_dyna_meth_q  , cp.ana_dyna_part_q   ) 
            self.guiSection('Dynamic Phi Partition', cp.ana_dyna_meth_phi, cp.ana_dyna_part_phi ) 

        self.setLayout(self.grid)


        self.showToolTips()
        self.setStyle()

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
        self.frame.setVisible(False)

    def setStyle(self):
        self.setMinimumWidth(380)
        self.setStyleSheet(cp.styleBkgd)

    def guiSection(self, title, method, par) :
        tit0     = QtGui.QLabel(title)
        tit1     = QtGui.QLabel('Method')
        tit2     = QtGui.QLabel('File/Number/Span')
        edi      = QtGui.QLineEdit( str(par.value()) )        
        but      = QtGui.QPushButton('Browse')
        box      = QtGui.QComboBox( self ) 
        box.addItems(self.list_of_methods)
        box.setCurrentIndex( self.list_of_methods.index(method.value()) )

        #edi.setReadOnly( True )  

        edi.setToolTip('Edit number in this field\nor click on "Browse"\nto select the file.')
        but.setToolTip('Click on this button\nand select the file.')
        box.setToolTip('Click on this box\nand select the partitioning method.')

        self.grid.addWidget(tit0, self.grid_row,   0, 1, 8)
        self.grid.addWidget(tit1, self.grid_row+1, 1)
        self.grid.addWidget(box,  self.grid_row+1, 2, 1, 5)
        self.grid.addWidget(but,  self.grid_row+1, 7)
        self.grid.addWidget(tit2, self.grid_row+2, 1, 1, 2)
        self.grid.addWidget(edi,  self.grid_row+2, 6)
        self.grid_row += 3

        tit0   .setStyleSheet (cp.styleTitle)
        tit1   .setStyleSheet (cp.styleLabel)
        tit2   .setStyleSheet (cp.styleLabel)
        edi    .setStyleSheet (cp.styleEdit) # cp.styleEditInfo
        box    .setStyleSheet (cp.styleButton) 
        but    .setStyleSheet (cp.styleButton) 
        edi    .setAlignment (QtCore.Qt.AlignRight)

        width = 60
        but    .setMinimumWidth(width)
        edi    .setFixedWidth(width)
        #box    .setFixedWidth(160)
        #tit0   .setFixedWidth(200)

        self.connect(edi, QtCore.SIGNAL('editingFinished()'),        self.onEdit )
        self.connect(but, QtCore.SIGNAL('clicked()'),                self.onBut  )
        self.connect(box, QtCore.SIGNAL('currentIndexChanged(int)'), self.onBox  )
                                 #   0     1     2    3    4    5       6    7
        self.sect_fields.append( (tit0, tit1, tit2, box, edi, but, method, par ) )


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
        #try    : del cp.guianapartitions # GUIAnaPartitions
        #except : pass # silently ignore

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def onApply(self):
        logger.debug('onApply - is already applied...', __name__)

    def onShow(self):
        logger.debug('onShow - is not implemented yet...', __name__)


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
                dir   = './'
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
                method = fields[6]
                method_selected = box.currentText()
                method.setValue( method_selected ) 
                logger.info('onBox for ' + str(tit.text()) + ' - selected method: ' + method_selected, __name__)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIAnaPartitions ()
    widget.show()
    app.exec_()

#-----------------------------
