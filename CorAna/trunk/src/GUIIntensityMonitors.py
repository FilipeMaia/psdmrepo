#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIIntensityMonitors...
#
#------------------------------------------------------------------------

"""GUI sets parameters for intensity monitors"""

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

#---------------------
#  Class definition --
#---------------------
class GUIIntensityMonitors ( QtGui.QWidget ) :
    """GUI sets parameters for intensity monitors"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 600,300)
        self.setWindowTitle('GUI for Intensity Monitors')
        self.setFrame()

        self.list_of_fields  = []

        self.grid = QtGui.QGridLayout()
        self.grid_row = 0
        self.setTitleBar()
        for i,name in enumerate(cp.imon_name_list) :
            self.guiSection(name, cp.imon_ch1_list[i],
                                  cp.imon_ch2_list[i],
                                  cp.imon_ch3_list[i],
                                  cp.imon_ch4_list[i])

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
        self.setMinimumSize(600,300)
        #self.setMinimumWidth(380)
        #self.setMinimumHeight(300)
        self.setStyleSheet(cp.styleBkgd)


    def setTitleBar(self) :
        list_of_titles = ['Intensity Monitor', 'Ch.1', 'Ch.2', 'Ch.3', 'Ch.4', 'Plot']                               
        for i,t in enumerate(list_of_titles) : 
            label = QtGui.QLabel(t)
            label.setStyleSheet(cp.styleLabel)
            label.setFixedHeight(10)
            self.grid.addWidget(label, self.grid_row, i)
        self.grid_row += 1


    def guiSection(self, name, cbch1=None, cbch2=None, cbch3=None, cbch4=None) :
        edi      = QtGui.QLineEdit( str(name.value()) )        
        but      = QtGui.QPushButton('Browse')
        #box      = QtGui.QComboBox( self ) 
        #box.addItems(self.list_of_methods)
        #box.setCurrentIndex( self.list_of_methods.index(method.value()) )
        cb1 = QtGui.QCheckBox('   +', self)
        cb2 = QtGui.QCheckBox('   +', self)
        cb3 = QtGui.QCheckBox('   +', self)
        cb4 = QtGui.QCheckBox('    ', self)

        cb1.setChecked( cbch1.value() )
        cb2.setChecked( cbch2.value() )
        cb3.setChecked( cbch3.value() )
        cb4.setChecked( cbch4.value() )

        edi.setReadOnly( True )  

        edi.setToolTip('Edit number in this field\nor click on "Browse"\nto select the file.')
        but.setToolTip('Click on this button\nand select the file.')
        #box.setToolTip('Click on this box\nand select the partitioning method.')

        self.grid.addWidget(edi, self.grid_row, 0)
        self.grid.addWidget(cb1, self.grid_row, 1)
        self.grid.addWidget(cb2, self.grid_row, 2)
        self.grid.addWidget(cb3, self.grid_row, 3)
        self.grid.addWidget(cb4, self.grid_row, 4)
        self.grid.addWidget(but, self.grid_row, 5)
        self.grid_row += 1

        edi    .setStyleSheet (cp.styleEditInfo) # cp.styleEditInfo
        #box    .setStyleSheet (cp.styleButton) 
        but    .setStyleSheet (cp.styleButton) 
        edi    .setAlignment (QtCore.Qt.AlignLeft)

        width = 60
        but    .setFixedWidth(width)
        edi    .setFixedWidth(250)
        #box    .setFixedWidth(160)

        self.connect(cb1,  QtCore.SIGNAL('stateChanged(int)'), self.onCBox )
        self.connect(cb2,  QtCore.SIGNAL('stateChanged(int)'), self.onCBox )
        self.connect(cb3,  QtCore.SIGNAL('stateChanged(int)'), self.onCBox )
        self.connect(cb4,  QtCore.SIGNAL('stateChanged(int)'), self.onCBox )

        #self.connect(edi, QtCore.SIGNAL('editingFinished()'),        self.onEdit )
        #self.connect(but, QtCore.SIGNAL('clicked()'),                self.onBut  )
        #self.connect(box, QtCore.SIGNAL('currentIndexChanged(int)'), self.onBox  )
                                 #   0     1      2      3      4  
        self.list_of_fields.append( (edi,  cb1,   cb2,   cb3,   cb4,
                                     name, cbch1, cbch2, cbch3, cbch4 ) )


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
        #try    : del cp.guiintensitymonitors # GUIIntensityMonitors
        #except : pass # silently ignore

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def onApply(self):
        logger.debug('onApply - is already applied...', __name__)

    def onShow(self):
        logger.debug('onShow - is not implemented yet...', __name__)


    def onCBox(self) :
        for row, fields in enumerate(self.list_of_fields) :            
            for col in range(4) :
                cbx, par = fields[col+1], fields[col+1+5]
                if cbx.hasFocus() : 
                    msg = 'onCBox - set status %s of checkbox in row:%s col:%s' % (cbx.isChecked(), row+1, col+1)
                    par.setValue( cbx.isChecked() )
                    logger.info(msg, __name__ )
                    return


#    def onEdit(self):
#        logger.debug('onEdit', __name__)
#        for fields in self.sect_fields :
#            edi = fields[4]
#            par = fields[7]
#            if edi.isModified() :            
#                edi.setModified(False)
#                par.setValue( str(edi.displayText()) )
#                logger.info('Set parameter = ' + str( par.value()), __name__ )
#
#        
#    def onBut(self):
#        logger.debug('onBut', __name__)
#        for fields in self.sect_fields :
#            but = fields[5]
#            if but.hasFocus() :
#                tit = fields[0]
#                edi = fields[4]
#                par = fields[7]
#                #fname = par.value()
#                dir   = './'
#                logger.info('Section: ' + str(tit.text()) + ' - browser for file', __name__ )
#                path  = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',dir) )
#                dname, fname = os.path.split(path)
#
#                if dname == '' or fname == '' :
#                    logger.warning('Input directiry name or file name is empty... keep file name unchanged...', __name__)
#                    return
#
#                edi.setText (path)
#                par.setValue(path)
#                logger.info('selected the file name: ' + str(par.value()), __name__ )
#
#
#    def onBox(self):
#        for fields in self.sect_fields :
#            box = fields[3]
#            if box.hasFocus() :
#                tit    = fields[0]
#                method = fields[6]
#                method_selected = box.currentText()
#                method.setValue( method_selected ) 
#                logger.info('onBox for ' + str(tit.text()) + ' - selected method: ' + method_selected, __name__)
#
#

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIIntensityMonitors ()
    widget.show()
    app.exec_()

#-----------------------------
