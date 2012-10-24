#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIAnaSettingsLeft...
#
#------------------------------------------------------------------------

"""GUI sets path to files"""

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
class GUIAnaSettingsLeft ( QtGui.QWidget ) :
    """GUI sets path to files"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Log files')
        self.setFrame()

        self.lmethods     = ['evenly spaced','non-evenly spaced'] 
        self.sect_fields  = []

        self.tit_ana_type = QtGui.QLabel('Select Analysis Type')
        self.tit_mask_set = QtGui.QLabel('Mask Settings')

        self.rad_ana_stat = QtGui.QRadioButton('static analysis')
        self.rad_ana_dyna = QtGui.QRadioButton('dymamic analysis')
        self.rad_grp_ana  = QtGui.QButtonGroup()
        self.rad_grp_ana.addButton(self.rad_ana_stat)
        self.rad_grp_ana.addButton(self.rad_ana_dyna)
        self.rad_ana_stat.setChecked(True)

        self.rad_ana_stat = QtGui.QRadioButton('static analysis')
        self.rad_ana_dyna = QtGui.QRadioButton('dymamic analysis')
        self.rad_ana_grp  = QtGui.QButtonGroup()
        self.rad_ana_grp.addButton(self.rad_ana_stat)
        self.rad_ana_grp.addButton(self.rad_ana_dyna)
        self.rad_ana_stat.setChecked(True)

        self.rad_mask_none = QtGui.QRadioButton('no mask (use all pixels)')
        self.rad_mask_new  = QtGui.QRadioButton('new mask')
        self.rad_mask_file = QtGui.QRadioButton('from existing file')

        self.rad_mask_grp  = QtGui.QButtonGroup()
        self.rad_mask_grp.addButton(self.rad_mask_none)
        self.rad_mask_grp.addButton(self.rad_mask_new )
        self.rad_mask_grp.addButton(self.rad_mask_file)
        self.rad_mask_new.setChecked(True)

        self.but_mask_poly = QtGui.QPushButton('Mask Polygon')
        self.but_browser   = QtGui.QPushButton('Browser')
        self.edi_mask_file = QtGui.QLineEdit( cp.ana_mask_file.value() )        
        self.edi_mask_file.setReadOnly( True )  

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.tit_ana_type, 0, 0, 1, 9)
        self.grid.addWidget(self.rad_ana_stat, 1, 1, 1, 9)
        self.grid.addWidget(self.rad_ana_dyna, 2, 1, 1, 9)
        self.grid_row = 3
        self.guiSection('Static  Q   Partition', self.lmethods[0], cp.ana_stat_part_q   )
        self.guiSection('Static  Phi Partition', self.lmethods[1], cp.ana_stat_part_phi )
        self.guiSection('Dynamic Q   Partition', self.lmethods[0], cp.ana_dyna_part_q   ) 
        self.guiSection('Dynamic Phi Partition', self.lmethods[1], cp.ana_dyna_part_phi ) 
        self.grid.addWidget(self.tit_mask_set,  self.grid_row+1, 0, 1, 9)
        self.grid.addWidget(self.rad_mask_none, self.grid_row+2, 1, 1, 8)
        self.grid.addWidget(self.rad_mask_new , self.grid_row+3, 1, 1, 8)
        self.grid.addWidget(self.rad_mask_file, self.grid_row+4, 1, 1, 8)

        self.grid.addWidget(self.but_mask_poly, self.grid_row+3, 8, 1, 2)
        self.grid.addWidget(self.but_browser,   self.grid_row+4, 8, 1, 2)
        self.grid.addWidget(self.edi_mask_file, self.grid_row+5, 1, 1, 9)

        self.setLayout(self.grid)

#        self.connect( self.but_close,    QtCore.SIGNAL('clicked()'), self.onClose )
#        self.connect( self.but_apply,    QtCore.SIGNAL('clicked()'), self.onApply )
#        self.connect( self.but_show ,    QtCore.SIGNAL('clicked()'), self.onShow )

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        self.but_mask_poly.setToolTip('Click on this button\nto use the polygon')
        self.but_browser  .setToolTip('Click on this button\nto change the mask file.')
        self.edi_mask_file.setToolTip('Click on "Browse"\nto change this field.')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setStyleSheet(cp.styleBkgd)
        self.tit_ana_type.setStyleSheet (cp.styleTitle)
        self.rad_ana_stat.setStyleSheet (cp.styleLabel)
        self.rad_ana_dyna.setStyleSheet (cp.styleLabel)

        self.tit_mask_set .setStyleSheet (cp.styleTitle)
        self.rad_mask_none.setStyleSheet (cp.styleLabel)
        self.rad_mask_new .setStyleSheet (cp.styleLabel)
        self.rad_mask_file.setStyleSheet (cp.styleLabel)

        self.but_mask_poly.setStyleSheet (cp.styleButton)
        self.but_browser  .setStyleSheet (cp.styleButton)
        self.edi_mask_file.setStyleSheet (cp.styleEditInfo)
        self.edi_mask_file.setAlignment (QtCore.Qt.AlignRight)

        #width = 80
        #self.but_mask_poly.setFixedWidth(width)
        #self.but_browser  .setFixedWidth(width)

    def guiSection(self, title, method, par) :

        tit0     = QtGui.QLabel(title)
        tit1     = QtGui.QLabel('Method')
        tit2     = QtGui.QLabel('File/Number/Span')
        edi      = QtGui.QLineEdit( str(par.value()) )        
        but      = QtGui.QPushButton('Browse')
        box      = QtGui.QComboBox( self ) 
        box.addItems(self.lmethods)
        box.setCurrentIndex( self.lmethods.index(method) )

        #edi.setReadOnly( True )  

        edi.setToolTip('Edit number in this field\nor click on "Browse"')
        but.setToolTip('Click on this button\nand select the file.')
        box.setToolTip('Click on this butto\nand select the directory.')

        self.grid.addWidget(tit0, self.grid_row,   0, 1, 9)
        self.grid.addWidget(tit1, self.grid_row+1, 1)
        self.grid.addWidget(tit2, self.grid_row+2, 1)
        self.grid.addWidget(box,  self.grid_row+1, 2, 1, 7)
        self.grid.addWidget(edi,  self.grid_row+2, 2, 1, 8)
        self.grid.addWidget(but,  self.grid_row+1, 9)
        self.grid_row += 3

        tit0   .setStyleSheet (cp.styleTitle)
        tit1   .setStyleSheet (cp.styleLabel)
        tit2   .setStyleSheet (cp.styleLabel)
        edi    .setStyleSheet (cp.styleEdit) # cp.styleEditInfo
        box    .setStyleSheet (cp.styleButton) 
        but    .setStyleSheet (cp.styleButton) 

        edi    .setAlignment (QtCore.Qt.AlignRight)

        width = 80
        but    .setFixedWidth(width)

#        self.connect(edi_dir,  QtCore.SIGNAL('editingFinished ()'), self.onEditDir )
#        self.connect(edi_file, QtCore.SIGNAL('editingFinished ()'), self.onEditFile)
#        self.connect(but_dir,  QtCore.SIGNAL('clicked()'),          self.onButDir  )
#        self.connect(but_file, QtCore.SIGNAL('clicked()'),          self.onButFile )
 
        self.sect_fields.append( (tit0, tit1, tit2, box, edi, but, par ) )


    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        logger.debug('moveEvent', __name__) 
#        cp.posGUIMain = (self.pos().x(),self.pos().y())

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)

#        try    : cp.guiconfigparameters.close()
#        except : pass

        try    : del cp.guianasettingsleft # GUIAnaSettingsLeft
        except : pass # silently ignore

    def onClose(self):
        logger.info('onClose', __name__)
        self.close()

    def onApply(self):
        logger.info('onApply - is already applied...', __name__)

    def onShow(self):
        logger.info('onShow - is not implemented yet...', __name__)


    def onEditDir(self):
        logger.debug('onEditDir')
        for fields in self.sect_fields :
            edi = fields[3]
            par = fields[7]
            if edi.isModified() :            
                edi.setModified(False)
                par.setValue( str(edi.displayText()) )
                logger.info('Set dir = ' + str( par.value()), __name__ )


    def onEditFile(self):
        logger.debug('onEditFile', __name__)
        for fields in self.sect_fields :
            edi = fields[4]
            par = fields[8]
            if edi.isModified() :            
                edi.setModified(False)
                par.setValue( str(edi.displayText()) )
                logger.info('Set dir = ' + str( par.value()), __name__ )

        
    def onButDir(self):
        logger.debug('onButDir', __name__)
        for fields in self.sect_fields :
            but = fields[5]
            if but.hasFocus() :
                tit = fields[0]
                edi = fields[3]
                par = fields[7]
                dir0 = par.value()
                logger.info('Section: ' + str(tit.text()) + ' - browser for directory.', __name__)
                path, name = os.path.split(dir0)
                dir = str( QtGui.QFileDialog.getExistingDirectory(self,'Select directory',path) )

                if dir == dir0 or dir == '' :
                    logger.info('Input directiry has not been changed.', __name__)
                    return

                edi.setText(dir)        
                par.setValue(dir)
                logger.info('Set directory: ' + str(par.value()), __name__)


    def onButFile(self):
        logger.debug('onButFile', __name__)
        for fields in self.sect_fields :
            but = fields[6]
            if but.hasFocus() :
                tit = fields[0]
                edi = fields[4]
                par = fields[8]
                dir = fields[7].value()
                #dir   = edi.text()
                logger.info('Section: ' + str(tit.text()) + ' - browser for file', __name__ )
                path  = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',dir) )
                dname, fname = os.path.split(path)

                if dname == '' or fname == '' :
                    logger.warning('Input directiry name or file name is empty... keep file name unchanged...')
                    return

                edi.setText(fname)
                par.setValue(fname)
                logger.info('selected the file name: ' + str(par.value()), __name__ )


    def onButDirWork(self):
        logger.info('onButDirWork - Select work directory.', __name__)
        dir0 = cp.dir_work.value()
        path, name = os.path.split(dir0)
        dir = str( QtGui.QFileDialog.getExistingDirectory(self,'Select directory',path) )

        if dir == dir0 or dir == '' :
            logger.info('Work directiry has not been changed.', __name__)
            return

        self.edi_dir_work.setText(dir)        
        cp.dir_work.setValue(dir)
        logger.info('Set directory: ' + str(cp.dir_work.value()), __name__)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIAnaSettingsLeft ()
    widget.show()
    app.exec_()

#-----------------------------
