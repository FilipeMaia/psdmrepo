#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIAnaSettingsLeft...
#
#------------------------------------------------------------------------

"""GUI sets parameters for analysis"""

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
    """GUI sets parameters for analysis"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Log files')
        self.setFrame()

        self.list_ana_types  = ['static', 'dynamic']
        self.list_mask_types = ['no-mask', 'new-mask', 'from-file']
        self.list_of_methods = ['evenly-spaced','non-evenly-spaced'] 
        self.sect_fields     = []

        self.tit_ana_type = QtGui.QLabel('Select Analysis Type')
        self.rad_ana_stat = QtGui.QRadioButton('static analysis')
        self.rad_ana_dyna = QtGui.QRadioButton('dymamic analysis')
        self.rad_ana_grp  = QtGui.QButtonGroup()
        self.rad_ana_grp.addButton(self.rad_ana_stat)
        self.rad_ana_grp.addButton(self.rad_ana_dyna)
        if cp.ana_type.value() == self.list_ana_types[0] : self.rad_ana_stat.setChecked(True)
        else                                             : self.rad_ana_dyna.setChecked(True)

        self.tit_mask_set  = QtGui.QLabel('Mask Settings')
        self.rad_mask_none = QtGui.QRadioButton('no mask (use all pixels)')
        self.rad_mask_new  = QtGui.QRadioButton('new mask')
        self.rad_mask_file = QtGui.QRadioButton('from existing file')
        self.rad_mask_grp  = QtGui.QButtonGroup()
        self.rad_mask_grp.addButton(self.rad_mask_none)
        self.rad_mask_grp.addButton(self.rad_mask_new )
        self.rad_mask_grp.addButton(self.rad_mask_file)
        if cp.ana_mask_type.value() == self.list_mask_types[0] : self.rad_mask_none.setChecked(True)
        if cp.ana_mask_type.value() == self.list_mask_types[1] : self.rad_mask_new .setChecked(True)
        if cp.ana_mask_type.value() == self.list_mask_types[2] : self.rad_mask_file.setChecked(True)

        self.but_mask_poly = QtGui.QPushButton('Mask Polygon')
        self.but_browser   = QtGui.QPushButton('Browser')
        self.edi_mask_file = QtGui.QLineEdit( cp.ana_mask_file.value() )        
        self.edi_mask_file.setReadOnly( True )  

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.tit_ana_type, 0, 0, 1, 9)
        self.grid.addWidget(self.rad_ana_stat, 1, 1, 1, 9)
        self.grid.addWidget(self.rad_ana_dyna, 2, 1, 1, 9)
        self.grid_row = 3

        self.guiSection('Static  Q   Partition', cp.ana_stat_meth_q  , cp.ana_stat_part_q   )
        self.guiSection('Static  Phi Partition', cp.ana_stat_meth_phi, cp.ana_stat_part_phi )
        self.guiSection('Dynamic Q   Partition', cp.ana_dyna_meth_q  , cp.ana_dyna_part_q   ) 
        self.guiSection('Dynamic Phi Partition', cp.ana_dyna_meth_phi, cp.ana_dyna_part_phi ) 

        self.grid.addWidget(self.tit_mask_set,  self.grid_row+1, 0, 1, 9)
        self.grid.addWidget(self.rad_mask_none, self.grid_row+2, 1, 1, 8)
        self.grid.addWidget(self.rad_mask_new , self.grid_row+3, 1, 1, 8)
        self.grid.addWidget(self.rad_mask_file, self.grid_row+4, 1, 1, 8)

        self.grid.addWidget(self.but_mask_poly, self.grid_row+3, 8, 1, 2)
        self.grid.addWidget(self.but_browser,   self.grid_row+4, 8, 1, 2)
        self.grid.addWidget(self.edi_mask_file, self.grid_row+5, 1, 1, 9)

        self.setLayout(self.grid)

        self.connect( self.rad_ana_stat,     QtCore.SIGNAL('clicked()'), self.onAnaRadioGrp )
        self.connect( self.rad_ana_dyna,     QtCore.SIGNAL('clicked()'), self.onAnaRadioGrp )
        self.connect( self.rad_mask_none,    QtCore.SIGNAL('clicked()'), self.onMaskRadioGrp )
        self.connect( self.rad_mask_new,     QtCore.SIGNAL('clicked()'), self.onMaskRadioGrp )
        self.connect( self.rad_mask_file,    QtCore.SIGNAL('clicked()'), self.onMaskRadioGrp )

        self.connect( self.but_mask_poly,    QtCore.SIGNAL('clicked()'), self.onMaskPoly     )
        self.connect( self.but_browser,      QtCore.SIGNAL('clicked()'), self.onButBrowser   )

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        self.but_mask_poly.setToolTip('Click on this button\nto use the polygon mask')
        self.but_browser  .setToolTip('Click on this button\nto change the mask file.')
        self.edi_mask_file.setToolTip('Click on "Browse"\nto change this field.')
        msg_rad_ana  = 'Use this group of radio buttons\nto select the analysis type'
        msg_rad_mask = 'Use this group of radio buttons\nto select the type of mask'
        self.rad_ana_stat .setToolTip(msg_rad_ana)
        self.rad_ana_dyna .setToolTip(msg_rad_ana)
        self.rad_mask_none.setToolTip(msg_rad_mask)
        self.rad_mask_new .setToolTip(msg_rad_mask)
        self.rad_mask_file.setToolTip(msg_rad_mask)


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.setMinimumWidth(450)
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
        box.addItems(self.list_of_methods)
        box.setCurrentIndex( self.list_of_methods.index(method.value()) )

        #edi.setReadOnly( True )  

        edi.setToolTip('Edit number in this field\nor click on "Browse"\nto select the file.')
        but.setToolTip('Click on this button\nand select the file.')
        box.setToolTip('Click on this box\nand select the partitioning method.')

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

        self.connect(edi, QtCore.SIGNAL('editingFinished()'),        self.onEdit )
        self.connect(but, QtCore.SIGNAL('clicked()'),                self.onBut  )
        self.connect(box, QtCore.SIGNAL('currentIndexChanged(int)'), self.onBox  )
                                 #   0     1     2    3    4    5       6    7
        self.sect_fields.append( (tit0, tit1, tit2, box, edi, but, method, par ) )


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
        try    : del cp.guianasettingsleft # GUIAnaSettingsLeft
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


    def onButBrowser(self):
        logger.info('onButBrowser', __name__)

        path = cp.ana_mask_file.value()
        if path == None : dname = './'
        else            : dname, fname = os.path.split(path)

        path  = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',dname) )
        dname, fname = os.path.split(path)

        if dname == '' or fname == '' :
            logger.warning('Input directiry name or file name is empty... keep file name unchanged...', __name__)
            return

        self.edi_mask_file.setText(path)
        cp.ana_mask_file.setValue(path)
        logger.info('selected file for mask: ' + str(cp.ana_mask_file.value()), __name__ )


    def onAnaRadioGrp(self):
        if self.rad_ana_stat.isChecked() : cp.ana_type.setValue(self.list_ana_types[0])
        if self.rad_ana_dyna.isChecked() : cp.ana_type.setValue(self.list_ana_types[1])
        logger.info('onAnaRadioGrp - set cp.ana_type = '+ cp.ana_type.value(), __name__)

    def onMaskRadioGrp(self):
        if self.rad_mask_none.isChecked() : cp.ana_mask_type.setValue(self.list_mask_types[0])
        if self.rad_mask_new .isChecked() : cp.ana_mask_type.setValue(self.list_mask_types[1])
        if self.rad_mask_file.isChecked() : cp.ana_mask_type.setValue(self.list_mask_types[2])
        logger.info('onMaskRadioGrp - set cp.ana_mask_type = ' + cp.ana_mask_type.value(), __name__)

    def onMaskPoly(self):
        logger.info('onMaskPoly - is not implemented yet...', __name__)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIAnaSettingsLeft ()
    widget.show()
    app.exec_()

#-----------------------------
