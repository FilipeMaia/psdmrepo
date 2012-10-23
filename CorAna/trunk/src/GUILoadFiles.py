#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUILoadFiles...
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

from GUIConfigParameters import *
from Logger              import logger

#---------------------
#  Class definition --
#---------------------
class GUILoadFiles ( QtGui.QWidget ) :
    """GUI sets path to files"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 500, 30)
        self.setWindowTitle('Log files')
        self.setFrame()

        self.sect_fields  = []

        self.grid = QtGui.QGridLayout()
        self.grid_row = 0
        self.guiSection('Dark images run', cp.in_dir_dark, cp.in_file_dark)
        self.guiSection('Flat field',      cp.in_dir_flat, cp.in_file_flat)
        self.guiSection('Data',            cp.in_dir_data, cp.in_file_data) 
        cp.guiconfigparameters = GUIConfigParameters()
        self.grid.addWidget(cp.guiconfigparameters, self.grid_row, 0, 1, 10)
        self.setLayout(self.grid)

        #self.connect( self.box_kin_mode       , QtCore.SIGNAL('currentIndexChanged(int)'), self.on_box_kin_mode        )

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        msg = 'Edit field'
#        self.titKinetic.setToolTip('This section allows to monitor/modify\nthe beam zero parameters\nin transmission mode')
#        self.edi_kin_win_size   .setToolTip( msg )


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):
        self.setStyleSheet(cp.styleBkgd)


    def guiSection(self, title, par_dir, par_file) :

        tit_sect= QtGui.QLabel(title)
        tit_dir = QtGui.QLabel('Dir:')
        tit_file= QtGui.QLabel('File:')
        edi_dir = QtGui.QLineEdit( par_dir .value() )        
        edi_file= QtGui.QLineEdit( par_file.value() )        
        but_dir = QtGui.QPushButton('Browse')
        but_file= QtGui.QPushButton('Browse')

        self.grid.addWidget(tit_sect, self.grid_row,   0, 1, 9)
        self.grid.addWidget(tit_dir,  self.grid_row+1, 1)
        self.grid.addWidget(tit_file, self.grid_row+2, 1)
        self.grid.addWidget(edi_dir,  self.grid_row+1, 2, 1, 7)
        self.grid.addWidget(edi_file, self.grid_row+2, 2, 1, 7)
        self.grid.addWidget(but_dir,  self.grid_row+1, 9)
        self.grid.addWidget(but_file, self.grid_row+2, 9)
        self.grid_row += 3

        tit_sect   .setStyleSheet (cp.styleTitle)
        tit_dir    .setStyleSheet (cp.styleLabel)
        tit_file   .setStyleSheet (cp.styleLabel)
        edi_dir    .setStyleSheet (cp.styleEdit) 
        edi_file   .setStyleSheet (cp.styleEdit) 
        but_dir    .setStyleSheet (cp.styleButton) 
        but_file   .setStyleSheet (cp.styleButton) 

        edi_dir    .setAlignment (QtCore.Qt.AlignRight)
        edi_file   .setAlignment (QtCore.Qt.AlignRight)

        width = 80
        but_dir    .setFixedWidth(width)
        but_file   .setFixedWidth(width)

        self.connect(edi_dir,  QtCore.SIGNAL('editingFinished ()'), self.onEditDir )
        self.connect(edi_file, QtCore.SIGNAL('editingFinished ()'), self.onEditFile)
        self.connect(but_dir,  QtCore.SIGNAL('clicked()'),          self.onButDir  )
        self.connect(but_file, QtCore.SIGNAL('clicked()'),          self.onButFile )
 
        self.sect_fields.append( (tit_sect, tit_dir, tit_file, edi_dir, edi_file, but_dir, but_file, par_dir, par_file ) )


    def setParent(self,parent) :
        self.parent = parent

    def closeEvent(self, event):
        logger.debug('closeEvent')
        try: # try to delete self object in the cp
            del cp.guiloadfiles # GUILoadFiles
        except AttributeError:
            pass # silently ignore

        try    : cp.guiconfigparameters.close()
        except : pass

    def onClose(self):
        logger.debug('onClose')
        self.close()

    def resizeEvent(self, e):
        logger.debug('resizeEvent') 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        logger.debug('moveEvent') 
        pass
#        cp.posGUIMain = (self.pos().x(),self.pos().y())


    def onEditDir(self):
        logger.debug('onEditDir')
        for fields in self.sect_fields :
            edi = fields[3]
            par = fields[7]
            if edi.isModified() :            
                edi.setModified(False)
                par.setValue( str(edi.displayText()) )
                logger.info('Set dir = ' + str( par.value()) )


    def onEditFile(self):
        logger.debug('onEditFile')
        for fields in self.sect_fields :
            edi = fields[4]
            par = fields[8]
            if edi.isModified() :            
                edi.setModified(False)
                par.setValue( str(edi.displayText()) )
                logger.info('Set dir = ' + str( par.value()) )

        
    def onButDir(self):
        logger.debug('onButDir')
        for fields in self.sect_fields :
            but = fields[5]
            if but.hasFocus() :
                tit = fields[0]
                edi = fields[3]
                par = fields[7]
                dir0 = par.value()
                logger.info('Section: ' + str(tit.text()) + ' - browser for directory.')
                path, name = os.path.split(dir0)
                dir = str( QtGui.QFileDialog.getExistingDirectory(self,'Select directory',path) )

                if path == dir0 or path == '' :
                    logger.info('Input directiry has not been changed.')
                    return

                edi.setText(dir)        
                par.setValue(dir)
                logger.info('Set directory: ' + str(par.value()))


    def onButFile(self):
        logger.debug('onButFile')
        for fields in self.sect_fields :
            but = fields[6]
            if but.hasFocus() :
                tit = fields[0]
                edi = fields[4]
                par = fields[8]
                dir   = fields[7].value()
                #dir   = edi.text()
                logger.info('Section: ' + str(tit.text()) + ' - browser for file' )
                path  = str( QtGui.QFileDialog.getOpenFileName(self,'Select file',dir) )
                dname, fname = os.path.split(path)

                if dname == '' or fname == '' :
                    logger.warning('Input directiry name or file name is empty... keep file name unchanged...')
                    return

                edi.setText(fname)
                par.setValue(fname)
                logger.info('selected the file name: ' + str(par.value()) )

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUILoadFiles ()
    widget.show()
    app.exec_()

#-----------------------------
