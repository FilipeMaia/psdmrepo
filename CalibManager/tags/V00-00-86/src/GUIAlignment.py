
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIAlignment...
#
#------------------------------------------------------------------------

"""Renders the main GUI for the CalibManager.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""

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
import numpy as np

# For self-run debugging:
#if __name__ == "__main__" :
#    import matplotlib
#    matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from CalibManager.Frame                  import Frame
from CalibManager.ConfigParametersForApp import cp
from CalibManager.Logger                 import logger
import CalibManager.GlobalUtils          as     gu
from CalibManager.PlotImgSpe             import *
from CalibManager.GUIFileBrowser         import *
#from CalibManager.FileNameManager        import fnm

#---------------------
#  Class definition --
#---------------------
#class GUIAlignment ( QtGui.QWidget ) :
class GUIAlignment ( Frame ) : 
    """GUI to set parameters for stand-alome detector segments alignment application geo.

    @see BaseClass
    @see OtherClass
    """
    def __init__ (self, parent=None, app=None) :

        self.name = 'GUIAlignment'
        self.myapp = app
        Frame.__init__(self, parent, mlw=1)

        self.fname_geo_img_nda      = cp.fname_geo_img_nda  
        self.fname_geo_in           = cp.fname_geo_in
        self.fname_geo_out          = cp.fname_geo_out
        self.geo_log_level          = cp.geo_log_level  
        self.list_geo_log_levels    = cp.list_geo_log_levels  

        cp.setIcons()

        self.setGeometry(10, 25, 725,360)
        self.setWindowTitle('Detector Alignment')
        #self.setWindowIcon(cp.icon_monitor)
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False


        self.lab_status = QtGui.QLabel('Status: ')

        self.but_geo_in           = QtGui.QPushButton('  1. Select geometry input file')
        self.but_geo_out          = QtGui.QPushButton('Output geo file')
        self.but_geo_img_nda      = QtGui.QPushButton('  2. Select ndarray for image file')
        self.but_geo              = QtGui.QPushButton('  3. Launch\n  command:')

        self.but_plot             = QtGui.QPushButton('Plot')
        self.but_view             = QtGui.QPushButton('View')

        self.edi_geo_in       = QtGui.QLineEdit (self.fname_geo_in.value())
        self.edi_geo_out      = QtGui.QLineEdit (self.fname_geo_out.value())
        self.edi_geo_img_nda  = QtGui.QLineEdit (self.fname_geo_img_nda.value())
        self.edi_geo          = QtGui.QTextEdit ('Command is here')

        #self.lab_geo        = QtGui.QLabel('Command:') 
        self.lab_log_level  = QtGui.QLabel('Optional arguments:    Log level') 
        self.box_log_level  = QtGui.QComboBox(self) 
        self.box_log_level.addItems(self.list_geo_log_levels)
        self.box_log_level.setCurrentIndex(self.list_geo_log_levels.index(self.geo_log_level.value()))

        self.grid = QtGui.QGridLayout()
        self.grid_row = 0
        self.grid.addWidget(self.but_geo_in,        self.grid_row,     0, 1, 4)
        self.grid.addWidget(self.edi_geo_in,        self.grid_row,     4, 1, 6)

        self.grid.addWidget(self.but_geo_img_nda,   self.grid_row+2,   0, 1, 4)
        self.grid.addWidget(self.edi_geo_img_nda,   self.grid_row+2,   4, 1, 6)

        self.grid.addWidget(self.lab_log_level,     self.grid_row+3,   0, 1, 3)
        self.grid.addWidget(self.box_log_level,     self.grid_row+3,   3, 1, 1)
        self.grid.addWidget(self.but_geo_out,       self.grid_row+3,   4, 1, 2)
        self.grid.addWidget(self.edi_geo_out,       self.grid_row+3,   6, 1, 4)

        #self.grid.addWidget(self.lab_geo,           self.grid_row+4,   0, 1)

        self.grid.addWidget(self.but_geo,           self.grid_row+4,   0)
        self.grid.addWidget(self.edi_geo,           self.grid_row+4,   1, 1, 9)

        self.grid.addWidget(self.but_plot,          self.grid_row+8,   8)
        self.grid.addWidget(self.but_view,          self.grid_row+8,   9)

        self.hboxS = QtGui.QHBoxLayout()
        self.hboxS.addWidget(self.lab_status)
        self.hboxS.addStretch(1)     

        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addLayout(self.grid)
        self.vbox.addStretch(1)
        self.vbox.addLayout(self.hboxS) 
        self.setLayout(self.vbox)

        self.connect(self.but_geo_in     , QtCore.SIGNAL('clicked()'), self.on_but_geo_in     ) 
        self.connect(self.but_geo_out    , QtCore.SIGNAL('clicked()'), self.on_but_geo_out    ) 
        self.connect(self.but_geo_img_nda, QtCore.SIGNAL('clicked()'), self.on_but_geo_img_nda) 
        self.connect(self.but_geo        , QtCore.SIGNAL('clicked()'), self.on_but_geo        ) 
        self.connect(self.but_plot       , QtCore.SIGNAL('clicked()'), self.on_but_plot       ) 
        self.connect(self.but_view       , QtCore.SIGNAL('clicked()'), self.on_but_view       ) 
        self.connect(self.box_log_level  , QtCore.SIGNAL('currentIndexChanged(int)'), self.on_box_log_level)
 
        self.showToolTips()
        self.setStyle()

        self.setStatus(0)
        self.set_command_window()

        cp.guialignment = self
        self.move(10,25)
        
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        self.but_geo_img_nda.setToolTip('Open file browser dialog window \nand select the file with image data') 
        #pass


    def setStyle(self):

        self.setMinimumSize(725,360)
        self.setMaximumWidth(800)
        self.lab_status.setMinimumWidth(600) 

        self.                setStyleSheet(cp.styleBkgd)
        #self.but_mask_editor.setStyleSheet(cp.styleButton)
        #self.but_mask_editor.setFixedWidth(200)
        #self.but_mask_editor.setMinimumHeight(60)
        #self.but_mask_editor.setMinimumSize(180,40)
        #self.but_roi_convert.setMinimumSize(180,40)
        #self.but_reco_image .setMinimumSize(180,40)

        self.but_geo_in     .setStyleSheet(cp.styleButtonLeft)
        self.but_geo_out    .setStyleSheet(cp.styleButton)
        self.but_geo_img_nda.setStyleSheet(cp.styleButtonLeft)
        self.but_geo        .setStyleSheet(cp.styleButtonLeft)
        self.but_geo.setFixedHeight(80)
        self.edi_geo.setFixedHeight(80)

        #self.but_plot        .setFixedWidth(100)    
        #self.but_view        .setFixedWidth(100)    
        self.but_plot        .setIcon(cp.icon_monitor)
        self.but_view        .setIcon(cp.icon_table) # cp.icon_logviewer)

        #self.edi_roi_img.setFixedWidth(400)

        self.edi_geo_in      .setReadOnly(True)
        self.edi_geo_out     .setReadOnly(True)
        self.edi_geo_img_nda .setReadOnly(True)
        self.edi_geo         .setReadOnly(True)

        self.edi_geo_in      .setEnabled(False)
        self.edi_geo_out     .setEnabled(False)
        self.edi_geo_img_nda .setEnabled(False)

        self.lab_log_level.setStyleSheet(cp.styleLabel) 
        #self.lab_geo      .setStyleSheet(cp.styleLabel) 
        
        #self.edi_geometry    .setStyleSheet(cp.styleEditInfo)
        #self.edi_roi_img_nda .setStyleSheet(cp.styleEditInfo)
        #self.edi_roi_img     .setStyleSheet(cp.styleEditInfo)
        #self.edi_roi_mask_img.setStyleSheet(cp.styleEditInfo)
        #self.edi_roi_mask_nda.setStyleSheet(cp.styleEditInfo)

        self.but_plot.setVisible(False)


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self.name) 
        #self.frame.setGeometry(self.rect())
        pass


    def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', self.name)

        #try    : cp.guimain.close()
        #except : pass

        try    : cp.guifilebrowser.close()
        except : pass

        try    : cp.plotimgspe.close()
        except : pass

        cp.guialignment = None


    def onExit(self):
        logger.debug('onExit', self.name)
        self.close()


    def on_but_geo_in(self):
        logger.info('Select the "geometry" input file', __name__)
        self.set_file_name(self.edi_geo_in, self.fname_geo_in, mode='open')


    def on_but_geo_out(self):
        logger.info('Select the "geometry" output file', __name__)
        self.set_file_name(self.edi_geo_out, self.fname_geo_out, mode='save')


    def on_but_geo_img_nda(self):
        logger.info('Select file with ndarray for image', __name__)
        self.set_file_name(self.edi_geo_img_nda, self.fname_geo_img_nda, mode='open')


    def on_box_log_level(self):
        level_selected = self.box_log_level.currentText()
        self.geo_log_level.setValue(level_selected) 
        logger.info('on_box_log_level - selected level: ' + self.geo_log_level.value(), self.name)

        #self.box_txt.setText( logger.getLogContent() )

        self.set_command_window()

 
    def on_but_geo(self):
        #logger.info('Start geo', __name__)

        resp = gu.confirm_or_cancel_dialog_box(parent=self, text='Please confirm or cancel command', title='Confirm or cancel') 
        if resp :
            cmd = self.get_command()
            logger.info('Start app by the command: %s' % cmd, __name__)
            os.system(cmd)
        else :
            logger.info('Command is canceled', __name__)


    def set_file_name(self, edi, par, mode='save', filter='*.txt *.dat *.data\nAll files (*)'):
        #logger.debug('set_file_name', __name__)

        self.setStatus(1, 'Waiting for input of the file name...')
        
        path = str(edi.displayText())
        dname, fname = os.path.split(path)
        msg = 'Current dir: %s   file: %s' % (dname, fname)
        logger.info(msg, __name__)
        
        path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save file', path, filter=filter)) \
               if mode == 'save' else \
               str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', path, filter=filter))

        dname, fname = os.path.split(path)

        if dname == '' or fname == '' :
            logger.info('Input directiry name or file name is empty... use default values', __name__)
            return
        else :
            edi.setText(path)
            par.setValue(path)
            logger.info('Selected file:\n' + path, __name__)

        self.set_command_window()

        self.setStatus(0)


#    def select_file_for_plot(self):
#
#        list = [ self.fname_roi_img.value() \
#               , self.fname_roi_mask_img.value() \
#               , self.fname_roi_mask_nda.value() \
#               , self.fname_roi_mask_nda_tst.value() \
#                 ]
#        fname = gu.selectFromListInPopupMenu(list)
#        if fname is None :
#            return None
#
#        if not os.path.exists(fname) :
#            logger.warning('File %s does not exist. There is nothing to plot...' % fname, __name__)
#            return None
#
#        logger.info('Selected file for plot: %s' % fname, __name__)
#
#        return fname
#
#

    def on_but_plot(self):
        logger.info('TBD: on_but_plot', __name__)
#        try :
#            cp.plotimgspe.close()
#            try    : del cp.plotimgspe
#            except : pass
#
#        except :
#
#            ifname = self.select_file_for_plot()
#            if ifname is None :
#                return
#            
#            self.setStatus(1, 'Plot image from file %s' % os.path.basename(ifname))
#
#            ofname = os.path.join(fnm.path_dir_work(),'image.png')
#            tit = 'Plot for %s' % os.path.basename(ifname)            
#            cp.plotimgspe = PlotImgSpe(None, ifname=ifname, ofname=ofname, title=tit, is_expanded=False)
#            cp.plotimgspe.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
#            cp.plotimgspe.show()
#
#        self.setStatus(0)


    def on_but_view(self):
        logger.info('on_but_view', __name__)

        try    :
            cp.guifilebrowser.close()

        except :            
            list_of_fnames = [ self.fname_geo_in.value() \
                             , self.fname_geo_out.value() \
                             , self.fname_geo_img_nda.value() \
                               ] 

            cp.guifilebrowser = GUIFileBrowser(None, list_of_fnames, list_of_fnames[0])
            cp.guifilebrowser.move(self.pos().__add__(QtCore.QPoint(880,40))) # open window with offset w.r.t. parent
            cp.guifilebrowser.show()


    def set_command_window(self):
        cmd = self.get_command()
        logger.info('command: %s' % cmd, __name__)
        self.edi_geo.clear()
        self.edi_geo.setText(cmd)
        

    def get_command(self):

        cmd = 'geo -g %s -i %s -L %s' % (self.fname_geo_in.value(),
                                         self.fname_geo_img_nda.value(),
                                         self.geo_log_level.value())

        #if self.fname_geo_out.value() is not '' :        
        #    cmd += ' -o %s'% self.fname_geo_out.value()

        return cmd
    

    def setStatus(self, status_index=0, msg='Waiting for the next command'):
        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0 : self.lab_status.setStyleSheet(cp.styleStatusGood)
        if status_index == 1 : self.lab_status.setStyleSheet(cp.styleStatusWarning)
        if status_index == 2 : self.lab_status.setStyleSheet(cp.styleStatusAlarm)

        #self.lab_status.setText('Status: ' + list_of_states[status_index] + msg)
        self.lab_status.setText(msg)

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIAlignment()
    ex.show()
    app.exec_()
#-----------------------------
