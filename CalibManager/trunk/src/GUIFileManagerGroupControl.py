
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIFileManagerGroupControl...
#
#------------------------------------------------------------------------

"""Renders the main GUI for the CalibManager.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id:$

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
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersForApp import cp

from Logger               import logger
from FileNameManager      import fnm

from CorAna.MaskEditor import MaskEditor
import GlobalUtils     as     gu
import RegDBUtils      as     ru
from GUIFileBrowser         import *
from PlotImgSpe             import *
from FileDeployer           import fd

#---------------------
#  Class definition --
#---------------------
class GUIFileManagerGroupControl ( QtGui.QWidget ) :
    """Main GUI for main button bar.

    @see BaseClass
    @see OtherClass
    """
    char_expand    = u' \u25BC' # down-head triangle
    #char_expand    = '' # down-head triangle

    def __init__ (self, parent=None, app=None) :

        self.name = 'GUIFileManagerGroupControl'
        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(10, 25, 120, 300)
        self.setWindowTitle('File Manager Select & Action GUI')
        #self.setWindowIcon(cp.icon_monitor)
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        self.setFrame()

        #cp.setIcons()

        self.setParams()
 
        self.lab_from       = QtGui.QLabel('for run range')
        self.lab_to         = QtGui.QLabel(':')
        self.edi_from       = QtGui.QLineEdit  ( self.str_run_from )
        self.edi_to         = QtGui.QLineEdit  ( self.str_run_to )

        self.edi_from.setValidator(QtGui.QIntValidator(0,9999,self))
        self.edi_to  .setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]\\d{0,3}|end$"),self))
 
        self.but_move   = QtGui.QPushButton('-> Move ->')
        self.but_copy   = QtGui.QPushButton('-> Copy -> ')
        self.but_delete = QtGui.QPushButton('Delete')
        self.but_list   = QtGui.QPushButton('<- List')
        self.but_view   = QtGui.QPushButton('<- View')
        self.but_plot   = QtGui.QPushButton('<- Plot')
        #self.but_copy  .setIcon(cp.icon_monitor)

        self.hboxC = QtGui.QHBoxLayout() 
        #self.hboxC.addStretch(1)     
        self.hboxC.addWidget( self.edi_from )
        self.hboxC.addWidget( self.lab_to   )
        self.hboxC.addWidget( self.edi_to   )
        self.hboxC.addStretch(1)     

        self.vboxW = QtGui.QVBoxLayout() 
        self.vboxW.addStretch(1)
        self.vboxW.addWidget( self.but_list   )
        self.vboxW.addWidget( self.but_view   )
        self.vboxW.addWidget( self.but_plot   )
        self.vboxW.addWidget( self.but_delete )
        self.vboxW.addStretch(1)
        self.vboxW.addWidget( self.but_move )
        self.vboxW.addWidget( self.but_copy )
        self.vboxW.addWidget( self.lab_from )
        self.vboxW.addLayout( self.hboxC ) 
        self.vboxW.addStretch(1)
        
        self.setLayout(self.vboxW)

        self.connect( self.but_move,   QtCore.SIGNAL('clicked()'), self.onButMove ) 
        self.connect( self.but_copy,   QtCore.SIGNAL('clicked()'), self.onButCopy ) 
        self.connect( self.but_list,   QtCore.SIGNAL('clicked()'), self.onButList ) 
        self.connect( self.but_view,   QtCore.SIGNAL('clicked()'), self.onButView ) 
        self.connect( self.but_plot,   QtCore.SIGNAL('clicked()'), self.onButPlot ) 
        self.connect( self.but_delete, QtCore.SIGNAL('clicked()'), self.onButDelete ) 
        self.connect( self.edi_from,   QtCore.SIGNAL('editingFinished()'), self.onEdiFrom )
        self.connect( self.edi_to,     QtCore.SIGNAL('editingFinished()'), self.onEdiTo )
  
        self.showToolTips()
        self.setStyle()

        cp.guifilemanagercontrol = self
        self.move(10,25)
        
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        #pass
        self.but_move  .setToolTip('Move selected file')
        self.but_copy  .setToolTip('Copy selected file')
        self.but_list  .setToolTip('List checked files in log')
        self.but_view  .setToolTip('Launch file browser')
        self.but_plot  .setToolTip('Launch plot browser')
        self.but_delete.setToolTip('Delete selected file\nDelete  is allowed for\nWORK or CALIB directories only')
        self.edi_from  .setToolTip('Enter run number in range [0,9999]')
        self.edi_to    .setToolTip('Enter run number in range [0,9999] or "end"')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setStyle(self):
        self.          setStyleSheet(cp.styleBkgd)
        self.but_move  .setStyleSheet(cp.styleButton)
        self.but_copy  .setStyleSheet(cp.styleButton)
        self.but_delete.setStyleSheet(cp.styleButton)
        self.but_list  .setStyleSheet(cp.styleButton)
        self.but_view  .setStyleSheet(cp.styleButton)
        self.but_plot  .setStyleSheet(cp.styleButton)
        
        self.setMinimumSize(120, 200)
        self.setFixedWidth(120)
        self.setContentsMargins (QtCore.QMargins(0,-9,0,-9))

        self.edi_from.setFixedWidth(40)
        self.edi_to  .setFixedWidth(40)

        self.lab_from  .setStyleSheet(cp.styleLabel)
        self.lab_to    .setStyleSheet(cp.styleLabel)
 
        self.setStyleButtons()


    def setStyleButtons(self):
        """Set buttons enabled or disabled depending on status of other fields
        """

        self.but_move.setVisible(False)

        file_is_enable = True

        self.but_list.setEnabled(file_is_enable)
        self.but_view.setEnabled(file_is_enable)
        self.but_plot.setEnabled(file_is_enable)

        is_enable_delete = file_is_enable 

        self.but_delete.setEnabled(is_enable_delete)

        is_enable_copy = file_is_enable \
                         and self.source_name != 'Select' \
                         and self.calib_type != 'Select' 
        self.but_move  .setEnabled(is_enable_copy)
        #self.but_copy  .setEnabled(is_enable_copy)
        
 
    def setParams(self) :
        self.str_run_from     = '0'
        self.str_run_to       = 'end'
        self.source_name      = 'Select'
        self.calib_type       = 'Select'


    def resetFields(self) :
        self.setParams()
        self.edi_from  .setText(self.str_run_from)
        self.edi_to    .setText(self.str_run_to)
        self.setStyleButtons()


    def resetFieldsOnDelete(self) :
        self.setStyleButtons()


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self.name) 
        self.frame.setGeometry(self.rect())
        #print 'GUIFileManagerGroupControl resizeEvent: %s' % str(self.size())


    def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', self.name)

        try    : cp.guifilebrowser.close()
        except : pass

        try    : cp.plotimgspe.close()
        except : pass

        cp.guifilemanagergroupcontrol = None


    def onExit(self):
        logger.debug('onExit', self.name)
        self.close()


    def onButDelete(self):
        logger.warning('onButDelete - Please use the "Single File" tab to delete file(s)', __name__)
        return
            

    def onButMove(self):
        logger.warnind('onButMove - Please use the "Single File" tab to move file(s)', __name__)
        return


    def onButCopy(self):
        logger.info('onButCopy - in implementation', __name__)
        #return
    
        list_of_cmds = self.list_of_group_copy_cmds()

        if list_of_cmds == [] :
            logger.info('\nThe list of commands IS EMPTY ! Click on check-box of desired file(s).', __name__)
            return

        txt = '\nList of commands for tentetive file deployment:'
        for cmd in list_of_cmds :
            txt += '\n' + cmd
        logger.info(txt, __name__)

        msg = 'Approve commands \njust printed in the logger'
        if self.approveCommand(self.but_copy, msg) :

            for cmd in list_of_cmds :
                fd.procDeployCommand(cmd, 'group-file-manager')
            #    #os.system(cmd)

            if cp.guistatus is not None : cp.guistatus.updateStatusInfo()


    def approveCommand(self, but, msg):
        resp = gu.confirm_or_cancel_dialog_box(parent=but, text=msg, title='Please confirm or cancel!')
        if resp : logger.info(msg, __name__)
        return resp


    def list_of_group_copy_cmds(self):

        list_of_fnames = cp.guidirtree.get_list_of_checked_item_names()

        dst_calib_dir = fnm.path_to_calib_dir()
        dst_fname = '%s-%s.data' % (self.str_run_from, self.str_run_to)

        #print 'dst_calib_dir:', dst_calib_dir
        #print 'dst_fname:', dst_fname

        list_of_cmds = []

        for fname in list_of_fnames :
            #print '   split fname', fname
            fields = fname.split('/')
            if len(fields) < 5 :
                logger.info('File %s has a un-expected path: ' % fname, __name__)
                continue
            if  fields[-5] != 'calib' :
                logger.info('File %s is not from "calib" directory: ' % fname, __name__)
                continue
                
            dst_path = os.path.join(dst_calib_dir, fields[-4], fields[-3], fields[-2], dst_fname)
            cmd = 'cp %s %s' % (fname, dst_path)
            list_of_cmds.append(cmd)

        return list_of_cmds


    def onEdiFrom(self):
        logger.debug('onEdiFrom', __name__ )
        self.str_run_from = str( self.edi_from.displayText() )        
        msg = 'Set the run validity range from %s' % self.str_run_from
        logger.info(msg, __name__ )


    def onEdiTo(self):
        logger.debug('onEdiTo', __name__ )
        self.str_run_to = str( self.edi_to.displayText() )        
        msg = 'Set the run validity range up to %s' % self.str_run_to
        logger.info(msg, __name__ )


    def get_detector_selected(self):
        lst = cp.list_of_dets_selected()
        len_lst = len(lst)
        msg = '%d detector(s) selected: %s' % (len_lst, str(lst))
        #logger.info(msg, __name__ )

        if len_lst !=1 :
            msg += ' Select THE ONE!'
            logger.warning(msg, __name__ )
            return None

        return lst[0]


    def onButList(self):
        if cp.dirtreemodel is None : return
        list_of_fnames = cp.guidirtree.get_list_of_checked_item_names()

        txt = '\nList of checked files'

        if list_of_fnames == [] : txt += ' IS EMPTY !' 
        for fname in list_of_fnames:
            txt += '\n' + fname
            
        logger.info(txt, __name__)

    
    def onButView(self):
        #self.exportLocalPars()
        logger.info('onButView', __name__)

        try    :
            cp.guifilebrowser.close()

        except :            
            list_of_fnames = cp.guidirtree.get_list_of_checked_item_names()
            if list_of_fnames == [] :
                logger.info('\nThe list of checked files IS EMPTY ! Click on check-box of desired file(s).', __name__)
                return

            cp.guifilebrowser = GUIFileBrowser(None, list_of_fnames, list_of_fnames[0])
            cp.guifilebrowser.move(self.pos().__add__(QtCore.QPoint(880,40))) # open window with offset w.r.t. parent
            cp.guifilebrowser.show()


    def onButPlot(self):
        logger.info('onButPlot', __name__)

        try :
            cp.plotimgspe.close()
            cp.plotimgspe = None

        except :
            list_of_fnames = cp.guidirtree.get_list_of_checked_item_names()
            if list_of_fnames == [] :
                logger.info('\nThe list of checked files is empty IS EMPTY ! Click on check-box of desired file(s).', __name__)
                return

            fname = list_of_fnames[0]
            if len(list_of_fnames) > 1 :
                fname = gu.selectFromListInPopupMenu(list_of_fnames)

            if fname is None or fname == '' : return

            msg = 'Selected file to plot: %s' % fname
            logger.info(msg, __name__)
            #print msg

            ifname = fname
            ofname = os.path.join(fnm.path_dir_work(),'image.png')
            tit = 'Plot for %s' % os.path.basename(ifname)            
            cp.plotimgspe = PlotImgSpe(None, ifname=ifname, ofname=ofname, title=tit, load_is_visible=True)
            cp.plotimgspe.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.plotimgspe.show()


#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIFileManagerGroupControl()
    ex.show()
    app.exec_()
#-----------------------------
