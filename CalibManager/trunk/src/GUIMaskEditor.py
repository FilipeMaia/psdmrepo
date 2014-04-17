
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMaskEditor...
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
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

# For self-run debugging:
if __name__ == "__main__" :
    import matplotlib
    matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)

#import matplotlib
#if matplotlib.get_backend() != 'Qt4Agg' : matplotlib.use('Qt4Agg')

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

#---------------------
#  Class definition --
#---------------------
class GUIMaskEditor ( QtGui.QWidget ) :
    """Main GUI for main button bar.

    @see BaseClass
    @see OtherClass
    """
    def __init__ (self, parent=None, app=None) :

        self.name = 'GUIMaskEditor'
        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.fname_prefix  = cp.fname_prefix
        self.path_mask_img = cp.path_mask_img
        self.img_arr = None

        cp.setIcons()

        self.setGeometry(10, 25, 650, 30)
        self.setWindowTitle('Mask Editor Control')
        #self.setWindowIcon(cp.icon_monitor)
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        self.setFrame()
 
        self.titFile = QtGui.QLabel('File:')
        self.lab_status = QtGui.QLabel('Status: ')

        self.ediFile = QtGui.QLineEdit ( self.path_mask_img.value() )
        self.ediFile.setReadOnly(True)
 
        self.butMaskEditor  = QtGui.QPushButton('Open Mask Editor')
        self.butMaskEditor  .setIcon(cp.icon_monitor)

        self.butBrowse = QtGui.QPushButton( 'Browse' )

        self.hboxB = QtGui.QHBoxLayout() 
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget( self.titFile )
        self.hboxB.addWidget( self.ediFile )
        self.hboxB.addWidget( self.butBrowse )
        self.hboxB.addStretch(1)     

        self.hboxE = QtGui.QHBoxLayout() 
        self.hboxE.addStretch(1)     
        self.hboxE.addWidget( self.butMaskEditor )
        self.hboxE.addStretch(1)     

        self.hboxS = QtGui.QHBoxLayout()
        self.hboxS.addWidget(self.lab_status)
        self.hboxS.addStretch(1)     

        self.vboxW = QtGui.QVBoxLayout() 
        self.vboxW.addStretch(1)
        self.vboxW.addLayout( self.hboxB ) 
        self.vboxW.addLayout( self.hboxE ) 
        self.vboxW.addStretch(1)
        self.vboxW.addLayout( self.hboxS ) 
        
        self.setLayout(self.vboxW)

        self.connect( self.butMaskEditor, QtCore.SIGNAL('clicked()'), self.onButMaskEditor )
        self.connect( self.butBrowse,     QtCore.SIGNAL('clicked()'), self.onButBrowse     ) 
 
        self.showToolTips()
        self.setStyle()

        self.setStatus(0, 'Status: operations with ROI mask editor')

        cp.guimaskeditor = self
        self.move(10,25)
        
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        #pass
        self.ediFile.setToolTip('Path to the file with image data') 
        self.butBrowse.setToolTip('Open file browser dialog window \nand select the file with image data') 
        self.butMaskEditor.setToolTip('Open/Close Mask Editor window')

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)


    def setStyle(self):
        self.              setStyleSheet(cp.styleBkgd)
        self.butMaskEditor.setStyleSheet(cp.styleButton)
        #self.butMaskEditor.setFixedWidth(200)
        #self.butMaskEditor.setMinimumHeight(60)
        self.butMaskEditor.setMinimumSize(180,60)

        self.ediFile.setFixedWidth(400)
        self.ediFile.setStyleSheet(cp.styleEditInfo) 
        self.ediFile.setEnabled(False)            

 
        #self.butFBrowser.setVisible(False)
        #self.butSave.setText('')
        #self.butExit.setText('')
        #self.butExit.setFlat(True)


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self.name) 
        self.frame.setGeometry(self.rect())


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

        #try    : del cp.guimain
        #except : pass

        cp.guimaskeditor = None


    def onExit(self):
        logger.debug('onExit', self.name)
        self.close()


    def onButBrowse(self):
        logger.debug('onButBrowse', __name__)
        self.path = str( self.ediFile.displayText() )
        self.dname, self.fname = os.path.split(self.path)
        msg = 'dir : %s   file : %s' % (self.dname, self.fname)
        logger.info(msg, __name__)
        prefix = self.fname_prefix.value()
        file_filter = 'Text files (' + prefix + '*.txt ' + prefix + '*.dat)\nAll files (*)'
        self.path = str( QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.dname, filter=file_filter) )
        self.dname, self.fname = os.path.split(self.path)

        if self.dname == '' or self.fname == '' :
            logger.info('Input directiry name or file name is empty... use default values', __name__)
            return
        else :
            self.ediFile.setText(self.path)
            self.path_mask_img.setValue(self.path)
            logger.info('Selected file for mask editor:\n' + self.path, __name__)



    def openFileWithImageArray(self) :
         #self.path = str( self.ediFile.displayText() )
         self.path = self.path_mask_img.value()
         self.img_arr = gu.get_image_array_from_file(self.path) # , dtype=np.float32)
         #print 'openFileWithImageArray: self.arr.shape:', self.img_arr.shape
         #print self.img_arr


    def dictOfMaskEditorPars (self):       
        pars = {'parent' : None, 
                'arr'    : self.img_arr, 
                'xyc'    : None, # xyc=(opts.xc,opts.yc)
                'ifname' : 'ifname', 
                'ofname' : 'ofname', 
                'mfname' : 'mfname',
                'title'  : 'Mask Editor', 
                'lw'     : 1, 
                'col'    : 'b',
                'picker' : 8,
                'verb'   : True,
                'ccd_rot': 0, 
                'updown' : False}

        #print 'Start MaskEditor with input parameters:'
        #for k,v in pars.items():
        #    print '%9s : %s' % (k,v)

        return pars


    def onButMaskEditor  (self):       
        logger.debug('onLogger', self.name)
        try    :
            cp.maskeditor.close()
            del cp.maskeditor
            self.butMaskEditor.setStyleSheet(cp.styleButton)
            self.butMaskEditor.setText('Open Mask Editor')
        except :
            self.butMaskEditor.setStyleSheet(cp.styleButtonGood)
            self.butMaskEditor.setText('Close Mask Editor')

            self.openFileWithImageArray()

            pars = self.dictOfMaskEditorPars ()
            cp.maskeditor = MaskEditor(**pars)
            cp.maskeditor.move(self.pos().__add__(QtCore.QPoint(820,-7))) # open window with offset w.r.t. parent
            cp.maskeditor.show()

    def setStatus(self, status_index=0, msg=''):
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
    ex  = GUIMaskEditor()
    ex.show()
    app.exec_()
#-----------------------------
