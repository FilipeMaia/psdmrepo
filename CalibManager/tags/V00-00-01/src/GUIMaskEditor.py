
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

import matplotlib
#matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
if matplotlib.get_backend() != 'Qt4Agg' : matplotlib.use('Qt4Agg')

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersForApp import cp

from Logger               import logger
from FileNameManager      import fnm

from CorAna.MaskEditor import MaskEditor

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

        cp.setIcons()

        self.setGeometry(10, 25, 650, 30)
        self.setWindowTitle('Mask Editor Control')
        #self.setWindowIcon(cp.icon_monitor)
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        self.setFrame()
 
        self.butMaskEditor  = QtGui.QPushButton('Open Mask Editor')
        self.butMaskEditor  .setIcon(cp.icon_monitor)

        self.hboxB = QtGui.QHBoxLayout() 
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.butMaskEditor     )
        self.hboxB.addStretch(1)     

        self.vboxW = QtGui.QHBoxLayout() 
        self.vboxW.addStretch(1)
        self.vboxW.addLayout(self.hboxB) 
        self.vboxW.addStretch(1)
        
        self.setLayout(self.vboxW)

        self.connect( self.butMaskEditor, QtCore.SIGNAL('clicked()'), self.onButMaskEditor )

        self.showToolTips()
        self.setStyle()

        cp.guimaskeditor = self
        self.move(10,25)
        
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        pass
        #self.butSave.setToolTip('Save all current settings in the \nfile with configuration parameters') 


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
        self.butMaskEditor.setMinimumHeight(60)

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



    def onExit(self):
        logger.debug('onExit', self.name)
        self.close()

        
    def onFile(self):
        logger.debug('onFile', self.name)
        path  = fnm.path_gui_image()
        #dir, fname = os.path.split(path)
        path  = str( QtGui.QFileDialog.getSaveFileName(self,
                                                       caption='Select file to save the GUI',
                                                       directory = path,
                                                       filter = '*.png'
                                                       ) )
        if path == '' :
            logger.debug('Saving is cancelled.', self.name)
            return
        logger.info('Save GUI image in file: ' + path, self.name)
        pixmap = QtGui.QPixmap.grabWidget(self)
        status = pixmap.save(path, 'PNG')
        #logger.info('Save status: '+str(status), self.name)


#    def onELog(self):
#        logger.debug('onELog', self.name)
#        pixmap = QtGui.QPixmap.grabWidget(self)
#        fname  = fnm.path_gui_image()
#        status = pixmap.save(fname, 'PNG')
#        logger.info('1. Save GUI image in file: ' + fname + ' status: '+str(status), self.name)
#        if not status : return
#        logger.info('2. Send GUI image in ELog: ', fname)
#        wdialog = GUIELogPostingDialog (self, fname=fname)
#        resp=wdialog.exec_()
  

    def dictOfMaskEditorPars (self):       
        pars = {'parent' : None, 
                'arr'    : None, 
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

        print 'Start MaskEditor with input parameters:'
        for k,v in pars.items():
            print '%9s : %s' % (k,v)

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

            pars = self.dictOfMaskEditorPars ()
            cp.maskeditor = MaskEditor(**pars)
            cp.maskeditor.move(self.pos().__add__(QtCore.QPoint(820,-7))) # open window with offset w.r.t. parent
            cp.maskeditor.show()

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIMaskEditor()
    ex.show()
    app.exec_()
#-----------------------------
