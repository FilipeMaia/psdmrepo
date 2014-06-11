
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMain...
#
#------------------------------------------------------------------------

"""Renders the main GUI for the image time-correlation analysis.

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

from PyQt4 import QtGui, QtCore
import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersCorAna import confpars as cp

#from GUIConfigParameters import * 
from GUIFiles             import *
from GUISetupInfo         import *
from GUIAnaSettings       import *
from GUISystemSettings    import *
from GUIIntensityMonitors import *
from GUIRun               import *
from GUIViewResults       import *
from GUILogger            import *
from Logger               import logger
from FileNameManager      import fnm
from GUIFileBrowser       import *

#---------------------
#  Class definition --
#---------------------
class GUIMain ( QtGui.QWidget ) :
    """Main GUI for the interactive analysis project.

    @see BaseClass
    @see OtherClass
    """
    def __init__ (self, parent=None, app=None) :

        self.name = 'GUIMain'
        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(10, 25, 150, 500)
        self.setWindowTitle('Interactive Analysis')
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        self.setFrame()
 
        self.titControl     = QtGui.QLabel('Control Panel')
        self.butFiles       = QtGui.QPushButton('Files')    
        self.butBatchInfo   = QtGui.QPushButton('Setup Info')    
        self.butAnaSettings = QtGui.QPushButton('Analysis Settings')
        self.butSystem      = QtGui.QPushButton('System')
        self.butIntMon      = QtGui.QPushButton('Intensity Monitors')
        self.butRun         = QtGui.QPushButton('Run')
        self.butViewResults = QtGui.QPushButton('View Results')
        self.butStop        = QtGui.QPushButton('Stop')
        self.butSave        = QtGui.QPushButton('Save')
        self.butExit        = QtGui.QPushButton('Exit')
        self.butLogger      = QtGui.QPushButton('Logger')
        self.butFBrowser    = QtGui.QPushButton('File Viewer')

        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addWidget(self.titControl    )
        self.vbox.addWidget(self.butFiles      )
        self.vbox.addWidget(self.butBatchInfo  )
        self.vbox.addWidget(self.butAnaSettings)
        self.vbox.addWidget(self.butSystem     )
        self.vbox.addWidget(self.butIntMon     )
        self.vbox.addWidget(self.butRun        )
        self.vbox.addWidget(self.butViewResults)
        self.vbox.addStretch(1)     
        self.vbox.addWidget(self.butLogger     )
        self.vbox.addWidget(self.butFBrowser   )
        self.vbox.addStretch(1)     
        self.vbox.addWidget(self.butStop       )
        self.vbox.addWidget(self.butSave       )
        self.vbox.addWidget(self.butExit       )

        self.setLayout(self.vbox)

        self.connect(self.butFiles      ,  QtCore.SIGNAL('clicked()'), self.onFiles   )
        self.connect(self.butBatchInfo  ,  QtCore.SIGNAL('clicked()'), self.onBatchInfo   )
        self.connect(self.butAnaSettings,  QtCore.SIGNAL('clicked()'), self.onAnaSettings )
        self.connect(self.butSystem     ,  QtCore.SIGNAL('clicked()'), self.onSystem      )
        self.connect(self.butIntMon     ,  QtCore.SIGNAL('clicked()'), self.onIntMon      )
        self.connect(self.butRun        ,  QtCore.SIGNAL('clicked()'), self.onRun         )
        self.connect(self.butViewResults,  QtCore.SIGNAL('clicked()'), self.onViewResults )
        self.connect(self.butStop       ,  QtCore.SIGNAL('clicked()'), self.onStop        )
        self.connect(self.butSave       ,  QtCore.SIGNAL('clicked()'), self.onSave        )
        self.connect(self.butExit       ,  QtCore.SIGNAL('clicked()'), self.onExit        )
        self.connect(self.butLogger     ,  QtCore.SIGNAL('clicked()'), self.onLogger      )
        self.connect(self.butFBrowser   ,  QtCore.SIGNAL('clicked()'), self.onFBrowser    )

        self.showToolTips()
        self.setStyle()
        self.printStyleInfo()

        self.onLogger()

        gu.create_directory(cp.dir_work.value())

        cp.guimain = self
        self.move(10,25)
        
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------

    def printStyleInfo(self):
        qstyle     = self.style()
        qpalette   = qstyle.standardPalette()
        qcolor_bkg = qpalette.color(1)
        #r,g,b,alp  = qcolor_bkg.getRgb()
        msg = 'Background color: r,g,b,alpha = %d,%d,%d,%d' % ( qcolor_bkg.getRgb() )
        logger.debug(msg)


    def showToolTips(self):
        self.butSave.setToolTip('Save all current settings in the \nfile with configuration parameters.') 
        self.butExit.setToolTip('Close all windows and \nexit this program') 


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.               setStyleSheet(cp.styleBkgd)
        self.titControl    .setStyleSheet(cp.styleTitle)
        self.butFiles      .setStyleSheet(cp.styleButton)
        self.butBatchInfo  .setStyleSheet(cp.styleButton) 
        self.butAnaSettings.setStyleSheet(cp.styleButton)
        self.butSystem     .setStyleSheet(cp.styleButton)
        self.butIntMon     .setStyleSheet(cp.styleButton)
        self.butRun        .setStyleSheet(cp.styleButton)
        self.butViewResults.setStyleSheet(cp.styleButton)
        self.butStop       .setStyleSheet(cp.styleButton)
        #self.butLogger     .setStyleSheet(cp.styleGreenish)
        self.butFBrowser   .setStyleSheet(cp.styleButton)
        self.butSave       .setStyleSheet(cp.styleButton)
        self.butExit       .setStyleSheet(cp.styleButton)
        self.titControl    .setAlignment(QtCore.Qt.AlignCenter)

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

        if cp.res_save_log : 
            logger.saveLogInFile     ( fnm.log_file() )
            logger.saveLogTotalInFile( fnm.log_file_total() )

        try    : cp.guifiles.close()
        except : pass

        try    : cp.guisetupinfo.close()
        except : pass

        try    : cp.guianasettings.close()
        except : pass

        try    : cp.guisystemsettings.close()
        except : pass

        try    : cp.guiintensitymonitors.close()
        except : pass

        try    : cp.guiviewresults.close()
        except : pass

        try    : cp.guirun.close()
        except : pass

        try    : cp.guilogger.close()
        except : pass

        try    : cp.guifilebrowser.close()
        except : pass

        try    : del cp.guimain
        except : pass


    def onExit(self):
        logger.debug('onExit', self.name)
        self.close()
        
    def onPrint(self):
        logger.debug('onPrint', self.name)
        
    def onFiles(self):
        logger.debug('onFiles', self.name)
        try :
            cp.guifiles.close()
            self.butFiles.setStyleSheet(cp.styleButton)
        except :
            self.butFiles.setStyleSheet(cp.styleButtonOn)
            cp.guifiles = GUIFiles()
            cp.guifiles.move(self.pos().__add__(QtCore.QPoint(160,60))) # open window with offset w.r.t. parent
            cp.guifiles.show()


    def onBatchInfo(self):
        logger.debug('onBatchInfo', self.name)
        try :
            cp.guisetupinfo.close()
        except :
            cp.guisetupinfo = GUISetupInfo()
            cp.guisetupinfo.setParent(self)
            cp.guisetupinfo.move(self.pos().__add__(QtCore.QPoint(160,90))) # open window with offset w.r.t. parent
            cp.guisetupinfo.show()


    def onSave(self):
        logger.debug('onSave', self.name)
        cp.saveParametersInFile( cp.fname_cp.value() )


    def onAnaSettings(self):    
        logger.debug('onAnaSettings', self.name)
        try :
            cp.guianasettings.close()
        except :
            cp.guianasettings = GUIAnaSettings()
            cp.guianasettings.move(self.pos().__add__(QtCore.QPoint(160,130))) # open window with offset w.r.t. parent
            cp.guianasettings.show()


    def onSystem(self):     
        logger.debug('onSystem', self.name)
        try    :
            cp.guisystemsettings.close()
        except :
            cp.guisystemsettings = GUISystemSettings()
            cp.guisystemsettings.move(self.pos().__add__(QtCore.QPoint(160,160))) # open window with offset w.r.t. parent
            cp.guisystemsettings.show()


    def onIntMon(self):     
        logger.debug('onIntMon', self.name)
        try    :
            cp.guiintensitymonitors.close()
        except :
            cp.guiintensitymonitors = GUIIntensityMonitors()
            cp.guiintensitymonitors.move(self.pos().__add__(QtCore.QPoint(160,160))) # open window with offset w.r.t. parent
            cp.guiintensitymonitors.show()


    def onRun (self):       
        logger.debug('onRun', self.name)
        try    :
            cp.guirun.close()
        except :
            cp.guirun = GUIRun()
            cp.guirun.move(self.pos().__add__(QtCore.QPoint(160,195))) # open window with offset w.r.t. parent
            cp.guirun.show()


    def onViewResults(self):
        logger.debug('onViewResults', self.name)
        try    :
            cp.guiviewresults.close()
        except :
            cp.guiviewresults = GUIViewResults()
            cp.guiviewresults.move(self.pos().__add__(QtCore.QPoint(160,230))) # open window with offset w.r.t. parent
            cp.guiviewresults.show()


    def onLogger (self):       
        logger.debug('onLogger', self.name)
        try    :
            cp.guilogger.close()
        except :
            self.butLogger.setStyleSheet(cp.styleButtonGood)
            cp.guilogger = GUILogger()
            cp.guilogger.move(self.pos().__add__(QtCore.QPoint(200,0))) # open window with offset w.r.t. parent
            cp.guilogger.show()


    def onFBrowser (self):       
        logger.debug('onFBrowser', self.name)
        try    :
            cp.guifilebrowser.close()
        except :
            cp.guifilebrowser = GUIFileBrowser(None, fnm.get_list_of_files_total())
            cp.guifilebrowser.move(self.pos().__add__(QtCore.QPoint(240,40))) # open window with offset w.r.t. parent
            cp.guifilebrowser.show()

    def onStop(self):       
        logger.debug('onStop - not implemented yet...', self.name)

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
    #def mousePressEvent(self, event):
    #    print 'event.x, event.y, event.button =', str(event.x()), str(event.y()), str(event.button())         

    #def mouseReleaseEvent(self, event):
    #    print 'event.x, event.y, event.button =', str(event.x()), str(event.y()), str(event.button())                

#http://doc.qt.nokia.com/4.6/qt.html#Key-enum
    def keyPressEvent(self, event):
        #print 'event.key() = %s' % (event.key())
        if event.key() == QtCore.Qt.Key_Escape:
            #self.close()
            self.SHowIsOn = False    
            pass

        if event.key() == QtCore.Qt.Key_B:
            #print 'event.key() = %s' % (QtCore.Qt.Key_B)
            pass

        if event.key() == QtCore.Qt.Key_Return:
            #print 'event.key() = Return'
            pass

        if event.key() == QtCore.Qt.Key_Home:
            #print 'event.key() = Home'
            pass

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIMain()
    ex.show()
    app.exec_()
#-----------------------------
