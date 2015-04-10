
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMainTB...
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
from GUIELogPostingDialog import *

#---------------------
#  Class definition --
#---------------------
class GUIMainTB ( QtGui.QWidget ) :
    """Main GUI for the interactive analysis project.

    @see BaseClass
    @see OtherClass
    """
    def __init__ (self, parent=None, app=None) :

        self.name = 'GUIMainTB'
        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        cp.setIcons()

        self.setGeometry(10, 25, 850, 300)
        self.setWindowTitle('Data Processing Environment')
        self.setWindowIcon(cp.icon_monitor)
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        self.setFrame()
 
        #self.titControl     = QtGui.QLabel('Control Panel')
        #self.butFiles       = QtGui.QPushButton('Files')    
        #self.butBatchInfo   = QtGui.QPushButton('Batch Information')    
        #self.butAnaSettings = QtGui.QPushButton('Analysis Settings')
        #self.butSystem      = QtGui.QPushButton('System')
        #self.butRun         = QtGui.QPushButton('Run')
        #self.butViewResults = QtGui.QPushButton('View Results')
        #self.butStop        = QtGui.QPushButton('Stop')
        self.butSave        = QtGui.QPushButton('Save')
        self.butExit        = QtGui.QPushButton('Exit')
        self.butFile        = QtGui.QPushButton(u'GUI \u2192 &File')
        self.butELog        = QtGui.QPushButton(u'GUI \u2192 &ELog')
        self.butLogger      = QtGui.QPushButton('Logger')
        self.butFBrowser    = QtGui.QPushButton('File Viewer')

        self.butELog    .setIcon(cp.icon_mail_forward)
        self.butFile    .setIcon(cp.icon_save)
        self.butExit    .setIcon(cp.icon_exit)
        self.butLogger  .setIcon(cp.icon_logger)
        self.butFBrowser.setIcon(cp.icon_browser)
        self.butSave    .setIcon(cp.icon_save_cfg)
        #self.butStop    .setIcon(cp.icon_stop)

        self.hboxW = QtGui.QHBoxLayout() 

        self.vboxW = QtGui.QVBoxLayout() 
        self.vboxW.addStretch(1)
        self.vboxW.addLayout(self.hboxW) 
        self.vboxW.addStretch(1)

        self.hboxWW= QtGui.QHBoxLayout() 
        self.hboxWW.addStretch(1)
        self.hboxWW.addLayout(self.vboxW) 
        self.hboxWW.addStretch(1)


        self.list_of_tabs = ['Files',
                             'Setup Info',
                             'Analysis Settings',
                             'System',
                             'Intensity Monitors',
                             'Run',
                             'View Results']

        #print 'number of tabs:', len(self.list_of_tabs)

        self.makeTabBar()
        self.guiSelector()

        self.hboxB = QtGui.QHBoxLayout() 
        self.hboxB.addWidget(self.butLogger     )
        self.hboxB.addWidget(self.butFBrowser   )
        self.hboxB.addWidget(self.butFile       )
        self.hboxB.addWidget(self.butELog       )
        self.hboxB.addStretch(1)     
        #self.hboxB.addWidget(self.butStop       )
        self.hboxB.addWidget(self.butSave       )
        self.hboxB.addWidget(self.butExit       )

        self.vbox = QtGui.QVBoxLayout() 
        #self.vbox.addWidget(self.titControl    )
        #self.vbox.addWidget(self.butFiles      )
        #self.vbox.addWidget(self.butBatchInfo  )
        #self.vbox.addWidget(self.butAnaSettings)
        #self.vbox.addWidget(self.butSystem     )
        #self.vbox.addWidget(self.butRun        )
        #self.vbox.addWidget(self.butViewResults)
        #self.vbox.addStretch(2)
        self.vbox.addLayout(self.hboxB)
        self.vbox.addWidget(self.tab_bar)
        self.vbox.addLayout(self.hboxWW)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        #self.connect(self.butFiles      ,  QtCore.SIGNAL('clicked()'), self.onFiles   )
        #self.connect(self.butBatchInfo  ,  QtCore.SIGNAL('clicked()'), self.onBatchInfo   )
        #self.connect(self.butAnaSettings,  QtCore.SIGNAL('clicked()'), self.onAnaSettings )
        #self.connect(self.butSystem     ,  QtCore.SIGNAL('clicked()'), self.onSystem      )
        #self.connect(self.butRun        ,  QtCore.SIGNAL('clicked()'), self.onRun         )
        #self.connect(self.butViewResults,  QtCore.SIGNAL('clicked()'), self.onViewResults )
        #self.connect(self.butStop       ,  QtCore.SIGNAL('clicked()'), self.onStop        )
        self.connect(self.butSave       ,  QtCore.SIGNAL('clicked()'), self.onSave        )
        self.connect(self.butExit       ,  QtCore.SIGNAL('clicked()'), self.onExit        )
        self.connect(self.butELog       ,  QtCore.SIGNAL('clicked()'), self.onELog        )
        self.connect(self.butFile       ,  QtCore.SIGNAL('clicked()'), self.onFile        )
        self.connect(self.butLogger     ,  QtCore.SIGNAL('clicked()'), self.onLogger      )
        self.connect(self.butFBrowser   ,  QtCore.SIGNAL('clicked()'), self.onFBrowser    )

        self.showToolTips()
        self.setStyle()
        self.printStyleInfo()

        self.onLogger()
        self.butFBrowser.setStyleSheet(cp.styleButtonBad)

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
        self.butSave.setToolTip('Save all current settings in the \nfile with configuration parameters') 
        self.butExit.setToolTip('Close all windows and \nexit this program') 
        self.butFile.setToolTip('Save current GUI image in PNG file')
        self.butELog.setToolTip('1. Save current GUI image in PNG file\n'\
                                '2. Submit PNG file with msg in ELog')
        self.butLogger.setToolTip('On/Off logger widow')
        self.butFBrowser.setToolTip('On/Off file viewer')
        #self.butStop.setToolTip('Not implemented yet...')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self.               setStyleSheet(cp.styleBkgd)
        #self.titControl    .setStyleSheet(cp.styleTitle)
        #self.butFiles      .setStyleSheet(cp.styleButton)
        #self.butBatchInfo  .setStyleSheet(cp.styleButton) 
        #self.butAnaSettings.setStyleSheet(cp.styleButton)
        #self.butSystem     .setStyleSheet(cp.styleButton)
        #self.butRun        .setStyleSheet(cp.styleButton)
        #self.butViewResults.setStyleSheet(cp.styleButton)
        #self.butStop       .setStyleSheet(cp.styleButton)
        #self.butLogger     .setStyleSheet(cp.styleGreenish)
        #self.butFBrowser   .setStyleSheet(cp.styleButton)
        self.butSave       .setStyleSheet(cp.styleButton)
        self.butExit       .setStyleSheet(cp.styleButton)
        self.butELog       .setStyleSheet(cp.styleButton)
        self.butFile       .setStyleSheet(cp.styleButton)
        #self.titControl    .setAlignment(QtCore.Qt.AlignCenter)

        #self.butSave.setText('')
        #self.butExit.setText('')
        #self.butExit.setFlat(True)


    def makeTabBar(self,mode=None) :
        #if mode is not None : self.tab_bar.close()
        self.tab_bar = QtGui.QTabBar()

        #Uses self.list_file_types
        self.ind_tab_files  = self.tab_bar.addTab( self.list_of_tabs[0] )
        self.ind_tab_batch  = self.tab_bar.addTab( self.list_of_tabs[1] )
        self.ind_tab_anaset = self.tab_bar.addTab( self.list_of_tabs[2] )
        self.ind_tab_system = self.tab_bar.addTab( self.list_of_tabs[3] )
        self.ind_tab_intmon = self.tab_bar.addTab( self.list_of_tabs[4] )
        self.ind_tab_run    = self.tab_bar.addTab( self.list_of_tabs[5] )
        self.ind_tab_result = self.tab_bar.addTab( self.list_of_tabs[6] )

        self.tab_bar.setTabTextColor(self.ind_tab_files  , QtGui.QColor('green'))
        self.tab_bar.setTabTextColor(self.ind_tab_batch  , QtGui.QColor('red'))
        self.tab_bar.setTabTextColor(self.ind_tab_anaset , QtGui.QColor('gray'))
        self.tab_bar.setTabTextColor(self.ind_tab_system , QtGui.QColor('blue'))
        self.tab_bar.setTabTextColor(self.ind_tab_intmon , QtGui.QColor('red'))
        self.tab_bar.setTabTextColor(self.ind_tab_run    , QtGui.QColor('magenta'))
        self.tab_bar.setTabTextColor(self.ind_tab_result , QtGui.QColor('gray'))
        self.tab_bar.setShape(QtGui.QTabBar.RoundedNorth)

        #self.tab_bar.setTabEnabled(1, False)
        #self.tab_bar.setTabEnabled(3, False)
        
        try    :
            tab_index = self.list_of_tabs.index(cp.current_tab.value())
        except :
            tab_index = 0
            cp.current_tab.setValue(self.list_of_tabs[tab_index])

        #tab_index = 0

        logger.info(' make_tab_bar - set mode: ' + cp.current_tab.value(), __name__)

        self.tab_bar.setCurrentIndex(tab_index)

        self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.onTabBar)


    def guiSelector(self):

        try    : self.gui_win.close()
        except : pass

        try    : del self.gui_win
        except : pass

        if   cp.current_tab.value() == self.list_of_tabs[0] :
            self.gui_win = GUIFiles(self)
            #self.setStatus(0, 'Status: processing for pedestals')
            
        elif cp.current_tab.value() == self.list_of_tabs[1] :
            self.gui_win = GUISetupInfo(self)
            #self.setStatus(0, 'Status: set file for flat field')

        elif cp.current_tab.value() == self.list_of_tabs[2] :
            self.gui_win = GUIAnaSettings(self)
            #self.setStatus(0, 'Status: set file for blemish mask')

        elif cp.current_tab.value() == self.list_of_tabs[3] :
            self.gui_win = GUISystemSettings(self)
            #self.setStatus(0, 'Status: processing for data')

        elif cp.current_tab.value() == self.list_of_tabs[4] :
            self.gui_win = GUIIntensityMonitors(self)
            #self.setStatus(0, 'Status: set pars for intensity mons.')

        elif cp.current_tab.value() == self.list_of_tabs[5] :
            self.gui_win = GUIRun(self)
            #self.setStatus(0, 'Status: set file for config. pars.')

        elif cp.current_tab.value() == self.list_of_tabs[6] :
            self.gui_win = GUIViewResults(self)
            #self.setStatus(0, 'Status: set work and result dirs.')


        #self.gui_win.setMinimumHeight(300)

        self.hboxW.addWidget(self.gui_win)

        min_height = self.gui_win.minimumHeight() 
        self.setFixedHeight(min_height + 90)


    def onTabBar(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        cp.current_tab.setValue( tab_name )
        logger.info(' ---> selected tab: ' + str(tab_ind) + ' - open GUI to work with: ' + tab_name, __name__)
        self.guiSelector()


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

        try    : self.gui_win.close()
        except : pass

        try    : del self.gui_win
        except : pass

        #try    : cp.guifiles.close()
        #except : pass

        #try    : cp.guisetupinfo.close()
        #except : pass

        #try    : cp.guianasettings.close()
        #except : pass

        #try    : cp.guisystemsettings.close()
        #except : pass

        #try    : cp.guiviewresults.close()
        #except : pass

        #try    : cp.guirun.close()
        #except : pass

        try    : cp.guilogger.close()
        except : pass

        try    : cp.guifilebrowser.close()
        except : pass

        try    : cp.plotimgspe.close()
        except : pass

        try    : cp.plotimgspe_g.close()
        except : pass

        try    : cp.plottime.close()
        except : pass

        try    : cp.plotarray.close()
        except : pass

        try    : cp.plot_g2.close()
        except : pass

        try    : cp.plot_gr.close()
        except : pass

        try    : cp.maskeditor.close()
        except : pass

        #try    : del cp.guimain
        #except : pass



    def onExit(self):
        logger.debug('onExit', self.name)
        self.close()
        
    def onPrint(self):
        logger.debug('onPrint', self.name)
        
    def onFiles(self):
        logger.debug('onFiles', self.name)
        try :
            cp.guifiles.close()
            #self.butFiles.setStyleSheet(cp.styleButton)
        except :
            #self.butFiles.setStyleSheet(cp.styleButtonOn)
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
        logger.info('Save status: '+str(status), self.name)


    def onELog(self):
        logger.debug('onELog', self.name)
        pixmap = QtGui.QPixmap.grabWidget(self)
        fname  = fnm.path_gui_image()
        status = pixmap.save(fname, 'PNG')
        logger.info('1. Save GUI image in file: ' + fname + ' status: '+str(status), self.name)
        if not status : return
        logger.info('2. Send GUI image in ELog: ', fname)
        wdialog = GUIELogPostingDialog (self, fname=fname)
        resp=wdialog.exec_()
  

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
            cp.guilogger.move(self.pos().__add__(QtCore.QPoint(860,00))) # open window with offset w.r.t. parent
            cp.guilogger.show()


    def onFBrowser (self):       
        logger.debug('onFBrowser', self.name)
        try    :
            cp.guifilebrowser.close()
        except :
            self.butFBrowser.setStyleSheet(cp.styleButtonGood)
            cp.guifilebrowser = GUIFileBrowser(None, fnm.get_list_of_files_total())
            cp.guifilebrowser.move(self.pos().__add__(QtCore.QPoint(880,40))) # open window with offset w.r.t. parent
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
    ex  = GUIMainTB()
    ex.show()
    app.exec_()
#-----------------------------
