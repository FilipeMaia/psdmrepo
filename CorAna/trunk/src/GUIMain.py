
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMain...
#
#------------------------------------------------------------------------

"""Renders the main GUI in the analysis shell.

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
import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersCorAna import confpars as cp

#from GUIConfigParameters import * 
from GUILoadFiles        import *
from GUIBatchInfo        import *
from GUIAnaSettings      import *
from Logger              import logger

#---------------------
#  Class definition --
#---------------------
class GUIMain ( QtGui.QWidget ) :
    """Main GUI for the interactive analysis project.

    @see BaseClass
    @see OtherClass
    """

    #--------------------
    #  Class variables --
    #--------------------
    #publicStaticVariable = 0 
    #__privateStaticVariable = "A string"

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, parent=None, app=None) :
        """Constructor of GUIMain."""

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(10, 20, 150, 500)
        self.setWindowTitle('Interactive Analysis')
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        self.setFrame()
 
        self.titControl    = QtGui.QLabel('Control Panel')
        self.butLoadFiles  = QtGui.QPushButton('Load files')    
        self.butBatchInfo  = QtGui.QPushButton('Batch information')    
        self.butAnaDisp    = QtGui.QPushButton('Analysis && Display')
        self.butSystem     = QtGui.QPushButton('System')
        self.butRun        = QtGui.QPushButton('Run')
        self.butViewResults= QtGui.QPushButton('View Results')
        self.butStop       = QtGui.QPushButton('Stop')
        self.butSave       = QtGui.QPushButton('Save')
        self.butExit       = QtGui.QPushButton('Exit')

        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addWidget(self.titControl    )
        self.vbox.addWidget(self.butLoadFiles  )
        self.vbox.addWidget(self.butBatchInfo  )
        self.vbox.addWidget(self.butAnaDisp    )
        self.vbox.addWidget(self.butSystem     )
        self.vbox.addWidget(self.butRun        )
        self.vbox.addWidget(self.butViewResults)
        self.vbox.addStretch(1)     
        self.vbox.addWidget(self.butStop       )
        self.vbox.addWidget(self.butSave       )
        self.vbox.addWidget(self.butExit       )

        self.setLayout(self.vbox)

        self.connect(self.butLoadFiles  ,  QtCore.SIGNAL('clicked()'), self.onLoadFiles   )
        self.connect(self.butBatchInfo  ,  QtCore.SIGNAL('clicked()'), self.onBatchInfo   )
        self.connect(self.butAnaDisp    ,  QtCore.SIGNAL('clicked()'), self.onAnaDisp     )
        self.connect(self.butSystem     ,  QtCore.SIGNAL('clicked()'), self.onSystem      )
        self.connect(self.butRun        ,  QtCore.SIGNAL('clicked()'), self.onRun         )
        self.connect(self.butViewResults,  QtCore.SIGNAL('clicked()'), self.onViewResults )
        self.connect(self.butStop       ,  QtCore.SIGNAL('clicked()'), self.onStop        )
        self.connect(self.butSave       ,  QtCore.SIGNAL('clicked()'), self.onSave        )
        self.connect(self.butExit       ,  QtCore.SIGNAL('clicked()'), self.onExit        )

        self.showToolTips()
        self.setStyle()
        self.printStyleInfo()
        
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
        self.butLoadFiles  .setStyleSheet(cp.styleButton)
        self.butBatchInfo  .setStyleSheet(cp.styleButton) 
        self.butAnaDisp    .setStyleSheet(cp.styleButton)
        self.butSystem     .setStyleSheet(cp.styleButton)
        self.butRun        .setStyleSheet(cp.styleButton)
        self.butViewResults.setStyleSheet(cp.styleButton)
        self.butStop       .setStyleSheet(cp.styleButton)
        self.butSave       .setStyleSheet(cp.styleButton)
        self.butExit       .setStyleSheet(cp.styleButton)
        self.titControl    .setAlignment(QtCore.Qt.AlignCenter)

    def moveEvent(self, e):
        pass
#        print 'moveEvent' 
#        cp.posGUIMain = (self.pos().x(),self.pos().y())

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def processPrint(self):
        print 'processPrint()'

    def closeEvent(self, event):
        logger.info('closeEvent')
        try    : del cp.guimain
        except : pass

        try    : cp.guiloadfiles.close()
        except : pass

        try    : cp.guibatchinfo.close()
        except : pass

        try    : cp.guianasettings.close()
        except : pass

    def onExit(self):
        logger.info('onExit')
        self.close()
        
        
    def onLoadFiles(self):
        logger.info('onLoadFiles')
        try :
            cp.guiloadfiles.close()
            self.butLoadFiles.setStyleSheet(cp.styleButton)
        except : # AttributeError: #NameError 
            cp.guiloadfiles = GUILoadFiles()
            cp.guiloadfiles.setParent(self)
            cp.guiloadfiles.move(self.pos().__add__(QtCore.QPoint(160,60))) # open window with offset w.r.t. parent
            cp.guiloadfiles.show()
            self.butLoadFiles.setStyleSheet(cp.styleButtonOn)


    def onBatchInfo(self):
        logger.info('onBatchInfo')
        try :
            cp.guibatchinfo.close()
        except : # AttributeError: #NameError 
            cp.guibatchinfo = GUIBatchInfo()
            cp.guibatchinfo.setParent(self)
            cp.guibatchinfo.move(self.pos().__add__(QtCore.QPoint(160,90))) # open window with offset w.r.t. parent
            cp.guibatchinfo.show()


    def onSave(self):
        logger.info('onSave')
        cp.saveParametersInFile( cp.fname_cp.value() )

    def onAnaDisp(self):    
        logger.info('onAnaDisp')
        try :
            cp.guianasettings.close()
        except : # AttributeError: #NameError 
            cp.guianasettings = GUIAnaSettings()
            cp.guianasettings.setParent(self)
            cp.guianasettings.move(self.pos().__add__(QtCore.QPoint(160,130))) # open window with offset w.r.t. parent
            cp.guianasettings.show()

    def onSystem(self):     
        logger.info('onSystem - not implemented yet...')

    def onRun (self):       
        logger.info('onRun - not implemented yet...')

    def onViewResults(self):
        logger.info('onViewResults - not implemented yet...')

    def onStop(self):       
        logger.info('onStop - not implemented yet...')
                
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
#http://doc.qt.nokia.com/4.6/qt.html#Key-enum
    def keyPressEvent(self, event):
        print 'event.key() = %s' % (event.key())
        if event.key() == QtCore.Qt.Key_Escape:
    #        self.close()
            self.SHowIsOn = False    

        if event.key() == QtCore.Qt.Key_B:
            print 'event.key() = %s' % (QtCore.Qt.Key_B)

        if event.key() == QtCore.Qt.Key_Return:
            print 'event.key() = Return'

        if event.key() == QtCore.Qt.Key_Home:
            print 'event.key() = Home'

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIMain()
    ex.show()
    app.exec_()
#-----------------------------
