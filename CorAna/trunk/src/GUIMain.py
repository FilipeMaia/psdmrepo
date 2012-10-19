
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

@version $Id: on 2012-10-08 copied from HDF5Explorer MainGUI $

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

from GUIConfigParameters import * 
from GUIInstrExpRun      import *
from GUIBatchInfo        import *

#import GUIPlayer          as guiplr
#import GUIComplexCommands as guicomplex
#import GUIWhatToDisplay   as guiwtd
#import GUISelectItems     as guiselitems
#import GUISelection       as guisel
#import DrawEvent          as drev
#import PrintHDF5          as printh5 # for my print_group(g,offset)

#---------------------
#  Class definition --
#---------------------
class GUIMain ( QtGui.QWidget ) :
    """Deals with the main GUI for the interactive analysis project.

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

        #self.setGeometry(370, 10, 500, 300)
        self.setGeometry(10, 20, 150, 500)
        self.setWindowTitle('Interactive Analysis')
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

#        cp.guimain = self
#        cp.readParameters()
#        if not cp.readParsFromFileAtStart :
#            cp.setDefaultParameters()
#        cp.Print()
#        print 'Current event number : %d ' % (cp.eventCurrent)

	#print 'sys.argv=',sys.argv # list of input parameters


        self.setFrame()
 
        #self.drawev   = drev.DrawEvent()

        #self.titFile   = QtGui.QLabel('File:')
        #self.titTree   = QtGui.QLabel('HDF5 Tree GUI')
        self.char_expand    = u' \u25BE' # down-head triangle
        
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

        self.setButtonStyle()

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
        self.printStyleInfo()

        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------

    def printStyleInfo(self):
        qstyle     = self.style()
        qpalette   = qstyle.standardPalette()
        qcolor_bkg = qpalette.color(1)
        r,g,b,alp  = qcolor_bkg.getRgb()
        print 'Background color: r,g,b,alp=', r,g,b,alp


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


    def setButtonStyle(self):

        #self.titStyle   = "background-color: rgb(239, 235, 231, 255); color: rgb(100, 160, 100);" # Gray bkgd
        #self.titStyle   = "color: rgb(100, 160, 100);"

        #self.setStyleSheet(cp.styleYellow)
        self.titControl.setStyleSheet (cp.styleTitle)
        self.titControl.setAlignment(QtCore.Qt.AlignCenter)


#        if cp.step01IsDone : self.browse .setStyleSheet(cp.styleGray)
#        else                        : self.browse .setStyleSheet(cp.styleGreen)

#        if cp.step02IsDone : self.display.setStyleSheet(self.styleGray)
#        else                        : self.display.setStyleSheet(self.styleGreen)

#        if cp.step03IsDone : self.wtd    .setStyleSheet(self.styleGray)
#        else                        : self.wtd    .setStyleSheet(self.styleGreen)

#        if cp.step04IsDone : self.player .setStyleSheet(self.styleGray)
#        else                        : self.player .setStyleSheet(self.styleGreen)


    def moveEvent(self, e):
        pass
#        print 'moveEvent' 
#        cp.posGUIMain = (self.pos().x(),self.pos().y())

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def processPrint(self):
        print 'processPrint()'
#        fname = cp.dirName+'/'+cp.fileName
#        print 'Print structure of the HDF5 file:\n %s' % (fname)
#        printh5.print_hdf5_file_structure(fname)

    def closeEvent(self, event):
        #print 'closeEvent'
        try    : del cp.guimain
        except : pass

        try    : cp.guiconfigparameters.close()
        except : pass

        try    : cp.guibatchinfo.close()
        except : pass

    def onExit(self):
        #print 'Exit button is clicked'
        self.close()
        
        
    def onLoadFiles(self):
        print 'onLoadFiles'
        try :
            cp.guiconfigparameters.close()
        except : # AttributeError: #NameError 
            cp.guiconfigparameters = GUIConfigParameters()
            cp.guiconfigparameters.setParent(self)
            cp.guiconfigparameters.move(self.pos().__add__(QtCore.QPoint(160,60))) # open window with offset w.r.t. parent
            cp.guiconfigparameters.show()


    def onBatchInfo(self):
        print 'onBatchInfo'
        try :
            cp.guibatchinfo.close()
        except : # AttributeError: #NameError 
            cp.guibatchinfo = GUIBatchInfo()
            cp.guibatchinfo.setParent(self)
            cp.guibatchinfo.move(self.pos().__add__(QtCore.QPoint(160,90))) # open window with offset w.r.t. parent
            cp.guibatchinfo.show()


    def onSave(self):
        print 'onSave'
        cp.saveParametersInFile( cp.fname_cp.value() )


    def onAnaDisp(self):    
        print 'onAnaDisp'

    def onSystem(self):     
        print 'onSystem'

    def onRun (self):       
        print 'onRun'

    def onViewResults(self):
        print 'onViewResults'

    def onStop(self):       
        print 'onStop'

                
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

            #self.processFileEdit()
            #self.processNumbEdit()
            #self.processSpanEdit()
            #self.currentEventNo()

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
