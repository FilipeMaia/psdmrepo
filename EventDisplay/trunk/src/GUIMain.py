
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMain...
#
#------------------------------------------------------------------------

"""Renders the main GUI in the event display application.

Following paragraphs provide detailed description of the module.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

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
import GUIPlayer          as guiplr
import GUIComplexCommands as guicomplex
import GUIWhatToDisplay   as guiwtd
import GUISelectItems     as guiselitems
import GUIConfiguration   as guiconfig
import GUISelection       as guisel
import ConfigParameters   as cp
#import DrawEvent          as drev
import PrintHDF5          as printh5 # for my print_group(g,offset)

#---------------------
#  Class definition --
#---------------------
class GUIMain ( QtGui.QWidget ) :
    """Deals with the main GUI for the event display project

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
        """Constructor."""

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        #self.setGeometry(370, 10, 500, 300)
        self.setGeometry(10, 20, 500, 150)
        self.setWindowTitle('HDF5 Event Display')
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        cp.confpars.guimain = self

        cp.confpars.readParameters()
        if not cp.confpars.readParsFromFileAtStart :
            cp.confpars.setDefaultParameters()
        cp.confpars.Print()
        print 'Current event number : %d ' % (cp.confpars.eventCurrent)

	#print 'sys.argv=',sys.argv # list of input parameters

        # see http://www.riverbankcomputing.co.uk/static/Docs/PyQt4/html/qframe.html
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(True)

        #self.drawev   = drev.DrawEvent()

        #self.titFile   = QtGui.QLabel('File:')
        #self.titTree   = QtGui.QLabel('HDF5 Tree GUI')

        self.fileEdit  = QtGui.QLineEdit(cp.confpars.dirName+'/'+cp.confpars.fileName)

        self.browse    = QtGui.QPushButton("1. Select file:")    
        self.display   = QtGui.QPushButton("2. Check datasets in HDF5 tree")
        self.wtd       = QtGui.QPushButton("3. Set what and how to display")
        self.player    = QtGui.QPushButton("4. Plot data in several modes")
        self.exit      = QtGui.QPushButton("Exit")
        self.save      = QtGui.QPushButton("Save")
        #self.printfile = QtGui.QPushButton("Print HDF5 structure")    
        #self.config    = QtGui.QPushButton("Configuration")
        #self.selection = QtGui.QPushButton("Selection")
        #self.save.setMaximumWidth(40)   

        #self.wtd.setMinimumHeight(30)
        #self.wtd.setMinimumWidth(210)

        self.setButtonColors()

        hboxF = QtGui.QHBoxLayout()
        #hboxF.addWidget(self.titFile)
        hboxF.addWidget(self.browse)
        hboxF.addWidget(self.fileEdit)

        hboxC = QtGui.QHBoxLayout()
        #hboxC.addStretch(1)
        hboxC.addWidget(self.display)
        hboxC.addStretch(1)
        
        hboxE = QtGui.QHBoxLayout()
        #hboxE.addWidget(self.selection)
        #hboxE.addStretch(1)
        hboxE.addWidget(self.wtd)
        hboxE.addStretch(1)

        self.hboxT = QtGui.QHBoxLayout() 
        self.hboxA = QtGui.QHBoxLayout() 

        if cp.confpars.playerGUIIsOpen : # At initialization it means that "should be open..."
            self.setPlayerWidgets()

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.player)
        hbox.addStretch(1)
        hbox.addWidget(self.save)
        hbox.addWidget(self.exit)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hboxF)
        vbox.addStretch(1)     
        vbox.addLayout(hboxC)
        vbox.addStretch(1)     
        vbox.addLayout(hboxE)
        vbox.addStretch(1)     
        vbox.addLayout(hbox)
        vbox.addStretch(1)
        vbox.addLayout(self.hboxT)
        vbox.addLayout(self.hboxA)

        self.setLayout(vbox)

        self.connect(self.exit,      QtCore.SIGNAL('clicked()'), self.processQuit )
        self.connect(self.browse,    QtCore.SIGNAL('clicked()'), self.processBrowse )
        self.connect(self.display,   QtCore.SIGNAL('clicked()'), self.processDisplay )
        self.connect(self.wtd,       QtCore.SIGNAL('clicked()'), self.processWhatToDisplay )
        self.connect(self.player,    QtCore.SIGNAL('clicked()'), self.processPlayer )
        self.connect(self.fileEdit,  QtCore.SIGNAL('editingFinished ()'), self.processFileEdit )
        self.connect(self.save,      QtCore.SIGNAL('clicked()'), self.processSave )
        #self.connect(self.printfile, QtCore.SIGNAL('clicked()'), self.processPrint )
        #self.connect(self.config,    QtCore.SIGNAL('clicked()'), self.processConfig )
        #self.connect(self.selection, QtCore.SIGNAL('clicked()'), self.processSelection )

        #self.setFocus()
        #self.resize(500, 300)
        self.showToolTips()
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------


    def showToolTips(self):
        self.save.setToolTip('Save all current settings in the \nfile with configuration parameters.') 
        self.exit.setToolTip('Close all windows and \nexit this program') 


    def setButtonColors(self):
        #self.styleYellow = "background-color: rgb(255, 255, 230); color: rgb(0, 0, 0)" # Yellowish
        #self.stylePink   = "background-color: rgb(255, 240, 245); color: rgb(0, 0, 0)" # Pinkish
        self.styleGreen  = "background-color: rgb(220, 255, 220); color: rgb(0, 0, 0)" # Greenish
        self.styleGray   = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0)" # Pinkish

        if cp.confpars.step01IsDone : self.browse .setStyleSheet(self.styleGray)
        else                        : self.browse .setStyleSheet(self.styleGreen)

        if cp.confpars.step02IsDone : self.display.setStyleSheet(self.styleGray)
        else                        : self.display.setStyleSheet(self.styleGreen)

        if cp.confpars.step03IsDone : self.wtd    .setStyleSheet(self.styleGray)
        else                        : self.wtd    .setStyleSheet(self.styleGreen)

        if cp.confpars.step04IsDone : self.player .setStyleSheet(self.styleGray)
        else                        : self.player .setStyleSheet(self.styleGreen)


    def moveEvent(self, e):
        #print 'moveEvent' 
        cp.confpars.posGUIMain = (self.pos().x(),self.pos().y())

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def processPrint(self):
        fname = cp.confpars.dirName+'/'+cp.confpars.fileName
        print 'Print structure of the HDF5 file:\n %s' % (fname)
        printh5.print_hdf5_file_structure(fname)

    def closeEvent(self, event):
        #print 'closeEvent'
        self.processQuit()

    def processQuit(self):
        #print 'Quit GUIMain'
        #self.drawev.quitDrawEvent()
        if cp.confpars.playerGUIIsOpen :
            self.wplayer.processQuit()
            self.wcomplex.processQuit()
        self.SHowIsOn = False
        if cp.confpars.wtdWindowIsOpen :
            cp.confpars.guiwhat.close()
        if cp.confpars.treeWindowIsOpen :
            cp.confpars.guitree.close()
        if cp.confpars.configGUIIsOpen :
            cp.confpars.guiconfig.close()
        if cp.confpars.selectionGUIIsOpen :
            cp.confpars.guiselection.close()
        #print 'Segmentation fault may happen at closing of the Main GUI window. The reason for that is not clear yet...'
              #It happens after opening/closing HDF5 Tree and WTD GUIs...
        self.close()


    def processBrowse(self):
        print 'Browse'
        cp.confpars.step01IsDone = True
        self.setButtonColors()
        #self.drawev.closeHDF5File()
        str_path_file = str(self.fileEdit.displayText())
        cp.confpars.dirName,cp.confpars.fileName = os.path.split(str_path_file)
        print 'dirName  : %s' % (cp.confpars.dirName)         
        print 'fileName : %s' % (cp.confpars.fileName)
        path_file = QtGui.QFileDialog.getOpenFileName(self,'Open file',cp.confpars.dirName)
        #fname = open(filename)
        #data = fname.read()
        #self.textEdit.setText(data)
        print path_file
        str_path_file = str(path_file)
        self.fileEdit.setText(str_path_file)
        dirName,fileName = os.path.split(str_path_file)
        if dirName == '' or fileName == '' :
            print 'Input dirName or fileName is empty... use default values'  
        else :
            cp.confpars.dirName  = dirName
            cp.confpars.fileName = fileName
        print 'Set new dirName  : %s' % (cp.confpars.dirName)         
        print 'Set new fileName : %s' % (cp.confpars.fileName)         
        str_path_file = cp.confpars.dirName + '/' + cp.confpars.fileName
        self.fileEdit.setText(str_path_file)
        if not os.path.exists(str_path_file) :
            print 'The file %s does not exist' % (str_path_file)
            print 'Use existing file name ...'

    def processSelection(self):
        print 'Selection'
        if  cp.confpars.selectionGUIIsOpen : # close wtd window
            print 'Selection GUI: Close'
            #self.selection.setText('Open Selection')
            cp.confpars.guiselection.close()
        else :                           # Open wtd window
            print 'Selection GUI: Open'
            #self.selection.setText('Close Selection')
            cp.confpars.guiselection = guisel.GUISelection()
            cp.confpars.guiselection.move(self.pos().__add__(QtCore.QPoint(500,330))) # open window with offset w.r.t. parent
            cp.confpars.guiselection.show()

        
    def processConfig(self):
        print 'Configuration'
        if  cp.confpars.configGUIIsOpen :
            cp.confpars.guiconfig.close()
        else :    
            cp.confpars.guiconfig = guiconfig.GUIConfiguration()
            cp.confpars.guiconfig.setParent(self)
            cp.confpars.guiconfig.move(self.pos().__add__(QtCore.QPoint(100,330))) # open window with offset w.r.t. parent
            cp.confpars.guiconfig.show()


    def processSave(self):
        print 'Save'
        cp.confpars.writeParameters()


    def processWhatToDisplay(self):
        if cp.confpars.wtdWindowIsOpen : # close wtd window
            print 'Close What to display GUI'
            #self.wtd.setText('Open')
            cp.confpars.guiwhat.close()
        else :                           # Open wtd window
            print 'Open What to display GUI'
            #self.wtd.setText('Close')
            cp.confpars.guiwhat = guiwtd.GUIWhatToDisplay()
            cp.confpars.guiwhat.move(self.pos().__add__(QtCore.QPoint(0,380))) # open window with offset w.r.t. parent
            cp.confpars.guiwhat.show()

        cp.confpars.step03IsDone = True
        self.setButtonColors()


    def processDisplay(self):
        if cp.confpars.treeWindowIsOpen : # close wtd window
            print 'Close HDF5 tree GUI'
            #self.display.setText('Open HDF5 tree')
            cp.confpars.guitree.close()
        else :                           # Open wtd window
            print 'Open HDF5 tree GUI'
            #self.display.setText('Close HDF5 tree')
            cp.confpars.guitree = guiselitems.GUISelectItems()
            #cp.confpars.guitree.setParent(self) # bypass for parent initialization in the base QWidget
            cp.confpars.guitree.move(self.pos().__add__(QtCore.QPoint(510,0))) # (-360,0)open window with offset w.r.t. parent
            cp.confpars.guitree.show()
        cp.confpars.step02IsDone = True
        self.setButtonColors()


    def processPlayer(self):
        #print 'Player GUI'
        if  cp.confpars.playerGUIIsOpen :
            print 'Close Player sub-GUI'
            self.wplayer.close()
            self.wcomplex.close()
            self.setFixedSize(500,150)
        else :    
            print 'Open Player sub-GUI'
            self.setPlayerWidgets()
            #self.show()
        cp.confpars.step04IsDone = True
        self.setButtonColors()


    def setPlayerWidgets(self):
        #if cp.confpars.playerGUIIsOpen :
        self.wplayer  = guiplr.GUIPlayer()
        self.wcomplex = guicomplex.GUIComplexCommands(None, self.wplayer)
        self.hboxT.addWidget(self.wplayer)
        self.hboxA.addWidget(self.wcomplex)
        self.setFixedSize(500,350)


    def mousePressEvent(self, event):
        #print 'Do not click on mouse just for fun!'
        #print 'event.button() = %s at position' % (event.button()),        
        #print (event.pos()),
        #print ' x=%d, y=%d' % (event.x(),event.y()),        
        #print ' global x=%d, y=%d' % (event.globalX(),event.globalY())
        #self.emit(QtCore.SIGNAL('closeGUIApp()'))
        pass

    def processFileEdit(self):
        print 'FileEdit'
        str_path_file = str(self.fileEdit.displayText())
        cp.confpars.dirName,cp.confpars.fileName = os.path.split(str_path_file)
        print 'Set dirName      : %s' % (cp.confpars.dirName)         
        print 'Set fileName     : %s' % (cp.confpars.fileName)         
                
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
