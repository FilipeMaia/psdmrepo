
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMain...
#
#------------------------------------------------------------------------

"""Main GUI in the HDF5Analysis application.

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
import ConfigParameters   as cp
import GUIHDF5Tree        as guiselitems
import GUIDataSets        as guidatasets
import GUIConfiguration   as guiconfig

#import PrintHDF5          as printh5 # for my print_group(g,offset)
#import GUIPlayer          as guiplr
#import GUIComplexCommands as guicomplex
#import GUIWhatToDisplay   as guiwtd

#---------------------
#  Class definition --
#---------------------
class GUIMain ( QtGui.QWidget ) :
    """Deals with the main GUI for the HDF5Analysis project
    """
    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, parent=None, app=None) :
        """Constructor."""

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(10, 20, 500, 150)
        self.setWindowTitle('HDF5Analysis')
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False
        self.setFrame()

        cp.confpars.guimain = self

        cp.confpars.readParameters()
        if not cp.confpars.readParsFromFileAtStart :
            cp.confpars.setDefaultParameters()
        cp.confpars.Print()
        print 'Current event number : %d ' % (cp.confpars.eventCurrent)

	#print 'sys.argv=',sys.argv # list of input parameters



        self.editFile  = QtGui.QLineEdit(cp.confpars.dirName+'/'+cp.confpars.fileName)

        self.butBrowse    = QtGui.QPushButton("1. Select file:")    
        self.butHDF5GUI   = QtGui.QPushButton("2. Check datasets in HDF5 tree")
        self.butDataSets  = QtGui.QPushButton("3. Select parameters in datasets")
        self.butPlayer    = QtGui.QPushButton("4. Reserved")
        self.butConfig    = QtGui.QPushButton("Configuration")
        self.butSave      = QtGui.QPushButton("Save")
        self.butExit      = QtGui.QPushButton("Exit")

        #self.butDataSets.setMinimumHeight(30)
        #self.butDataSets.setMinimumWidth(210)

        self.setButtonColors()

        hboxF = QtGui.QHBoxLayout()
        hboxF.addWidget(self.butBrowse)
        hboxF.addWidget(self.editFile)

        hboxC = QtGui.QHBoxLayout()
        hboxC.addWidget(self.butHDF5GUI)
        hboxC.addStretch(1)
        
        hboxE = QtGui.QHBoxLayout()
        hboxE.addWidget(self.butDataSets)
        hboxE.addStretch(1)

        self.hboxT = QtGui.QHBoxLayout() 
        self.hboxA = QtGui.QHBoxLayout() 

        if cp.confpars.playerGUIIsOpen : # At initialization it means that "should be open..."
            self.setPlayerWidgets()

        hboxG = QtGui.QHBoxLayout()
        hboxG.addWidget(self.butPlayer)
        hboxG.addStretch(1)

        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.butConfig)
        hbox.addWidget(self.butSave)
        hbox.addWidget(self.butExit)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hboxF)
        vbox.addStretch(1)     
        vbox.addLayout(hboxC)
        vbox.addStretch(1)     
        vbox.addLayout(hboxE)
        vbox.addStretch(1)     
        vbox.addLayout(hboxG)
        vbox.addStretch(1)     
        vbox.addLayout(hbox)
        vbox.addStretch(1)
        vbox.addLayout(self.hboxT)
        vbox.addLayout(self.hboxA)

        self.setLayout(vbox)

        self.connect(self.butBrowse,    QtCore.SIGNAL('clicked()'),          self.processBrowse )
        self.connect(self.butHDF5GUI,   QtCore.SIGNAL('clicked()'),          self.processHDF5 )
        self.connect(self.butDataSets,  QtCore.SIGNAL('clicked()'),          self.processDataSets )
        self.connect(self.butPlayer,    QtCore.SIGNAL('clicked()'),          self.processPlayer )
        self.connect(self.butConfig,    QtCore.SIGNAL('clicked()'),          self.processConfig )
        self.connect(self.butSave,      QtCore.SIGNAL('clicked()'),          self.processSave )
        self.connect(self.butExit,      QtCore.SIGNAL('clicked()'),          self.processQuit )
        self.connect(self.editFile,     QtCore.SIGNAL('editingFinished ()'), self.processEditFile )

        #self.resize(500, 300)
        self.showToolTips()
        #print 'End of init'

        cp.confpars.guiMainIsOpen = True

        
    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        self.butSave.setToolTip('Save all current settings in the \nfile with configuration parameters.') 
        self.butExit.setToolTip('Close all windows and \nexit this program') 


    def setButtonColors(self):
        #self.styleYellow = "background-color: rgb(255, 255, 230); color: rgb(0, 0, 0)" # Yellowish
        #self.stylePink   = "background-color: rgb(255, 240, 245); color: rgb(0, 0, 0)" # Pinkish
        self.styleGreen  = "background-color: rgb(220, 255, 220); color: rgb(0, 0, 0)" # Greenish
        self.styleGray   = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0)" # Pinkish

        if cp.confpars.step01IsDone : self.butBrowse  .setStyleSheet(self.styleGray)
        else                        : self.butBrowse  .setStyleSheet(self.styleGreen)

        if cp.confpars.step02IsDone : self.butHDF5GUI .setStyleSheet(self.styleGray)
        else                        : self.butHDF5GUI .setStyleSheet(self.styleGreen)

        if cp.confpars.step03IsDone : self.butDataSets.setStyleSheet(self.styleGray)
        else                        : self.butDataSets.setStyleSheet(self.styleGreen)

        if cp.confpars.step04IsDone : self.butPlayer  .setStyleSheet(self.styleGray)
        else                        : self.butPlayer  .setStyleSheet(self.styleGreen)


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def moveEvent(self, e):
        #print 'moveEvent' 
        cp.confpars.posGUIMain = (self.pos().x(),self.pos().y())


    #def processPrint(self):
    #    fname = cp.confpars.dirName+'/'+cp.confpars.fileName
    #    print 'Print structure of the HDF5 file:\n %s' % (fname)
    #    printh5.print_hdf5_file_structure(fname)


    def closeEvent(self, event):
        #print 'closeEvent'
        #print 'Quit GUIMain'
        #self.drawev.quitDrawEvent()
        if cp.confpars.playerGUIIsOpen :
            self.wplayer.processQuit()
            self.wcomplex.processQuit()
        self.SHowIsOn = False
        if cp.confpars.dsetsGUIIsOpen :
            cp.confpars.guidsets.close()
        if cp.confpars.treeWindowIsOpen :
            cp.confpars.guitree.close()
        if cp.confpars.configGUIIsOpen :
            cp.confpars.guiconfig.close()
        #if cp.confpars.selectionGUIIsOpen :
        #    cp.confpars.guiselection.close()
        #print 'Segmentation fault may happen at exit, when the dialog is closed. \nThis is a known problem of python-qt4 version.'
        print 'Exit HDF5Explorer'

        
    def processQuit(self):
        print 'Exit button is clicked'
        self.close()


    def processBrowse(self):
        print 'Browse'
        cp.confpars.step01IsDone = True
        self.setButtonColors()
        str_path_file = str(self.editFile.displayText())
        cp.confpars.dirName,cp.confpars.fileName = os.path.split(str_path_file)
        print 'dirName  : %s' % (cp.confpars.dirName)         
        print 'fileName : %s' % (cp.confpars.fileName)
        path_file = QtGui.QFileDialog.getOpenFileName(self,'Open file',cp.confpars.dirName)
        print path_file
        str_path_file = str(path_file)
        self.editFile.setText(str_path_file)
        dirName,fileName = os.path.split(str_path_file)
        if dirName == '' or fileName == '' :
            print 'Input dirName or fileName is empty... use default values'  
        else :
            cp.confpars.dirName  = dirName
            cp.confpars.fileName = fileName
        print 'Set new dirName  : %s' % (cp.confpars.dirName)         
        print 'Set new fileName : %s' % (cp.confpars.fileName)         
        str_path_file = cp.confpars.dirName + '/' + cp.confpars.fileName
        self.editFile.setText(str_path_file)
        if not os.path.exists(str_path_file) :
            print 'The file %s does not exist' % (str_path_file)
            print 'Use existing file name ...'


    #def processSelection(self):
    #    print 'Selection'
    #    if  cp.confpars.selectionGUIIsOpen : # close wtd window
    #        print 'Selection GUI: Close'
    #        #self.selection.setText('Open Selection')
    #        cp.confpars.guiselection.close()
    #    else :                           # Open wtd window
    #        print 'Selection GUI: Open'
    #        #self.selection.setText('Close Selection')
    #        cp.confpars.guiselection = guisel.GUIDataSets()
    #        cp.confpars.guiselection.move(self.pos().__add__(QtCore.QPoint(500,330))) # open window with offset w.r.t. parent
    #        cp.confpars.guiselection.show()

        
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


    def processDataSets(self):
        if cp.confpars.dsetsGUIIsOpen : # close wtd window
            print 'Close GUIDataSets'
            cp.confpars.guidsets.close()
        else :                           # Open wtd window
            print 'Open GUIDataSets'
            cp.confpars.guidsets = guidatasets.GUIDataSets()
            cp.confpars.guidsets.move(self.pos().__add__(QtCore.QPoint(0,380))) # open window with offset w.r.t. parent
            cp.confpars.guidsets.show()

        cp.confpars.step03IsDone = True
        self.setButtonColors()


    def processHDF5(self):
        if cp.confpars.treeWindowIsOpen : # close wtd window
            print 'Close HDF5 tree GUI'
            #self.butHDF5GUI.setText('Open HDF5 tree')
            cp.confpars.guitree.close()
        else :                           # Open wtd window
            print 'Open HDF5 tree GUI'
            #self.butHDF5GUI.setText('Close HDF5 tree')
            cp.confpars.guitree = guiselitems.GUIHDF5Tree()
            #cp.confpars.guitree.setParent(self) # bypass for parent initialization in the base QWidget
            cp.confpars.guitree.move(self.pos().__add__(QtCore.QPoint(510,0))) # (-360,0)open window with offset w.r.t. parent
            cp.confpars.guitree.show()
        cp.confpars.step02IsDone = True
        self.setButtonColors()


    def processPlayer(self):
        print 'GUI Reserved ...'
        #if  cp.confpars.playerGUIIsOpen :
        #    print 'Close Player sub-GUI'
        #    self.wplayer.close()
        #    self.wcomplex.close()
        #    self.setFixedSize(500,150)
        #else :    
        #    print 'Open Player sub-GUI'
        #    self.setPlayerWidgets()
        #cp.confpars.step04IsDone = True
        #self.setButtonColors()


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


    def processEditFile(self):
        print 'EditFile'
        str_path_file = str(self.editFile.displayText())
        cp.confpars.dirName,cp.confpars.fileName = os.path.split(str_path_file)
        print 'Set dirName      : %s' % (cp.confpars.dirName)         
        print 'Set fileName     : %s' % (cp.confpars.fileName)         
                

    def keyPressEvent(self, event):
        #http://doc.qt.nokia.com/4.6/qt.html#Key-enum
        print 'event.key() = %s' % (event.key())
        if event.key() == QtCore.Qt.Key_Escape:
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
