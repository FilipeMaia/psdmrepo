
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
        self.setGeometry(10, 20, 500, 300)
        self.setWindowTitle('HDF5 Event Display')
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        cp.confpars.readParameters()
        if not cp.confpars.readParsFromFileAtStart :
            cp.confpars.setDefaultParameters()
        cp.confpars.Print()
        print 'Current event number directly : %d ' % (cp.confpars.eventCurrent)

        # see http://www.riverbankcomputing.co.uk/static/Docs/PyQt4/html/qframe.html
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(True)

        #self.drawev   = drev.DrawEvent()

        self.titFile   = QtGui.QLabel('File:')
        #self.titTree   = QtGui.QLabel('HDF5 Tree GUI')

        self.fileEdit  = QtGui.QLineEdit(cp.confpars.dirName+'/'+cp.confpars.fileName)

        self.browse    = QtGui.QPushButton("Browse")    
        self.printfile = QtGui.QPushButton("Print HDF5 structure")    
        self.display   = QtGui.QPushButton("HDF5 tree")
        self.wtd       = QtGui.QPushButton("What to display")
        self.config    = QtGui.QPushButton("Configuration")
        self.save      = QtGui.QPushButton("Save")
        self.selection = QtGui.QPushButton("Selection")
        self.exit      = QtGui.QPushButton("Exit")
        self.save.setMaximumWidth(40)   

        hboxF = QtGui.QHBoxLayout()
        hboxF.addWidget(self.titFile)
        hboxF.addWidget(self.fileEdit)
        hboxF.addWidget(self.browse)

        hboxC = QtGui.QHBoxLayout()
        hboxC.addStretch(2)
        hboxC.addWidget(self.display)
        
        hboxE = QtGui.QHBoxLayout()
        hboxE.addWidget(self.selection)
        hboxE.addStretch(2)
        hboxE.addWidget(self.wtd)

        self.wplayer = guiplr.GUIPlayer()
        hboxT = QtGui.QHBoxLayout() 
        hboxT.addWidget(self.wplayer)

        self.wcomplex = guicomplex.GUIComplexCommands(None, self.wplayer)
        hboxA = QtGui.QHBoxLayout() 
        hboxA.addWidget(self.wcomplex)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.config)
        hbox.addWidget(self.save)
        hbox.addStretch(3)
        hbox.addWidget(self.exit)

        #hboxL = QtGui.QHBoxLayout()
        #hboxL.addWidget(self.lcd)
        #hboxL.addWidget(self.slider)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hboxF)
        vbox.addStretch(1)     
        vbox.addLayout(hboxC)
        vbox.addStretch(1)     
        vbox.addLayout(hboxE)
        vbox.addStretch(1)     
        vbox.addLayout(hboxT)
        vbox.addLayout(hboxA)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.connect(self.exit,      QtCore.SIGNAL('clicked()'), self.processQuit )
        self.connect(self.browse,    QtCore.SIGNAL('clicked()'), self.processBrowse )
        self.connect(self.display,   QtCore.SIGNAL('clicked()'), self.processDisplay )
        self.connect(self.wtd,       QtCore.SIGNAL('clicked()'), self.processWhatToDisplay )
        self.connect(self.save,      QtCore.SIGNAL('clicked()'), self.processSave )
        self.connect(self.config,    QtCore.SIGNAL('clicked()'), self.processConfig )
        self.connect(self.printfile, QtCore.SIGNAL('clicked()'), self.processPrint )
        self.connect(self.selection, QtCore.SIGNAL('clicked()'), self.processSelection )
        self.connect(self.fileEdit,  QtCore.SIGNAL('editingFinished ()'), self.processFileEdit )

        #self.setFocus()
        #self.resize(500, 300)
        print 'End of init'

    #-------------------
    # Private methods --
    #-------------------

    def moveEvent(self, e):
        #print 'moveEvent' 
        cp.confpars.posGUIMain = (self.pos().x(),self.pos().y())

    def resizeEvent(self, e):
        print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def processPrint(self):
        fname = cp.confpars.dirName+'/'+cp.confpars.fileName
        print 'Print structure of the HDF5 file:\n %s' % (fname)
        printh5.print_hdf5_file_structure(fname)

    def processQuit(self):
        print 'Begin GUIMain Quit'
        #self.drawev.quitDrawEvent()
        self.wplayer.processQuit()
        self.SHowIsOn = False
        if cp.confpars.wtdWindowIsOpen :
            cp.confpars.guiwhat.close()
        if cp.confpars.treeWindowIsOpen :
            cp.confpars.guitree.close()
        if cp.confpars.configGUIIsOpen :
            self.configGUI.close()
        if cp.confpars.selectionGUIIsOpen :
            self.guiselection.close()
        print 'Segmentation fault may happen at closing of the Main GUI window. The reason for that is not clear yet...'
              #It happens after opening/closing HDF5 Tree and WTD GUIs...
        self.close()


        
    def processBrowse(self):
        print 'Browse'
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
            self.guiselection.close()
            cp.confpars.selectionGUIIsOpen = False            
        else :                           # Open wtd window
            print 'Selection GUI: Open'
            #self.selection.setText('Close Selection')
            self.guiselection = guisel.GUISelection()
            self.guiselection.move(self.pos().__add__(QtCore.QPoint(500,330))) # open window with offset w.r.t. parent
            self.guiselection.show()
            cp.confpars.selectionGUIIsOpen = True

        
    def processConfig(self):
        print 'Configuration'
        if  cp.confpars.configGUIIsOpen :
            cp.confpars.configGUIIsOpen = False
            self.configGUI.close()
        else :    
            self.configGUI = guiconfig.GUIConfiguration()
            self.configGUI.setParent(self)
            self.configGUI.move(self.pos().__add__(QtCore.QPoint(100,330))) # open window with offset w.r.t. parent
            self.configGUI.show()
            cp.confpars.configGUIIsOpen = True

    def processSave(self):
        print 'Save'
        cp.confpars.writeParameters()

    def processWhatToDisplay(self):
        if cp.confpars.wtdWindowIsOpen : # close wtd window
            print 'What to display GUI: Close'
            #self.wtd.setText('Open')
            cp.confpars.guiwhat.close()
            cp.confpars.wtdWindowIsOpen = False            
        else :                           # Open wtd window
            print 'What to display GUI: Open'
            #self.wtd.setText('Close')
            cp.confpars.guiwhat = guiwtd.GUIWhatToDisplay()
            cp.confpars.guiwhat.move(self.pos().__add__(QtCore.QPoint(0,360))) # open window with offset w.r.t. parent
            cp.confpars.guiwhat.show()
            cp.confpars.wtdWindowIsOpen = True
            
    def processDisplay(self):
        if cp.confpars.treeWindowIsOpen : # close wtd window
            print 'What to display GUI: Close'
            #self.display.setText('Open HDF5 tree')
            cp.confpars.guitree.close()
            cp.confpars.treeWindowIsOpen = False            
        else :                           # Open wtd window
            print 'What to display GUI: Open'
            #self.display.setText('Close HDF5 tree')
            cp.confpars.guitree = guiselitems.GUISelectItems()
            #cp.confpars.guitree.setParent(self) # bypass for parent initialization in the base QWidget
            cp.confpars.guitree.move(self.pos().__add__(QtCore.QPoint(-360,0))) # open window with offset w.r.t. parent
            cp.confpars.guitree.show()
            cp.confpars.treeWindowIsOpen = True

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

    def closeEvent(self, event):
        print 'closeEvent'
        self.processQuit()

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIMain()
    ex.show()
    app.exec_()
#-----------------------------
