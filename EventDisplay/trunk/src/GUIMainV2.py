
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMainV2...
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
import GUIPlayer        as guiplr
import GUIWhatToDisplay as guiwtd
import GUISelectItems   as guiselitems
import GUIConfiguration as guiconfig
import ConfigParameters as cp
#import DrawEvent        as drev
import PrintHDF5        as printh5 # for my print_group(g,offset)

#---------------------
#  Class definition --
#---------------------
class GUIMainV2 ( QtGui.QWidget ) :
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

        self.setGeometry(370, 10, 500, 300)
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

        self.titFile  = QtGui.QLabel('File:')
        self.titTree  = QtGui.QLabel('HDF5 Tree GUI')

        self.fileEdit = QtGui.QLineEdit(cp.confpars.dirName+'/'+cp.confpars.fileName)

        self.browse   = QtGui.QPushButton("Browse")    
        self.printfile= QtGui.QPushButton("Print HDF5 structure")    
        self.display  = QtGui.QPushButton("Open")
        self.wtd      = QtGui.QPushButton("What to display")
        self.config   = QtGui.QPushButton("Configuration")
        self.save     = QtGui.QPushButton("Save")

        self.selection= QtGui.QPushButton("Selection")
        self.exit     = QtGui.QPushButton("Exit")
        self.save   .setMaximumWidth(40)   

        hboxF = QtGui.QHBoxLayout()
        hboxF.addWidget(self.titFile)
        hboxF.addWidget(self.fileEdit)
        hboxF.addWidget(self.browse)

        hboxC = QtGui.QHBoxLayout()
#        hboxC.addWidget(self.printfile)
        hboxC.addStretch(2)
        hboxC.addWidget(self.titTree)
        hboxC.addWidget(self.display)
        
        hboxE = QtGui.QHBoxLayout()
        #hboxE.addWidget(self.selection)
        hboxE.addWidget(self.wtd)
        hboxE.addStretch(2)
        hboxE.addWidget(self.config)
        hboxE.addWidget(self.save)

        self.wplayer = guiplr.GUIPlayer()
        hboxT = QtGui.QHBoxLayout() 
        hboxT.addWidget(self.wplayer)

        hbox = QtGui.QHBoxLayout()
        #hbox.addWidget(self.closeplts)
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
        print 'moveEvent' 
        cp.confpars.posGUIMain = (self.pos().x(),self.pos().y())

    def resizeEvent(self, e):
        print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def processPrint(self):
        fname = cp.confpars.dirName+'/'+cp.confpars.fileName
        print 'Print structure of the HDF5 file:\n %s' % (fname)
        printh5.print_hdf5_file_structure(fname)

    def processQuit(self):
        print 'Quit'
        #self.drawev.quitDrawEvent()
        self.wplayer.processQuit()
        self.SHowIsOn = False
        if cp.confpars.wtdWindowIsOpen == True :
            self.guiwhat.close()
        if cp.confpars.treeWindowIsOpen == True :
            self.guitree.close()
        if cp.confpars.configGUIIsOpen == True :
            self.configGUI.close()
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
        print 'Is empty yet...'
        
    def processConfig(self):
        print 'Configuration'
        if  cp.confpars.configGUIIsOpen :
            cp.confpars.configGUIIsOpen = False
            self.configGUI.close()
        else :    
            self.configGUI = guiconfig.GUIConfiguration()
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
            self.guiwhat.close()
            cp.confpars.wtdWindowIsOpen = False            
        else :                           # Open wtd window
            print 'What to display GUI: Open'
            #self.wtd.setText('Close')
            self.guiwhat = guiwtd.GUIWhatToDisplay()
            self.guiwhat.move(self.pos().__add__(QtCore.QPoint(0,330))) # open window with offset w.r.t. parent
            self.guiwhat.show()
            cp.confpars.wtdWindowIsOpen = True

    def processDisplay(self):
        if cp.confpars.treeWindowIsOpen : # close wtd window
            print 'What to display GUI: Close'
            self.display.setText('Open')
            self.guitree.close()
            cp.confpars.treeWindowIsOpen = False            
        else :                           # Open wtd window
            print 'What to display GUI: Open'
            self.display.setText('Close')
            self.guitree = guiselitems.GUISelectItems(self)
            self.guitree.move(self.pos().__add__(QtCore.QPoint(-360,0))) # open window with offset w.r.t. parent
            self.guitree.show()
            cp.confpars.treeWindowIsOpen = True

    def mousePressEvent(self, event):
        print 'Do not click on mouse just for fun!\n'
        print 'event.button() = %s at position' % (event.button()),        
        #print (event.pos()),
        print ' x=%d, y=%d' % (event.x(),event.y()),        
        print ' global x=%d, y=%d' % (event.globalX(),event.globalY())
        #self.emit(QtCore.SIGNAL('closeGUIApp()'))

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
    ex  = GUIMainV2()
    ex.show()
    app.exec_()
#-----------------------------
