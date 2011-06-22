#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIHorizontal...
#
#------------------------------------------------------------------------

"""Module GUIHorizontal for CSPadAlignment package

CSPadAlignment package is intended to check quality of the CSPad alignment
using image of wires illuminated by flat field.
Shadow of wires are compared with a set of straight lines, which can be
interactively adjusted.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$
#------------------------------
#!/usr/bin/env python
#----------------------------------

import sys
import os

from PyQt4 import QtGui, QtCore
import ImageParameters   as imp


class GUIHorizontal( QtGui.QWidget ) :
    """Example of GUI with buttons, labels, input of numbers etc.""" 

    def __init__ (self, parent=None, app=None) :
        """Constructor"""

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 50, 600, 30)
        self.setWindowTitle('GUI with buttons')

        # see http://www.riverbankcomputing.co.uk/static/Docs/PyQt4/html/qframe.html
        self.frame = QtGui.QFrame(self)
        # Box, Pannel, VLine, HLine, NoFrame | Raised, Sunken
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setVisible(True)
        self.frame.setGeometry(self.rect())

        self.titFile  = QtGui.QLabel('File:')

        #self.fileEdit = QtGui.QLineEdit('./')
        #homedir = os.getenv('HOME') # /reg/neh/home/dubrovin/
        #self.fileEdit = QtGui.QLineEdit(homedir + '/.evtdispconfig')
        self.fileEdit = QtGui.QLineEdit(imp.impars.imgDirName + '/' + imp.impars.imgFileName)

        self.butBrowse   = QtGui.QPushButton("Browse")    
        self.butExit     = QtGui.QPushButton("Exit")
        #self.spaninc  = QtGui.QPushButton(u'\u25B6') # right-head triangle
        #self.spandec  = QtGui.QPushButton(u'\u25C0') # left-head triangle

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.titFile)
        self.hbox.addWidget(self.fileEdit)
        self.hbox.addWidget(self.butBrowse)
        self.hbox.addWidget(self.butExit)

        if parent == None :
            self.setLayout(self.hbox)

        #self.connect(self.slider, QtCore.SIGNAL('valueChanged(int)'), lcd, QtCore.SLOT('display(int)') )
        #self.connect(self,   QtCore.SIGNAL('closeGUIApp()'), QtCore.SLOT('close()') )
        #self.connect(self.exit,     QtCore.SIGNAL('clicked()'), QtCore.SLOT('close()') )
        self.connect(self.butExit,      QtCore.SIGNAL('clicked()'), self.processExit )
        self.connect(self.butBrowse,    QtCore.SIGNAL('clicked()'), self.processBrowse )

    def getBoxLayout(self):
        print 'getBoxLayout'
        return self.hbox
    
    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def processExit(self):
        print 'Exit'
        self.close() # call closeEvent(...) 

    def closeEvent(self, event):
        print 'closeEvent'
        imp.impars.msgw.addMessage('Save the log file ' + imp.impars.logFileName)        
        imp.impars.msgw.saveLogOfMessagesInFile()
        imp.impars.gmw.on_quit() # close main window

    def processSave(self):
        print 'Save'

    def processBrowse_v0(self):
        imp.impars.msgw.addMessage('File name')
        print 'Browse'
        self.path = str(self.fileEdit.displayText())
        self.dirname,self.fname = os.path.split(self.path)
        print 'dirName  : %s' % (self.dirname)
        print 'fileName : %s' % (self.fname)
        self.path = QtGui.QFileDialog.getOpenFileName(self,'Open file',self.dirname)
        #self.path = imp.impars.imgDirName + '/' + imp.impars.imgFileName
        self.fileEdit.setText(self.path)

    def processBrowse(self):
        print 'Browse'
        self.path = str(self.fileEdit.displayText())
        self.dirName,self.fileName = os.path.split(self.path)
        print 'dirName  : %s' % (self.dirName)
        print 'fileName : %s' % (self.fileName)
        self.path = QtGui.QFileDialog.getOpenFileName(self,'Open file',self.dirName)
        self.dirName,self.fileName = os.path.split(str(self.path))
        #self.path = cp.confpars.confParsDirName + '/' + cp.confpars.confParsFileName
        #self.path = self.dirName+'/'+self.fileName
        if self.dirName == '' or self.fileName == '' :
            print 'Input dirName or fileName is empty... use default values'  
        else :
            self.fileEdit.setText(self.path)
            imp.impars.imgDirName  = self.dirName
            imp.impars.imgFileName = self.fileName

    def processFileEdit(self):
        print 'FileEdit'
        self.path = str(self.fileEdit.displayText())
        imp.impars.imgDirName,imp.impars.imgFileName = os.path.split(self.path)
        print 'Set dirName  : %s' % (imp.impars.imgDirName)
        print 'Set fileName : %s' % (imp.impars.imgFileName)
 
    def mousePressEvent(self, event):
        print 'Do not click on mouse just for fun!\n'
        print 'event.button() = %s at position' % (event.button()),        
        #print (event.pos()),
        print ' x=%d, y=%d' % (event.x(),event.y()),        
        print ' global x=%d, y=%d' % (event.globalX(),event.globalY())
        #self.emit(QtCore.SIGNAL('closeGUIApp()'))

    #http://doc.qt.nokia.com/4.6/qt.html#Key-enum
    def keyPressEvent(self, event):
        print 'event.key() = %s' % (event.key())
        if event.key() == QtCore.Qt.Key_Escape:
            print 'Escape'

        if event.key() == QtCore.Qt.Key_B:
            print 'event.key() = %s' % (QtCore.Qt.Key_B)

        if event.key() == QtCore.Qt.Key_Return:
            print 'event.key() = Return'

        if event.key() == QtCore.Qt.Key_Home:
            print 'event.key() = Home'

#----------------------------------

def main():
    app = QtGui.QApplication(sys.argv)
    widget = GUIHorizontal()
    widget.show()
    app.exec_()

if __name__ == '__main__':
    main()

#----------------------------------
