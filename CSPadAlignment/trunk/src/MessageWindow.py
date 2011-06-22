#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module MessageWindow...
#
#------------------------------------------------------------------------

"""Module MessageWindow for CSPadAlignment package

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

import sys
import os

from PyQt4 import QtGui, QtCore
from time import localtime, strftime
import ImageParameters   as imp

class MessageWindow( QtGui.QWidget ) :
    """Example of GUI with buttons, labels, input of numbers etc.""" 

    def __init__ (self, parent=None) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(10, 10, 500, 40)
        self.setWindowTitle('Message window')

        # see http://www.riverbankcomputing.co.uk/static/Docs/PyQt4/html/qframe.html
        self.frame = QtGui.QFrame(self)
        # Box, Pannel, VLine, HLine, NoFrame | Raised, Sunken
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setVisible(True)
        self.frame.setGeometry(self.rect())

        time_stamp = strftime("%Y-%m-%d %H:%M:%S %Z", localtime())

        self.wMessageBox = QtGui.QTextEdit()
        self.wMessageBox.setText(time_stamp + ': Program messages will be printed in this box.')
        self.wMessageBox.setReadOnly(True)
        #self.wMessageBox.setTextBackgroundColor (QtGui.QColor(255, 0, 0, 127))
        #self.wMessageBox.setTextColor (QtGui.QColor(0, 0, 255, 127))

        self.wMessageBox.setMaximumHeight(100)
        self.wMessageBox.setMinimumHeight(40)
        #self.wMessageBox.setFixedHeight(40)
        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.wMessageBox)         
        self.setLayout(self.hbox)

        imp.impars.msgw = self

    def addMessage(self,message):
        time_stamp = strftime("%H:%M:%S", localtime())
        self.wMessageBox.append(time_stamp + ' ' + message)
    
    def saveLogOfMessagesInFile(self):        
        doc = self.wMessageBox.document()
        doctxt = doc.toPlainText()
        print 'Write log in file:',imp.impars.logFileName 
        print doctxt
        f=open(imp.impars.logFileName,'w')
        f.write(doctxt)
        f.close()

    def processExit(self):
        print 'Exit'
        self.close() # This call closeEvent()

    def closeEvent(self, event):
        print 'closeEvent'

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

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
    widget = MessageWindow()
    for mess in range(50) :
        widget.addMessage('Message number '+str(mess))

    widget.show()
    app.exec_()

if __name__ == '__main__':
    main()

#----------------------------------
