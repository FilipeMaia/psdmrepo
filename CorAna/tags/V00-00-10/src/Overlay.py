#!/usr/bin/env python

#----------------------------------

import sys
#from PyQt4.QtCore import Qt
#from PyQt4.QtGui import QtGui
from PyQt4 import QtGui, QtCore

class Overlay(QtGui.QWidget):
 
    def __init__(self, parent = None, text='xxx'):
 
        QtGui.QWidget.__init__(self, parent)
        palette = QtGui.QPalette(self.palette())
        palette.setColor(palette.Background, QtCore.Qt.transparent)
        self.setPalette(palette)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.text = text

 
    def paintEvent(self, event): 
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        self.drawCross(qp)
        self.drawText(qp)
        qp.end()


    def drawCross(self,qp) :
        qp.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0)))
        qp.drawLine(self.width()/8, self.height()/8, 7*self.width()/8, 7*self.height()/8)
        qp.drawLine(self.width()/8, 7*self.height()/8, 7*self.width()/8, self.height()/8)


    def drawText(self,qp) :
        qp.setFont(QtGui.QFont('Decorative', 10))
        #qp.setPen(QtCore.Qt.red)
        qp.setPen(QtGui.QPen(QtGui.QColor(150, 100, 50)))
        qp.drawText(10,10,self.text)

 
#----------------------------------
# TEST stuff:

class MyWindow(QtGui.QMainWindow):
 
    def __init__(self, parent = None):
 
        QtGui.QMainWindow.__init__(self, parent)
 
        widget = QtGui.QWidget(self)
        self.editor = QtGui.QTextEdit()
        layout = QtGui.QGridLayout(widget)
        layout.addWidget(self.editor,            0, 0, 1, 2)
        layout.addWidget(QtGui.QPushButton("Refresh"), 1, 0)
        layout.addWidget(QtGui.QPushButton("Cancel"),  1, 1)
 
        self.setCentralWidget(widget)
        self.overlay = Overlay(self.centralWidget())
 
    def resizeEvent(self, event):
 
        self.overlay.resize(event.size())
        event.accept()
 
#----------------------------------
 
if __name__ == "__main__": 
    app = QtGui.QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())

#----------------------------------
