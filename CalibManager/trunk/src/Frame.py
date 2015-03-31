
'''
Created on Dec 3, 2014

@author: Mikhail
'''

import sys
#from PyQt4.Qt import QFrame
from PyQt4 import QtGui, QtCore

class Frame(QtGui.QFrame):
    """ class Frame inherits from QFrame and sets its parameters.

    QFrame inherits from QWidget and hence Frame can be used in stead of QWidget
    """
    def __init__(self, parent=None, lw=0, mlw=1, vis=True, style=QtGui.QFrame.Box | QtGui.QFrame.Sunken):
        QtGui.QFrame.__init__(self, parent)
        self.parent = parent
        self.setFrame(lw, mlw, vis, style)


    def setFrame(self, lw=0, mlw=1, vis=False, style=QtGui.QFrame.Box | QtGui.QFrame.Sunken):
        self.setFrameStyle(style) #Box, Panel | Sunken, Raised 
        self.setLineWidth(lw)
        self.setMidLineWidth(mlw)
        self.setBoarderVisible(vis) 
        #self.setGeometry(self.parent.rect())


    def setBoarderVisible(self, vis=True) :
        if vis : self.setFrameShape(QtGui.QFrame.Box)
        else   : self.setFrameShape(QtGui.QFrame.NoFrame)
    
#    def resizeEvent(self, e):
#        print 'resizeEvent'
#        #self.setGeometry(self.parent.rect())

#---------------------------
# TEST AND EXAMPLE OF USAGE
#---------------------------

class GUILabel(QtGui.QLabel, Frame):
    def __init__(self, parent=None):
        Frame       .__init__(self, parent, mlw=5)
        #QtGui.QLabel.__init__(self, QtCore.QString('label'), parent)
        self.setText('GUILabel set')

  
class GUIWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        #super(GUIWidget, self).__init__(parent)
        but = QtGui.QPushButton('Button', self)
        but.move(30,20)


class GUIWidgetFrame(Frame, QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        Frame        .__init__(self, parent, mlw=5)
        but = QtGui.QPushButton('Button', self)
        but.move(20,10)


# Use Frame in stead of QWidget
class GUIFrame(Frame):
    def __init__(self, parent=None):
        #Frame.__init__(self, parent, mlw=5)
        Frame.__init__(self, mlw=30)
        but = QtGui.QPushButton('Button', self)
        but.move(30,20)

#-----------------------------
if __name__ == "__main__" :
    
    app = QtGui.QApplication(sys.argv)

    ##w = QtGui.QTextEdit()
    #w = QtGui.QLabel('QLabel')
    #fb = Frame(w, lw=0, mlw=3, vis=True)
    #w = GUILabel()
    #w = GUIWidgetFrame()
    #w = GUIWidget()
    w = GUIFrame()

    w.setWindowTitle('GUIWidget')
    w.setGeometry(200, 500, 200, 100)
    w.show()
    app.exec_()

#-----------------------------
