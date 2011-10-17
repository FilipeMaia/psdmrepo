
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIHelp...
#
#------------------------------------------------------------------------

"""This GUI is an envelop for all data set windows.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.
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
import ConfigParameters as cp

#---------------------
#  Class definition --
#---------------------
class GUIHelp ( QtGui.QWidget ) :
    """This GUI is an envelop for all data set windows"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, parent=None, app=None) :
        """Constructor"""

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(500, 200, 300, 100)
        self.setWindowTitle('Help')
        self.palette = QtGui.QPalette()
        self.setFrame()

        message = '\n HDF5Analysis' \
                + '\n V00-00-01 from 09/16/2011' \
                + '\n'

        self.labText = QtGui.QLabel(message)

        self.butWebPage = QtGui.QPushButton('Open manual')
        self.butQuit    = QtGui.QPushButton('Quit')
        #self.butQuit.setMaximumWidth(30)

        self.hboxB = QtGui.QHBoxLayout() 
        self.hboxB.addWidget(self.butWebPage)
        self.hboxB.addStretch(1)     
        self.hboxB.addWidget(self.butQuit)

        self.vboxGlobal = QtGui.QVBoxLayout()
        self.vboxGlobal.addWidget(self.labText)
        self.vboxGlobal.addStretch(1)     
        self.vboxGlobal.addLayout(self.hboxB)

        self.setLayout(self.vboxGlobal)

        self.connect(self.butWebPage,  QtCore.SIGNAL('clicked()'), self.processWebPage )
        self.connect(self.butQuit,     QtCore.SIGNAL('clicked()'), self.processQuit )

        self.showToolTips()

        cp.confpars.helpGUIIsOpen = True

    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        self.butQuit.setToolTip('Quit this window.')


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


    def closeEvent(self, event):
        #print 'closeEvent'
        cp.confpars.helpGUIIsOpen = False


    def processQuit(self):
        #print 'Quit button'
        self.close()


    def processWebPage(self):
        strWebPage = 'https://confluence.slac.stanford.edu/display/PCDS/HDF5+Explorer'
        print 'Open the web page in Firefox : ' + strWebPage
        os.system('firefox ' + strWebPage + '&') 
        #os.system('kcalc &') 

        
    def mousePressEvent(self, event):
        print 'Click on mouse',
        if   event.button() == 1 : print 'left button'
        elif event.button() == 2 : print 'right button'
        elif event.button() == 4 : print 'central button'
        else                     : print 'button', event.button(), 
        #print 'event.button() = %s at position' % (event.button()),        
        #print (event.pos()),
        #print ' x=%d, y=%d' % (event.x(),event.y()),        
        #print ' global x=%d, y=%d' % (event.globalX(),event.globalY())


    def keyPressEvent(self, event):
        print 'event.key() = %s' % (event.key())
        if event.key() == QtCore.Qt.Key_Escape:
            self.IsOn = False    
            self.close()

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
    ex  = GUIHelp()
    ex.show()
    app.exec_()
#-----------------------------
