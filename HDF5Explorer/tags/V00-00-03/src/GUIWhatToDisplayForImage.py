
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplayForImage...
#
#------------------------------------------------------------------------

"""This GUI defines the parameters for camera image plots.

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
import GUIWhatToDisplayForImageWindow as guiwin

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp

#---------------------
#  Class definition --
#---------------------
class GUIWhatToDisplayForImage ( QtGui.QWidget ) :
    """This GUI defines the parameters for camera image plots"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, parent=None, app=None) :
        """Constructor"""

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 500, 500, 250)
        self.setWindowTitle('What to display for image GUI')
        self.palette = QtGui.QPalette()

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(True)

        self.char_expand = u'\u25BE' # down-head triangle

        self.titNWin  = QtGui.QLabel('Number of windows:')

        self.butMenuNWin = QtGui.QPushButton(str(cp.confpars.imageNWindows) + self.char_expand)
        self.butMenuNWin.setMaximumWidth(30)

        self.popupMenuNWin = QtGui.QMenu()
        for nwin in range(1,cp.confpars.imageNWindowsMax+1) :
            self.popupMenuNWin.addAction(str(nwin))

        self.hboxN = QtGui.QHBoxLayout() 
        self.hboxN.addWidget(self.titNWin)
        self.hboxN.addWidget(self.butMenuNWin)
        self.hboxN.addStretch(1)     

        self.hboxT = QtGui.QHBoxLayout()
        self.makeTabBarLayout()

        self.guiWin = guiwin.GUIWhatToDisplayForImageWindow() # for 0th window
        self.guiWin.setMinimumHeight(150)

        self.hboxD = QtGui.QHBoxLayout()
        self.hboxD.addWidget(self.guiWin)

        self.vboxGlobal = QtGui.QVBoxLayout()
        self.vboxGlobal.addLayout(self.hboxN)
        self.vboxGlobal.addLayout(self.hboxT)
        self.vboxGlobal.addLayout(self.hboxD)

        self.setLayout(self.vboxGlobal)

        self.connect(self.butMenuNWin,  QtCore.SIGNAL('clicked()'), self.processMenuNWin )

        self.showToolTips()



    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self            .setToolTip('Click on QUAD or PAIR number using mouse left button')
        self.butMenuNWin.setToolTip('Click on mouse left button\n' +
                                    'and select the number of windows.')

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def closeEvent(self, event):
        #print 'closeEvent'
        cp.confpars.wtdIMWindowIsOpen = False


    def processQuit(self):
        #print 'Quit button'
        self.close()

        
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


    def processMenuNWin(self):
        #print 'MenuNWin'
        actionSelected = self.popupMenuNWin.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        cp.confpars.imageNWindows = int(actionSelected.text())
        self.butMenuNWin.setText( str(cp.confpars.imageNWindows) + self.char_expand )
        self.makeTabBarLayout(mode=1)


    def makeTabBarLayout(self,mode=None) :

        if mode != None : self.tabBar.close()
        self.tabBar = QtGui.QTabBar()
        #self.tabBar.setMovable(True) 
        for window in range(cp.confpars.imageNWindows) :

            indTab = self.tabBar.addTab( 'Win:' + str(window+1) )
            self.tabBar.setTabTextColor(indTab,QtGui.QColor('blue'))
            #self.tabBar.setShape(QtGui.QTabBar.RoundedWest)
            
        #self.hboxT = QtGui.QHBoxLayout() # it is already defined and added in layout
        self.hboxT.addWidget(self.tabBar) 
        self.connect(self.tabBar, QtCore.SIGNAL('currentChanged(int)'), self.processTabBar)


    def processTabBar(self):
        indTab = self.tabBar.currentIndex()
        #print 'TabBar index=',indTab

        self.guiWin.close()
        for window in range(cp.confpars.imageNWindows) :
            if indTab == window :
                self.guiWin = guiwin.GUIWhatToDisplayForImageWindow(None, window)

        self.guiWin.setMinimumHeight(150)
        self.guiWin.setMaximumHeight(200)
        self.hboxD.addWidget(self.guiWin)


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
    ex  = GUIWhatToDisplayForImage()
    ex.show()
    app.exec_()
#-----------------------------
