
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgGUI...
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
import time   # for sleep(sec)

from PyQt4 import QtGui, QtCore
#import ImgGUIWindow as guiwin

#-----------------------------
# Imports for other modules --
#-----------------------------
import ImgConfigParameters as icp
import ImgGUISpectrum as igspec

#---------------------
#  Class definition --
#---------------------
class ImgGUI ( QtGui.QWidget ) :
    """This GUI defines the parameters for camera image plots"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, parent=None) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 500, 500, 250)
        self.setWindowTitle('Image GUI')
        self.setFrame()
        self.palette       = QtGui.QPalette()

        self.hboxT = QtGui.QHBoxLayout()
        self.hboxB = QtGui.QHBoxLayout()

        self.makeTabBarTop()
        self.makeTabBarBot()

        self.minHeight = 100
        self.maxHeight = 250

        self.guiWin = igspec.ImgGUISpectrum() # QtGui.QTextEdit('First')
        self.guiWin.setMinimumHeight(self.minHeight)
        self.guiWin.setMaximumHeight(self.maxHeight)

        self.tabBarTop.setCurrentIndex(self.indTabSpec)
        self.tabBarBot.setCurrentIndex(self.indTabEmpB)

        self.hboxD = QtGui.QHBoxLayout()
        self.hboxD.addWidget(self.guiWin)

        self.vboxGlobal = QtGui.QVBoxLayout()
        self.vboxGlobal.addLayout(self.hboxT)
        self.vboxGlobal.addLayout(self.hboxD)
        self.vboxGlobal.addLayout(self.hboxB)

        self.setLayout(self.vboxGlobal)

        self.showToolTips()

        icp.imgconfpars.ImgGUIIsOpen = True


    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        # Tips for buttons and fields:
        #self            .setToolTip('Click on QUAD or PAIR number using mouse left button')
        #self.butMenuNWin.setToolTip('Click on mouse left button\n' +
        #                            'and select the number of windows.')
        pass


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
        print 'closeEvent for ImgGUI'
        icp.imgconfpars.ImgGUIIsOpen = False


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


    def makeTabBarTop(self,mode=None) :

        if mode != None : self.tabBarTop.close()
        self.tabBarTop = QtGui.QTabBar()
        #self.tabBar.setMovable(True) 

        self.indTabSpec = self.tabBarTop.addTab( 'Spectrum' )
        self.indTabProX = self.tabBarTop.addTab( 'Proj.X' )
        self.indTabProY = self.tabBarTop.addTab( 'Proj.Y' )
        self.indTabProR = self.tabBarTop.addTab( 'Proj.R' )
        self.indTabProP = self.tabBarTop.addTab( 'Proj.P' )
        self.indTabEmpt = self.tabBarTop.addTab( 5*' ' )
        self.tabBarTop.setTabEnabled(self.indTabEmpt,False)

        self.tabBarTop.setTabTextColor(self.indTabSpec,QtGui.QColor('red'))
        self.tabBarTop.setTabTextColor(self.indTabProX,QtGui.QColor('blue'))
        self.tabBarTop.setTabTextColor(self.indTabProY,QtGui.QColor('green'))
        self.tabBarTop.setTabTextColor(self.indTabProR,QtGui.QColor('magenta'))
        self.tabBarTop.setTabTextColor(self.indTabProP,QtGui.QColor('red'))
        self.tabBarTop.setShape(QtGui.QTabBar.RoundedNorth)
 
        #self.tabBarTop.setShape(QtGui.QTabBar.RoundedWest)
            
        #self.hboxT = QtGui.QHBoxLayout() # it is already defined and added in layout
        self.hboxT.addWidget(self.tabBarTop) 
        self.connect(self.tabBarTop, QtCore.SIGNAL('currentChanged(int)'), self.processTabBarTop)


    def processTabBarTop(self):

        indTab = self.tabBarTop.currentIndex()
        #print 'TabBar index=',indTab

        #self.hboxD.removeWidget(self.guiWin) 
        try :
            self.guiWin.close()
        except AttributeError :
            pass

        if indTab == self.indTabEmpt : return
        if indTab == self.indTabSpec : self.guiWin = igspec.ImgGUISpectrum() #QtGui.QTextEdit('Spectrum')
        if indTab == self.indTabProX : self.guiWin = QtGui.QTextEdit('Projection X')
        if indTab == self.indTabProY : self.guiWin = QtGui.QTextEdit('Projection Y')
        if indTab == self.indTabProR : self.guiWin = QtGui.QTextEdit('Projection R')
        if indTab == self.indTabProP : self.guiWin = QtGui.QTextEdit('Projection P')

        self.guiWin.setMinimumHeight(self.minHeight)
        self.guiWin.setMaximumHeight(self.maxHeight)
        self.hboxD.addWidget(self.guiWin)

        self.tabBarBot.setCurrentIndex(self.indTabEmpB)


    def makeTabBarBot(self,mode=None) :

        if mode != None : self.tabBarBot.close()
        self.tabBarBot = QtGui.QTabBar()
        #self.tabBarBot.setMovable(True) 

        self.indTabZmCp = self.tabBarBot.addTab( 'Zoom-copy' )
        self.indTabCent = self.tabBarBot.addTab( 'Center' )
        self.indTabLine = self.tabBarBot.addTab( 'Line' )
        self.indTabCirc = self.tabBarBot.addTab( 'Circ' )
        self.indTabEmpB = self.tabBarBot.addTab( 5*' ' )
        self.tabBarBot.setTabEnabled(self.indTabEmpB,False)

        self.tabBarBot.setTabTextColor(self.indTabZmCp,QtGui.QColor('magenta'))
        self.tabBarBot.setTabTextColor(self.indTabCent,QtGui.QColor('green'))
        self.tabBarBot.setTabTextColor(self.indTabLine,QtGui.QColor('red'))
        self.tabBarBot.setTabTextColor(self.indTabCirc,QtGui.QColor('blue'))
        self.tabBarBot.setShape(QtGui.QTabBar.RoundedSouth)
            
        #self.hboxT = QtGui.QHBoxLayout() # it is already defined and added in layout
        self.hboxB.addWidget(self.tabBarBot) 
        self.connect(self.tabBarBot, QtCore.SIGNAL('currentChanged(int)'), self.processTabBarBot)


    def processTabBarBot(self):

        indTab = self.tabBarBot.currentIndex()
        #print 'TabBar index=',indTab

        #self.hboxD.removeWidget(self.guiWin) 
        try :
            self.guiWin.close()
        except AttributeError :
            pass
        if indTab == self.indTabEmpB : return
        if indTab == self.indTabZmCp : self.guiWin = QtGui.QTextEdit('Zoomed copy of image')
        if indTab == self.indTabCent : self.guiWin = QtGui.QTextEdit('Set center')
        if indTab == self.indTabLine : self.guiWin = QtGui.QTextEdit('Profile along Line')
        if indTab == self.indTabCirc : self.guiWin = QtGui.QTextEdit('Profile along Circ')

        self.guiWin.setMinimumHeight(self.minHeight)
        self.guiWin.setMaximumHeight(self.maxHeight)
        self.hboxD.addWidget(self.guiWin)

        self.tabBarTop.setCurrentIndex(self.indTabEmpt)



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
#  Test
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = ImgGUI()
    ex.show()
    app.exec_()
#-----------------------------
