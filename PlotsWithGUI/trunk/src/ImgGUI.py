
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

import ImgGUIMode     as igm
import ImgGUIPlayer   as igplr
import ImgGUISpectrum as igspec
import ImgGUIProfile  as igprof
import ImgGUIZoom     as igzoom
import ImgGUIProjXY   as igprxy
import ImgGUIProjRP   as igprrp
import ImgGUICenter   as igcent


#---------------------
#  Class definition --
#---------------------
class ImgGUI ( QtGui.QWidget ) :
    """This GUI defines the parameters for camera image plots"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, icp=None) :
        """Constructor"""

        QtGui.QWidget.__init__(self, parent=None)

        self.icp      = icp
        self.icp.wgui = self

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

        self.gui_player = igplr.ImgGUIPlayer(self.icp)
        self.gui_mode   = igm.ImgGUIMode(self.icp)

        self.guiWin = igspec.ImgGUISpectrum(self.icp) # QtGui.QTextEdit('First')
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
        self.vboxGlobal.addWidget(self.gui_mode)
        self.vboxGlobal.addWidget(self.gui_player)

        self.setLayout(self.vboxGlobal)

        self.showToolTips()

        #icp.imgconfpars.ImgGUIIsOpen = True
        self.gui_mode.setStatus()

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
        #icp.imgconfpars.ImgGUIIsOpen = False


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
        self.indTabProf = self.tabBarTop.addTab( 'Profile' )
        self.indTabPrXY = self.tabBarTop.addTab( 'Proj. X-Y' )
        self.indTabPrRP = self.tabBarTop.addTab( 'Proj. R-Phi' )
        self.indTabEmpT = self.tabBarTop.addTab( 5*' ' )
        self.tabBarTop.setTabEnabled(self.indTabEmpT,False)

        self.tabBarTop.setTabTextColor(self.indTabSpec,QtGui.QColor('red'))
        self.tabBarTop.setTabTextColor(self.indTabProf,QtGui.QColor('magenta'))
        self.tabBarTop.setTabTextColor(self.indTabPrXY,QtGui.QColor('blue'))
        self.tabBarTop.setTabTextColor(self.indTabPrRP,QtGui.QColor('red'))
        self.tabBarTop.setShape(QtGui.QTabBar.RoundedNorth)
            
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

        if indTab == self.indTabEmpT : return
        if indTab == self.indTabSpec : self.guiWin = igspec.ImgGUISpectrum(self.icp) #QtGui.QTextEdit('Spectrum')
        if indTab == self.indTabProf : self.guiWin = igprof.ImgGUIProfile (self.icp)
        if indTab == self.indTabPrXY : self.guiWin = igprxy.ImgGUIProjXY  (self.icp)
        if indTab == self.indTabPrRP : self.guiWin = igprrp.ImgGUIProjRP  (self.icp)
        #if indTab == self.indTabPrRP : self.guiWin = QtGui.QTextEdit('Projections R-Phi')

        self.guiWin.setMinimumHeight(self.minHeight)
        self.guiWin.setMaximumHeight(self.maxHeight)
        self.hboxD.addWidget(self.guiWin)

        self.tabBarBot.setCurrentIndex(self.indTabEmpB)
        self.gui_mode.setStatus()


    def makeTabBarBot(self,mode=None) :

        if mode != None : self.tabBarBot.close()
        self.tabBarBot = QtGui.QTabBar()
        #self.tabBarBot.setMovable(True) 

        self.indTabZoom = self.tabBarBot.addTab( 'Zoom' )
        self.indTabCent = self.tabBarBot.addTab( 'Center' )
        #self.indTabLine = self.tabBarBot.addTab( 'Line' )
        #self.indTabCirc = self.tabBarBot.addTab( 'Circ' )
        self.indTabEmpB = self.tabBarBot.addTab( 5*' ' )
        self.tabBarBot.setTabEnabled(self.indTabEmpB,False)

        self.tabBarBot.setTabTextColor(self.indTabZoom,QtGui.QColor('magenta'))
        self.tabBarBot.setTabTextColor(self.indTabCent,QtGui.QColor('green'))
        #self.tabBarBot.setTabTextColor(self.indTabLine,QtGui.QColor('red'))
        #self.tabBarBot.setTabTextColor(self.indTabCirc,QtGui.QColor('blue'))
        self.tabBarBot.setShape(QtGui.QTabBar.RoundedSouth)
            
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
        if indTab == self.indTabZoom : self.guiWin = igzoom.ImgGUIZoom   (self.icp)
        if indTab == self.indTabCent : self.guiWin = igcent.ImgGUICenter (self.icp)
        #if indTab == self.indTabCent : self.guiWin = QtGui.QTextEdit('Set center')
        #if indTab == self.indTabLine : self.guiWin = QtGui.QTextEdit('Profile along Line')
        #if indTab == self.indTabCirc : self.guiWin = QtGui.QTextEdit('Profile along Circ')

        self.guiWin.setMinimumHeight(self.minHeight)
        self.guiWin.setMaximumHeight(self.maxHeight)
        self.hboxD.addWidget(self.guiWin)

        self.tabBarTop.setCurrentIndex(self.indTabEmpT)
        self.gui_mode.setStatus()


    def setTabBarCenter(self) :
        self.tabBarBot.setCurrentIndex(self.indTabEmpB)
        self.tabBarBot.setCurrentIndex(self.indTabCent)
        #self.processTabBarBot()


    def get_control(self) :
        return self.icp.control


    def keyPressEvent(self, event):
        #print 'ImgGUI : event.key() = %s' % (event.key())
        if event.key() == QtCore.Qt.Key_Escape:
            pass

        if event.key() == QtCore.Qt.Key_S:
            #print 'ImgGUI : S is pressed' 
            self.get_control().signal_save()

        if event.key() == QtCore.Qt.Key_Q:
            #print 'ImgGUI : Q is pressed' 
            self.get_control().signal_quit()

        if event.key() == QtCore.Qt.Key_P:
            #print 'ImgGUI : Q is pressed' 
            self.get_control().signal_print()

        if event.key() == QtCore.Qt.Key_Return:
            #print 'event.key() = Return'
            pass

        if event.key() == QtCore.Qt.Key_Home:
            #print 'event.key() = Home'
            pass


#-----------------------------
#  Test
#
if __name__ == "__main__" :

    import ImgConfigParameters as gicp
    icp = gicp.giconfpars.addImgConfigPars( None )

    app = QtGui.QApplication(sys.argv)
    w  = ImgGUI(icp)
    w.move(QtCore.QPoint(50,50))
    w.show()
    app.exec_()

#-----------------------------
