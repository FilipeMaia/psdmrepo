
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplayForWaveform...
#
#------------------------------------------------------------------------

"""GUI selects the CSpad Quad and Pair.

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
import GUISettingsForWaveformWindow as guiWin


#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp

#---------------------
#  Class definition --
#---------------------
class GUIWhatToDisplayForWaveform ( QtGui.QWidget ) :
    """GUI selects the CSpad Quad and Pair"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, parent=None, app=None) :
        """Constructor"""

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 500, 500, 150)
        self.setWindowTitle('Set windows for waveforms')
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False

        # see http://www.riverbankcomputing.co.uk/static/Docs/PyQt4/html/qframe.html
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(True)

        self.char_expand = u'\u25BE' # down-head triangle

        self.titNWin  = QtGui.QLabel('Number of windows:')

        self.butMenuNWin = QtGui.QPushButton(str(cp.confpars.waveformNWindows) + self.char_expand)
        self.butMenuNWin.setMaximumWidth(30)

        self.listActMenuNWin = []
        self.popupMenuNWin = QtGui.QMenu()
        for nwin in range(1,cp.confpars.waveformNWindowsMax+1) :
            self.listActMenuNWin.append(self.popupMenuNWin.addAction(str(nwin)))


        self.hboxN = QtGui.QHBoxLayout() 
        self.hboxN.addWidget(self.titNWin)
        self.hboxN.addWidget(self.butMenuNWin)
        self.hboxN.addStretch(1)     

        self.showToolTips()

        self.guiTab = guiWin.GUISettingsForWaveformWindow() # for 0th window
        self.guiTab.setMinimumHeight(150)

        self.hboxD = QtGui.QHBoxLayout()
        self.hboxD.addWidget(self.guiTab)

        self.hboxT = QtGui.QHBoxLayout()
        self.makeTabBarLayout()

        self.vboxGlobal = QtGui.QVBoxLayout()
        self.vboxGlobal.addLayout(self.hboxN)
        self.vboxGlobal.addLayout(self.hboxT)
        self.vboxGlobal.addLayout(self.hboxD)

        self.setLayout(self.vboxGlobal)

        self.connect(self.butMenuNWin,  QtCore.SIGNAL('clicked()'), self.processMenuNWin )

#        self.connect(self.editQuad,  QtCore.SIGNAL('editingFinished ()'), self.processEditQuad )
#        self.connect(self.editPair,  QtCore.SIGNAL('editingFinished ()'), self.processEditPair )


    #-------------------
    # Private methods --
    #-------------------


    def showToolTips(self):
        # Tips for buttons and fields:
        #self            .setToolTip('Click on QUAD or PAIR number using mouse left button')
        self.butMenuNWin.setToolTip('Click mouse left on this button\nand select the number of windows\nto be opened for waveforms.')


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())

    def processQuit(self):
        #print 'Quit'
        self.close()
        
    def mousePressEvent(self, event):
        print 'Click mouse left on this button.'
        #print 'event.button() = %s at position' % (event.button()),        
        #print (event.pos()),
        #print ' x=%d, y=%d' % (event.x(),event.y()),        
        #print ' global x=%d, y=%d' % (event.globalX(),event.globalY())

    def processMenuNWin(self):
        print 'MenuNWin'
        actionSelected = self.popupMenuNWin.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        for action in self.listActMenuNWin :
            cp.confpars.waveformNWindows = int(actionSelected.text())
            if actionSelected == action : self.butMenuNWin.setText( str(cp.confpars.waveformNWindows) + self.char_expand )
        self.makeTabBarLayout(mode=1)


    def makeTabBarLayout(self,mode=None) :

        if mode != None : self.tabBar.close()
        self.tabBar = QtGui.QTabBar()
        #self.tabBar.setMovable(True) 
        for window in range(cp.confpars.waveformNWindows) :

            indTab = self.tabBar.addTab( 'Win:' + str(window+1) )
            self.tabBar.setTabTextColor(indTab,QtGui.QColor('red'))
            #self.tabBar.setShape(QtGui.QTabBar.RoundedWest)
            
        #self.hboxT = QtGui.QHBoxLayout() # it is already defined and added in layout
        self.hboxT.addWidget(self.tabBar) 
        self.connect(self.tabBar,       QtCore.SIGNAL('currentChanged(int)'), self.processTabBar)


    #def resetGlobalLayout(self):
    #   #self.vboxGlobal.insertLayout   (0, self.hboxN)
    #    self.vboxGlobal.insertLayout   (1, self.vboxT)
    #   #self.vboxGlobal.insertaddLayout(2, self.hboxD)


    def processTabBar(self):
        indTab = self.tabBar.currentIndex()
        print 'TabBar index=',indTab

        #minSize = self.hboxD.minimumSize()
        self.guiTab.close()

        for window in range(cp.confpars.waveformNWindows) :
            if indTab == window :
                self.guiTab = guiWin.GUISettingsForWaveformWindow(None, window)

        self.guiTab.setMinimumHeight(150)
        self.hboxD.addWidget(self.guiTab)




#    def processEditQuad(self):    
#        cp.confpars.cspadQuad = int(self.editQuad.displayText())
#        print 'Set quad: ', cp.confpars.cspadQuad

#    def processEditPair(self):    
#        cp.confpars.cspadPair = int(self.editPair.displayText())
#        print 'Set pair: ', cp.confpars.cspadPair

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
    ex  = GUIWhatToDisplayForWaveform()
    ex.show()
    app.exec_()
#-----------------------------
