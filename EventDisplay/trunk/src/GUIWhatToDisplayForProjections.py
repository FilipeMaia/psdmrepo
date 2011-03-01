
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplayForProjections...
#
#------------------------------------------------------------------------

"""This GUI defines the parameters for Projections.

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
import GUIWhatToDisplayForProjR   as guiprojr
import GUIWhatToDisplayForProjPhi as guiprojphi
import GUIWhatToDisplayForProjX   as guiprojx
import GUIWhatToDisplayForProjY   as guiprojy

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp

#---------------------
#  Class definition --
#---------------------
class GUIWhatToDisplayForProjections ( QtGui.QWidget ) :
    """This GUI defines the parameters for Projections"""

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, parent=None, app=None) :
        """Constructor"""

        self.myapp = app
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(200, 500, 500, 200)
        self.setWindowTitle('Projections GUI')
        self.palette = QtGui.QPalette()

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(True)

        self.titProjections  = QtGui.QLabel('Set parameters for projections:')
        self.titCenter       = QtGui.QLabel('Center X and Y:')

        self.editProjCenterX = QtGui.QLineEdit(str(cp.confpars.projCenterX))
        self.editProjCenterY = QtGui.QLineEdit(str(cp.confpars.projCenterY))
        self.editProjCenterX.setValidator(QtGui.QIntValidator(0, 2000,self))
        self.editProjCenterY.setValidator(QtGui.QIntValidator(0, 2000,self))
        self.editProjCenterX.setMaximumWidth(50)
        self.editProjCenterY.setMaximumWidth(50)

        self.hboxN = QtGui.QHBoxLayout() 
        self.hboxN.addWidget(self.titProjections)
        self.hboxN.addStretch(1)     
        self.hboxN.addWidget(self.titCenter)
        self.hboxN.addWidget(self.editProjCenterX)
        self.hboxN.addWidget(self.editProjCenterY)

        self.hboxT = QtGui.QHBoxLayout()
        self.makeTabBarLayout()

        self.guiWin = QtGui.QLabel('Place holder.')
        self.guiWin.setMinimumHeight(150)

        self.hboxD = QtGui.QHBoxLayout()
        self.hboxD.addWidget(self.guiWin)

        self.vboxG = QtGui.QVBoxLayout()
        self.vboxG.addLayout(self.hboxN)
        self.vboxG.addLayout(self.hboxT)
        self.vboxG.addLayout(self.hboxD)

        self.setLayout(self.vboxG)

#        self.connect(self.butMenuNWin,  QtCore.SIGNAL('clicked()'), self.processMenuNWin )
        self.connect(self.editProjCenterX, QtCore.SIGNAL('editingFinished ()'), self.processEditProjCenterX)
        self.connect(self.editProjCenterY, QtCore.SIGNAL('editingFinished ()'), self.processEditProjCenterY)

        self.showToolTips()

        self.processTabBar()



    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        pass
        # Tips for buttons and fields:
        #self            .setToolTip('Click on QUAD or PAIR number using mouse left button')
        #self.butMenuNWin.setToolTip('Click mouse left on this button\nand select the number of windows\nfor selection.')


    def resizeEvent(self, e):
        #print 'resizeEvent'
        self.frame.setGeometry(self.rect())


    def processQuit(self):
        print 'Quit'
        #cp.confpars.selectionGUIIsOpen = False
        self.close()


    def closeEvent(self, event):
        print 'closeEvent'
        self.processQuit()


    def processEditProjCenterX(self):
        cp.confpars.projCenterX = int(self.editProjCenterX.displayText())        


    def processEditProjCenterY(self):
        cp.confpars.projCenterY = int(self.editProjCenterY.displayText())        

        
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


    def makeTabBarLayout(self,mode=None) :

        if mode != None : self.tabBar.close()
        self.tabBar = QtGui.QTabBar()

        self.indTabX   = self.tabBar.addTab( 'X' )
        self.indTabY   = self.tabBar.addTab( 'Y' )
        self.indTabR   = self.tabBar.addTab( 'R'   )
        self.indTabPhi = self.tabBar.addTab( 'Phi' )

        self.tabBar.setTabTextColor(self.indTabX,   QtGui.QColor('red'))
        self.tabBar.setTabTextColor(self.indTabY,   QtGui.QColor('magenta'))
        self.tabBar.setTabTextColor(self.indTabR,   QtGui.QColor('blue'))
        self.tabBar.setTabTextColor(self.indTabPhi, QtGui.QColor('green'))


        self.hboxT.addWidget(self.tabBar) 
        self.connect(self.tabBar, QtCore.SIGNAL('currentChanged(int)'), self.processTabBar)


    def processTabBar(self):
        indTab = self.tabBar.currentIndex()
        #print 'TabBar index=',indTab

        self.guiWin.close()

        if indTab == self.indTabX :
            self.guiWin = guiprojx.GUIWhatToDisplayForProjX()

        if indTab == self.indTabY :
            self.guiWin = guiprojy.GUIWhatToDisplayForProjY()

        if indTab == self.indTabR :
            self.guiWin = guiprojr.GUIWhatToDisplayForProjR()

        if indTab == self.indTabPhi :
            self.guiWin = guiprojphi.GUIWhatToDisplayForProjPhi()

        self.guiWin.setMinimumHeight(120)
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
    ex  = GUIWhatToDisplayForProjections()
    ex.show()
    app.exec_()

#-----------------------------
