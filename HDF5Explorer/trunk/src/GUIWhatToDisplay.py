#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIWhatToDisplay...
#
#------------------------------------------------------------------------

"""Generates GUI to select information for rendaring in the HDF5Explorer.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

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
from PyQt4 import QtGui, QtCore

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters                as cp
import GlobalMethods                   as gm
import GUIWhatToDisplayCBoxCSpad       as cboxCS
import GUIWhatToDisplayCBoxImage       as cboxIM
import GUIWhatToDisplayCBoxOther       as cboxOT
import GUIWhatToDisplayAlternative     as wtdAL
import GUIWhatToDisplayForImage        as wtdIM 
import GUIWhatToDisplayForCSpad        as wtdCS 
import GUIWhatToDisplayForWaveform     as wtdWF
import GUIWhatToDisplayForProjections  as wtdPR
import GUICorrelation                  as wtdCO
import GUIConfiguration                as guiconfig
import GUISelection                    as guisel

#---------------------
#  Class definition --
#---------------------
class GUIWhatToDisplay ( QtGui.QWidget ) :
    """Provides GUI to select information for rendering.

    Detailed description should be here...
    @see BaseClass
    @see OtherClass
    """

    #--------------------
    #  Class variables --
    #--------------------
    #publicStaticVariable = 0 
    #__privateStaticVariable = "A string"

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(370, 350, 500, 600)
        self.setWindowTitle('What and how to display')

        self.parent = parent

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        
        #self.butRefresh       = QtGui.QPushButton('Refresh')
        self.butClose         = QtGui.QPushButton('Quit')
        self.butSave          = QtGui.QPushButton('Save')
        #self.butSave.setMaximumWidth(40)   

        self.tabBar      = QtGui.QTabBar()
        self.tabBar.setShape(QtGui.QTabBar.RoundedNorth)
       #self.tabBar.setShape(QtGui.QTabBar.TriangularNorth)
        self.indTabCS    = self.tabBar.addTab('CSpad')
        self.indTabIM    = self.tabBar.addTab('Image')
        self.indTabWF    = self.tabBar.addTab('Waveform')
        self.indTabPR    = self.tabBar.addTab('Proj.')
        self.indTabCO    = self.tabBar.addTab('Corr.')
        self.indTabEmpty = self.tabBar.addTab(30*' ')
        self.tabBar.setTabEnabled(self.indTabEmpty,False)
        self.isFirstEntry = True

        self.tabBar.setTabTextColor(self.indTabCS,QtGui.QColor('red'))
        self.tabBar.setTabTextColor(self.indTabIM,QtGui.QColor('blue'))
        self.tabBar.setTabTextColor(self.indTabWF,QtGui.QColor('green'))
        self.tabBar.setTabTextColor(self.indTabPR,QtGui.QColor('magenta'))
        self.tabBar.setTabTextColor(self.indTabCO,QtGui.QColor('black'))
       #self.tabBar.setTabTextColor(self.indTabED,QtGui.QColor('white'))

        self.tabBarBot       = QtGui.QTabBar()
        self.tabBarBot.setShape(QtGui.QTabBar.RoundedSouth)
        #self.tabBarBot.setShape(QtGui.QTabBar.TriangularSouth)
        self.indTabBotConfig = self.tabBarBot.addTab('Configuration')
        self.indTabBotSelect = self.tabBarBot.addTab('Selection')
        self.indTabBotEmpty  = self.tabBarBot.addTab(64*' ')
        self.tabBarBot.setTabEnabled(self.indTabBotEmpty,False)
        
        self.hboxT = QtGui.QHBoxLayout()
        self.hboxT.addWidget(self.tabBar) 

        self.guiTab = wtdCS.GUIWhatToDisplayForCSpad()          
        self.guiTab.setMinimumHeight(240)

        self.hboxD = QtGui.QHBoxLayout()
        self.hboxD.addWidget(self.guiTab)
        
        self.hboxC = QtGui.QHBoxLayout()
        #self.hboxC.addWidget(self.butRefresh)
        self.hboxC.addWidget(self.butSave)
        self.hboxC.addStretch(2)
        self.hboxC.addWidget(self.butClose)

        self.hboxTB = QtGui.QHBoxLayout()
        self.hboxTB.addWidget(self.tabBarBot) 

        self.vboxB = QtGui.QVBoxLayout()
        self.isOpenCS = False
        self.isOpenIM = False
        self.isOpenOT = False
        
        self.makeVBoxB()

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.vboxB)
        self.vbox.addStretch(1)     
        self.vbox.addLayout(self.hboxT)
        self.vbox.addLayout(self.hboxD) 
        self.vbox.addLayout(self.hboxTB)
        self.vbox.addStretch(1)
        self.vbox.addLayout(self.hboxC)

        self.setLayout(self.vbox)
  
        #self.connect(self.butRefresh,       QtCore.SIGNAL('clicked()'),           self.processRefresh   )
        self.connect(self.butSave,          QtCore.SIGNAL('clicked()'),           self.processSave   )
        self.connect(self.butClose,         QtCore.SIGNAL('clicked()'),           self.processClose  )
        self.connect(self.tabBar,           QtCore.SIGNAL('currentChanged(int)'), self.processTabBar )
        self.connect(self.tabBarBot,        QtCore.SIGNAL('currentChanged(int)'), self.processTabBarBot )

        self.showToolTips()

        cp.confpars.wtdWindowIsOpen = True




    def makeVBoxB(self):

        cspadDatasetIsChecked = gm.CSpadDatasetIsChecked()
        imageDatasetIsChecked = gm.ImageDatasetIsChecked()
        wavefDatasetIsChecked = gm.WaveformDatasetIsChecked()
        correDatasetIsChecked = gm.CorrelationDatasetIsChecked()

        self.vertSize = 365 # accounts for the tab bars and buttons vertical size

        if self.isOpenCS: self.cboxguiCS.close()
        if self.isOpenIM: self.cboxguiIM.close()
        if self.isOpenOT: self.cboxguiOT.close()

        if cspadDatasetIsChecked :
            self.cboxguiCS = cboxCS.GUIWhatToDisplayCBoxCSpad(self)
            self.cboxguiCS.setFixedHeight(150)
            self.vertSize += 150
            self.vboxB.addWidget(self.cboxguiCS) 
            self.indTabOpen = self.indTabCS 
            self.isOpenCS = True

        if imageDatasetIsChecked :
            self.cboxguiIM = cboxIM.GUIWhatToDisplayCBoxImage(self)
            self.cboxguiIM.setFixedHeight(80)
            self.vertSize += 80
            self.vboxB.addWidget(self.cboxguiIM) 
            self.indTabOpen = self.indTabIM 
            self.isOpenIM = True

        if wavefDatasetIsChecked or correDatasetIsChecked :
            self.cboxguiOT = cboxOT.GUIWhatToDisplayCBoxOther(self)
            self.cboxguiOT.setFixedHeight(50)
            self.vertSize += 50
            self.vboxB.addWidget(self.cboxguiOT)
            if wavefDatasetIsChecked : self.indTabOpen = self.indTabWF 
            if correDatasetIsChecked : self.indTabOpen = self.indTabCO 
            self.isOpenOT = True

        #self.cboxguiAlternative = wtdAL.GUIWhatToDisplayAlternative(self, self.title)
        #self.vbox.addWidget(self.cboxguiAlternative)

        if  self.isFirstEntry:
            self.isFirstEntry = False
            self.tabBar.setCurrentIndex(self.indTabOpen)

        self.processTabBar()

        self.setFixedSize(500, self.vertSize)


    def showToolTips(self):
        self.butClose    .setToolTip('Close this window') 


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def closeEvent(self, event):
        print 'closeEvent for WTD GUI'
        if self.isOpenCS: self.cboxguiCS.close()
        if self.isOpenIM: self.cboxguiIM.close()
        if self.isOpenOT: self.cboxguiOT.close()
        self.guiTab.close()
        self.tabBar.close()
        cp.confpars.wtdWindowIsOpen = False


    def processClose(self):
        print 'Close button is clicked for WTD GUI'
        self.close()


    def processSelection(self):
        #print 'Selection'
        self.guiTab.close()
        self.guiTab = cp.confpars.guiselection = guisel.GUISelection() 
        self.guiTab.setMinimumHeight(240)
        self.hboxD.addWidget(self.guiTab)


    def processConfiguration(self):
        #print 'Configuration'
        self.guiTab.close()
        self.guiTab = cp.confpars.guiconfig = guiconfig.GUIConfiguration() 
        self.guiTab.setMinimumHeight(240)
        self.hboxD.addWidget(self.guiTab)


    def processSave(self):
        print 'Save'
        cp.confpars.writeParameters()


    def processRefresh(self):
        print 'Refresh WTD GUI'
        self.makeVBoxB()
    

    def processTabBarBot(self):
        indTab = self.tabBarBot.currentIndex()
        #print 'TabBarBot index=',indTab
        if indTab == self.indTabBotEmpty  : return
        if indTab == self.indTabBotSelect : self.processSelection()
        if indTab == self.indTabBotConfig : self.processConfiguration()

        self.tabBar.setCurrentIndex(self.indTabEmpty)


    def processTabBar(self):
        indTab = self.tabBar.currentIndex()
        self.indTabOpen = indTab
        #print 'TabBar index=',indTab

        if indTab == self.indTabEmpty: return

        self.guiTab.close()

        cspadDatasetIsChecked = gm.CSpadDatasetIsChecked()
        imageDatasetIsChecked = gm.ImageDatasetIsChecked()
        wavefDatasetIsChecked = gm.WaveformDatasetIsChecked()
        correDatasetIsChecked = gm.CorrelationDatasetIsChecked()

        if indTab == self.indTabCS:
            if cspadDatasetIsChecked: self.guiTab = wtdCS.GUIWhatToDisplayForCSpad()
            else :                    self.guiTab = wtdAL.GUIWhatToDisplayAlternative(None, 'CSpad')

        if indTab == self.indTabIM:
            if imageDatasetIsChecked: self.guiTab = wtdIM.GUIWhatToDisplayForImage()
            else :                    self.guiTab = wtdAL.GUIWhatToDisplayAlternative(None, 'Image')

        if indTab == self.indTabWF:
            if wavefDatasetIsChecked: self.guiTab = wtdWF.GUIWhatToDisplayForWaveform()
            else :                    self.guiTab = wtdAL.GUIWhatToDisplayAlternative(None, 'Waveforms')

        if indTab == self.indTabPR:
            if cspadDatasetIsChecked or imageDatasetIsChecked:
                                      self.guiTab = wtdPR.GUIWhatToDisplayForProjections()
            else :                    self.guiTab = wtdAL.GUIWhatToDisplayAlternative(None, 'CSpad or Image projections')

        if indTab == self.indTabCO:
            if correDatasetIsChecked: self.guiTab = wtdCO.GUICorrelation()
            else :                    self.guiTab = wtdAL.GUIWhatToDisplayAlternative(None, 'Correlations')

        #if indTab == self.indTabED:
        #    self.guiTab = QtGui.QLabel('Sorry, this GUI is not implemented yet.\n' + 
        #                               'Will select 1-D parameters for plot vs event number.')

        self.guiTab.setMinimumHeight(240)
        self.hboxD.addWidget(self.guiTab)
        
        self.tabBarBot.setCurrentIndex(self.indTabBotEmpty)


#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIWhatToDisplay()
    ex.show()
    app.exec_()
#-----------------------------

