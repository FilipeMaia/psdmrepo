#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUISettingsForWaveformWindow...
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
import h5py

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp
import PrintHDF5        as printh5

#---------------------
#  Class definition --
#---------------------
class GUISettingsForWaveformWindow ( QtGui.QWidget ) :
    """Provides GUI to select information for rendering."""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, window=0):
        QtGui.QWidget.__init__(self, parent)

        #print 'GUISettingsForWaveformWindow for window ', window

        self.window = window

        self.setGeometry(370, 350, 500, 150)
        self.setWindowTitle('Adjust Waveform Parameters')

        self.palette = QtGui.QPalette()

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

        titFont12 = QtGui.QFont("Sans Serif", 12, QtGui.QFont.Bold)
        titFont10 = QtGui.QFont("Sans Serif", 10, QtGui.QFont.Bold)

        self.titWFDataSet        = QtGui.QLabel('Dataset:')
        self.titWFIndexes        = QtGui.QLabel('WF:    Black')

        #self.titWFWaveformAminmax= QtGui.QLabel('Alims:')
        #self.titWFWaveformTminmax= QtGui.QLabel('Tlims:')

        self.cboxALimits   = QtGui.QCheckBox('A min, max:',self)
        self.cboxTLimits   = QtGui.QCheckBox('T min, max:',self)
        self.cboxAUnits    = QtGui.QCheckBox('Use A units (V)',self)
        self.cboxTUnits    = QtGui.QCheckBox('Use T units (ns)',self)

        if cp.confpars.waveformWindowParameters[self.window][1]&1 : self.cboxALimits.setCheckState(2)
        if cp.confpars.waveformWindowParameters[self.window][1]&2 : self.cboxTLimits.setCheckState(2)
        if cp.confpars.waveformWindowParameters[self.window][1]&4 : self.cboxAUnits .setCheckState(2)
        if cp.confpars.waveformWindowParameters[self.window][1]&8 : self.cboxTUnits .setCheckState(2)

        self.char_expand = u'\u25BE' # down-head triangle
        height = 22
        width  = 60

        self.butWFDataSet = QtGui.QPushButton(cp.confpars.waveformWindowParameters[self.window][0])
        self.butWFDataSet.setMaximumWidth(350)
        self.setButWFDataSetTextAlignment()

        self.popupMenuForDataSet = QtGui.QMenu()
        self.fillPopupMenuForDataSet()

        self.popupMenuForWaveNumber = QtGui.QMenu()
        self.fillPopupMenuForWaveNumber()

        self.editWFWaveformAmin  = QtGui.QLineEdit(str(cp.confpars.waveformWindowParameters[self.window][2]))
        self.editWFWaveformAmax  = QtGui.QLineEdit(str(cp.confpars.waveformWindowParameters[self.window][3]))
        self.editWFWaveformAmin  .setMaximumWidth(width)
        self.editWFWaveformAmax  .setMaximumWidth(width)

        self.editWFWaveformTmin  = QtGui.QLineEdit(str(cp.confpars.waveformWindowParameters[self.window][4]))
        self.editWFWaveformTmax  = QtGui.QLineEdit(str(cp.confpars.waveformWindowParameters[self.window][5]))
        self.editWFWaveformTmin  .setMaximumWidth(width)
        self.editWFWaveformTmax  .setMaximumWidth(width)

        #self.titWFInd1  = QtGui.QLabel('Black')
        self.titWFInd2  = QtGui.QLabel('  Red')
        self.titWFInd3  = QtGui.QLabel('Green')
        self.titWFInd4  = QtGui.QLabel(' Blue')

        self.butWFInd1  = QtGui.QPushButton(str(cp.confpars.waveformWindowParameters[self.window][7]))
        self.butWFInd2  = QtGui.QPushButton(str(cp.confpars.waveformWindowParameters[self.window][8]))
        self.butWFInd3  = QtGui.QPushButton(str(cp.confpars.waveformWindowParameters[self.window][9]))
        self.butWFInd4  = QtGui.QPushButton(str(cp.confpars.waveformWindowParameters[self.window][10]))

        self.butWFInd1.setMaximumHeight(height)
        self.butWFInd2.setMaximumHeight(height)
        self.butWFInd3.setMaximumHeight(height)
        self.butWFInd4.setMaximumHeight(height)

        self.butWFInd1.setMaximumWidth(width)
        self.butWFInd2.setMaximumWidth(width)
        self.butWFInd3.setMaximumWidth(width)
        self.butWFInd4.setMaximumWidth(width)

        self.editWFWaveformAmin.setMaximumHeight(height)
        self.editWFWaveformAmax.setMaximumHeight(height)
        self.editWFWaveformTmin.setMaximumHeight(height)
        self.editWFWaveformTmax.setMaximumHeight(height)

        #self.editWFWaveformAmin.setValidator(QtGui.QDoubleValidator(-1000000,1000000,self))
        #self.editWFWaveformAmax.setValidator(QtGui.QDoubleValidator(-1000000,1000000,self))
        #self.editWFWaveformTmin.setValidator(QtGui.QDoubleValidator(0,100000,self))
        #self.editWFWaveformTmax.setValidator(QtGui.QDoubleValidator(0,100000,self))

        self.setAEditFieldsStatus()
        self.setTEditFieldsStatus()

        #self.radioAuto   = QtGui.QRadioButton("Auto range control")
        #self.radioManual = QtGui.QRadioButton("Manual")
        #self.radioGroup  = QtGui.QButtonGroup()
        #self.radioGroup.addButton(self.radioAuto)
        #self.radioGroup.addButton(self.radioManual)
        #if cp.confpars.waveformWindowParameters[self.window][1] : self.radioAuto  .setChecked(True)
        #else :                                                    self.radioManual.setChecked(True)


        gridWF = QtGui.QGridLayout()
        gridWF.addWidget(self.titWFDataSet,        0, 0)
        gridWF.addWidget(self.butWFDataSet,        0, 1, 1, 7)
        gridWF.addWidget(self.titWFIndexes,        1, 0)

        gridWF.addWidget(self.cboxAUnits,          2, 0, 1, 2)
        gridWF.addWidget(self.cboxALimits,         2, 2, 1, 2)
        gridWF.addWidget(self.editWFWaveformAmin,  2, 4, 1, 2)
        gridWF.addWidget(self.editWFWaveformAmax,  2, 6, 1, 2)

        gridWF.addWidget(self.cboxTUnits,          3, 0, 1, 2)
        gridWF.addWidget(self.cboxTLimits,         3, 2, 1, 2)
        gridWF.addWidget(self.editWFWaveformTmin,  3, 4, 1, 2)
        gridWF.addWidget(self.editWFWaveformTmax,  3, 6, 1, 2)

        gridWF.addWidget(self.butWFInd1,           1, 1)
        gridWF.addWidget(self.titWFInd2,           1, 2)
        gridWF.addWidget(self.butWFInd2,           1, 3)
        gridWF.addWidget(self.titWFInd3,           1, 4)
        gridWF.addWidget(self.butWFInd3,           1, 5)
        gridWF.addWidget(self.titWFInd4,           1, 6)
        gridWF.addWidget(self.butWFInd4,           1, 7)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(gridWF) 
        self.vbox.addStretch(1)     

        if parent == None :
            self.setLayout(self.vbox)
            self.show()

        #self.connect(self.butClose,            QtCore.SIGNAL('clicked()'),         self.processClose )
        #self.connect(self.cboxWFImage,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxWFImage)
        #self.connect(self.cboxWFSpectrum,      QtCore.SIGNAL('stateChanged(int)'), self.processCBoxWFSpectrum)
        #self.connect(self.radioAuto,           QtCore.SIGNAL('clicked()'), self.processRadioAuto    )
        #self.connect(self.radioManual,         QtCore.SIGNAL('clicked()'), self.processRadioManual  )

        self.connect(self.editWFWaveformAmin,  QtCore.SIGNAL('editingFinished ()'), self.processEditWFWaveformAmin )
        self.connect(self.editWFWaveformAmax,  QtCore.SIGNAL('editingFinished ()'), self.processEditWFWaveformAmax )
        self.connect(self.editWFWaveformTmin,  QtCore.SIGNAL('editingFinished ()'), self.processEditWFWaveformTmin )
        self.connect(self.editWFWaveformTmax,  QtCore.SIGNAL('editingFinished ()'), self.processEditWFWaveformTmax )

        self.connect(self.butWFInd1,           QtCore.SIGNAL('clicked()'), self.processMenuWFInd1 )
        self.connect(self.butWFInd2,           QtCore.SIGNAL('clicked()'), self.processMenuWFInd2 )
        self.connect(self.butWFInd3,           QtCore.SIGNAL('clicked()'), self.processMenuWFInd3 )
        self.connect(self.butWFInd4,           QtCore.SIGNAL('clicked()'), self.processMenuWFInd4 )

        self.connect(self.cboxALimits,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxALimits)
        self.connect(self.cboxTLimits,         QtCore.SIGNAL('stateChanged(int)'), self.processCBoxTLimits)
        self.connect(self.cboxAUnits,          QtCore.SIGNAL('stateChanged(int)'), self.processCBoxAUnits)
        self.connect(self.cboxTUnits,          QtCore.SIGNAL('stateChanged(int)'), self.processCBoxTUnits)
        
        self.connect(self.butWFDataSet,        QtCore.SIGNAL('clicked()'), self.processMenuForDataSet )
 
        cp.confpars.wtdWFWindowIsOpen = True

        self.showToolTips()

    #-------------------
    # Private methods --
    #-------------------

    def getBitStatus(self,bit):
        return cp.confpars.waveformWindowParameters[self.window][1] & bit


    def setBitStatus(self,bit,isOn=True):
        if isOn:
            cp.confpars.waveformWindowParameters[self.window][1] |= bit # set bit
        else:
            cp.confpars.waveformWindowParameters[self.window][1] ^= bit # clear bit
        print 'rangeUnitBits =', cp.confpars.waveformWindowParameters[self.window][1]


    def processCBoxALimits(self):
        self.setBitStatus( 1, self.cboxALimits.isChecked() )
        self.setAEditFieldsStatus()


    def processCBoxTLimits(self):
        self.setBitStatus( 2, self.cboxTLimits.isChecked() )
        self.setTEditFieldsStatus()

    
    def processCBoxAUnits(self):
        self.setBitStatus( 4, self.cboxAUnits.isChecked() )
        self.cboxALimits.setCheckState(0)


    def processCBoxTUnits(self):
        self.setBitStatus( 8, self.cboxTUnits.isChecked() )
        self.cboxTLimits.setCheckState(0)


    def showToolTips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters for waveforms.')
        #self.radioAuto  .setToolTip('Select between Auto and Manual range control.')
        #self.radioManual.setToolTip('Select between Auto and Manual range control.')
        self.editWFWaveformAmin.setToolTip('This field can be edited for Manual control only.')
        self.editWFWaveformAmax.setToolTip('This field can be edited for Manual control only.')
        self.editWFWaveformTmin.setToolTip('This field can be edited for Manual control only.')
        self.editWFWaveformTmax.setToolTip('This field can be edited for Manual control only.')


    def fillPopupMenuForWaveNumber(self):
        #print 'fillPopupMenuForWaveNumber'

        self.popupMenuForWaveNumber.close()
        self.popupMenuForWaveNumber = QtGui.QMenu()
        self.popupMenuForWaveNumber.addAction('None')
        for wave_number in range(cp.confpars.waveformWindowParameters[self.window][6]) :

            self.popupMenuForWaveNumber.addAction(str(wave_number))


    def fillPopupMenuForDataSet(self):
        #print 'fillPopupMenuForDataSet'
        self.popupMenuForDataSet.addAction('None')
        for dsname in cp.confpars.list_of_checked_item_names :
            item_last_name = printh5.get_item_last_name(dsname)           
            if item_last_name == 'waveforms' :

                self.popupMenuForDataSet.addAction(dsname)

        

    def processMenuForDataSet(self):
        print 'MenuForDataSet'
        actionSelected = self.popupMenuForDataSet.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_ds = actionSelected.text()
        self.butWFDataSet.setText( selected_ds )
        self.setButWFDataSetTextAlignment()
        cp.confpars.waveformWindowParameters[self.window][0] = str(selected_ds)

        self.getNumberOfWavesInDataSet()
        self.fillPopupMenuForWaveNumber()
        self.initWFIndexes()
        

    def initWFIndexes(self) :

        #numberOfWavesInDataSet = cp.confpars.waveformWindowParameters[self.window][6]

        self.butWFInd1.setText('None')
        self.butWFInd2.setText('None')
        self.butWFInd3.setText('None')
        self.butWFInd4.setText('None')

        cp.confpars.waveformWindowParameters[self.window][7]  = None
        cp.confpars.waveformWindowParameters[self.window][8]  = None
        cp.confpars.waveformWindowParameters[self.window][9]  = None
        cp.confpars.waveformWindowParameters[self.window][10] = None

        self.setWFButtonColors()


    def setWFButtonColors(self) :
        
        if  self.butWFDataSet.text() == 'None' :
            self.butWFDataSet.setStyleSheet("background-color: rgb(255, 200, 200); color: rgb(0, 0, 0)")
        else :
            self.butWFDataSet.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0)")
        
        if  self.butWFInd1.text() == 'None' :
            self.butWFInd1.setStyleSheet("background-color: rgb(255, 200, 200); color: rgb(0, 0, 0)")
        else :
            self.butWFInd1.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0)")

        if  self.butWFInd2.text() == 'None' :
            self.butWFInd2.setStyleSheet("background-color: rgb(255, 200, 200); color: rgb(0, 0, 0)")
        else :
            self.butWFInd2.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0)")

        if  self.butWFInd3.text() == 'None' :
            self.butWFInd3.setStyleSheet("background-color: rgb(255, 200, 200); color: rgb(0, 0, 0)")
        else :
            self.butWFInd3.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0)")

        if  self.butWFInd4.text() == 'None' :
            self.butWFInd4.setStyleSheet("background-color: rgb(255, 200, 200); color: rgb(0, 0, 0)")
        else :
            self.butWFInd4.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0)")



    def setButWFDataSetTextAlignment(self):
        if self.butWFDataSet.text() == 'None' :
            self.butWFDataSet.setStyleSheet('Text-align:center')
        else :
            self.butWFDataSet.setStyleSheet('Text-align:right')


    def setAEditFieldsStatus(self):

        isChecked  = cp.confpars.waveformWindowParameters[self.window][1]&1
        isReadOnly = not isChecked

        if isReadOnly : self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('grey'))
        else :          self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('white'))

        self.editWFWaveformAmin.setPalette(self.palette)
        self.editWFWaveformAmax.setPalette(self.palette)
        self.editWFWaveformAmin.setReadOnly(isReadOnly)
        self.editWFWaveformAmax.setReadOnly(isReadOnly)


    def setTEditFieldsStatus(self):

        isChecked  = cp.confpars.waveformWindowParameters[self.window][1]&2
        isReadOnly = not isChecked

        if isReadOnly : self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('grey'))
        else :          self.palette.setColor(QtGui.QPalette.Base,QtGui.QColor('white'))
        
        self.editWFWaveformTmin.setPalette(self.palette)
        self.editWFWaveformTmax.setPalette(self.palette)
        self.editWFWaveformTmin.setReadOnly(isReadOnly)
        self.editWFWaveformTmax.setReadOnly(isReadOnly)


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def getVBoxForLayout(self):
        return self.vbox


    def setParentWidget(self,parent):
        self.parentWidget = parent


    def closeEvent(self, event):
        #print 'closeEvent'
        cp.confpars.wtdWFWindowIsOpen = False


    def processClose(self):
        #print 'Close button'
        self.close()


    def processEditWFWaveformAmin(self):
        cp.confpars.waveformWindowParameters[self.window][2] = float(self.editWFWaveformAmin.displayText())        

    def processEditWFWaveformAmax(self):
        cp.confpars.waveformWindowParameters[self.window][3] = float(self.editWFWaveformAmax.displayText())        

    def processEditWFWaveformTmin(self):
        cp.confpars.waveformWindowParameters[self.window][4] = float(self.editWFWaveformTmin.displayText())        

    def processEditWFWaveformTmax(self):
        cp.confpars.waveformWindowParameters[self.window][5] = float(self.editWFWaveformTmax.displayText())        

    def processMenuWFInd1(self):
        actionSelected = self.popupMenuForWaveNumber.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_wfnum = actionSelected.text()
        if selected_wfnum == 'None' :  wfnum = None
        else :                         wfnum = int(selected_wfnum) 
        self.butWFInd1.setText( selected_wfnum )
        cp.confpars.waveformWindowParameters[self.window][7] = wfnum
        self.setWFButtonColors()


    def processMenuWFInd2(self):
        actionSelected = self.popupMenuForWaveNumber.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_wfnum = actionSelected.text()
        if selected_wfnum == 'None' :  wfnum = None
        else :                         wfnum = int(selected_wfnum) 
        self.butWFInd2.setText( selected_wfnum )
        cp.confpars.waveformWindowParameters[self.window][8] = wfnum       
        self.setWFButtonColors()


    def processMenuWFInd3(self):
        actionSelected = self.popupMenuForWaveNumber.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_wfnum = actionSelected.text()
        if selected_wfnum == 'None' :  wfnum = None
        else :                         wfnum = int(selected_wfnum) 
        self.butWFInd3.setText( selected_wfnum )
        cp.confpars.waveformWindowParameters[self.window][9] = wfnum       
        self.setWFButtonColors()


    def processMenuWFInd4(self):
        actionSelected = self.popupMenuForWaveNumber.exec_(QtGui.QCursor.pos())
        if actionSelected==None : return
        selected_wfnum = actionSelected.text()
        if selected_wfnum == 'None' :  wfnum = None
        else :                         wfnum = int(selected_wfnum) 
        self.butWFInd4.setText( selected_wfnum )
        cp.confpars.waveformWindowParameters[self.window][10] = wfnum       
        self.setWFButtonColors()


    def processRadioAuto(self):
        #print 'RadioAuto'
        cp.confpars.waveformWindowParameters[self.window][1] = True
        self.setEditFieldsReadOnly(cp.confpars.waveformWindowParameters[self.window][1])
                      

    def processRadioManual(self):
        #print 'RadioManual'
        cp.confpars.waveformWindowParameters[self.window][1] = False
        self.setEditFieldsReadOnly(cp.confpars.waveformWindowParameters[self.window][1])

    def getNumberOfWavesInDataSet(self):

        dsname = cp.confpars.waveformWindowParameters[self.window][0]
        if dsname == 'None' :
            self.numberOfWavesInDataSet = 0
            #return self.numberOfWavesInDataSet
        else :
            fname = cp.confpars.dirName+'/'+cp.confpars.fileName
            print 'Open file : %s' % (fname)
            f  = h5py.File(fname, 'r') # open read-only
            ds = f[dsname]
            ds1ev = ds[cp.confpars.eventCurrent]

            self.numberOfWavesInDataSet, par2, dimX = ds1ev.shape
            print 'numberOfWavesInDataSet, par2, dimX = ', self.numberOfWavesInDataSet, par2, dimX

            f.close()

        cp.confpars.waveformWindowParameters[self.window][6] = self.numberOfWavesInDataSet
        #return self.numberOfWavesInDataSet


#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUISettingsForWaveformWindow()
    ex.show()
    app.exec_()
#-----------------------------

