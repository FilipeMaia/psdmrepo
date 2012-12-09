#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIIntensityMonitors...
#
#------------------------------------------------------------------------

"""GUI sets parameters for intensity monitors"""

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
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from ConfigParametersCorAna import confpars as cp
from Logger                 import logger
from FileNameManager        import fnm
from PlotArray              import *
import GlobalUtils          as     gu
#---------------------
#  Class definition --
#---------------------
class GUIIntensityMonitors ( QtGui.QWidget ) :
    """GUI sets parameters for intensity monitors"""

    def __init__ ( self, parent=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 600,300)
        self.setWindowTitle('GUI for Intensity Monitors')
        self.setFrame()

        self.list_of_dicts   = []

        self.grid = QtGui.QGridLayout()
        self.grid_row = 0
        self.setTitleBar()
        for i,name in enumerate(cp.imon_name_list) :
            print i, name.value()
            self.guiSection(name, cp.imon_ch1_list[i],
                                  cp.imon_ch2_list[i],
                                  cp.imon_ch3_list[i],
                                  cp.imon_ch4_list[i])

        self.setLayout(self.grid)

        self.showToolTips()
        self.setStyle()

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        msg = 'Use this GUI to set partitions.'
        self.setToolTip(msg)

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        self.frame.setVisible(False)

    def setStyle(self):
        self.setMinimumSize(600,300)
        #self.setMinimumWidth(380)
        #self.setMinimumHeight(300)
        self.setStyleSheet(cp.styleBkgd)


    def setTitleBar(self) :
        list_of_titles = ['Intensity Monitor', 'Ch.1', 'Ch.2', 'Ch.3', 'Ch.4', 'Plot']
        for i,t in enumerate(list_of_titles) : 
            label = QtGui.QLabel(t)
            label.setStyleSheet(cp.styleLabel)
            label.setFixedHeight(10)
            self.grid.addWidget(label, self.grid_row, i)
        self.grid_row += 1


    def guiSection(self, name, cbch1=None, cbch2=None, cbch3=None, cbch4=None) :
        edi      = QtGui.QLineEdit( str(name.value()) )        
        but      = QtGui.QPushButton('Browse')
        #box      = QtGui.QComboBox( self ) 
        #box.addItems(self.list_of_methods)
        #box.setCurrentIndex( self.list_of_methods.index(method.value()) )
        cb1 = QtGui.QCheckBox('   +', self)
        cb2 = QtGui.QCheckBox('   +', self)
        cb3 = QtGui.QCheckBox('   +', self)
        cb4 = QtGui.QCheckBox('   =', self)

        sec_dict = { 0:(edi,name),
                     1:(cb1,cbch1),
                     2:(cb2,cbch2),
                     3:(cb3,cbch3),
                     4:(cb4,cbch4),
                     5:(but,None) }

        self.list_of_dicts.append( sec_dict )

        for col,(fld, par) in sec_dict.iteritems() :
            self.grid.addWidget(fld, self.grid_row, col)
            if col>0 and col<5 :
                fld.setChecked( par.value() )
                self.connect(fld, QtCore.SIGNAL('stateChanged(int)'), self.onCBox )

        self.grid_row += 1
        
        edi.setReadOnly( True )  
        edi.setToolTip('Edit number in this field\nor click on "Browse"\nto select the file.')
        but.setToolTip('Click on this button\nand select the file.')
        #box.setToolTip('Click on this box\nand select the partitioning method.')

        edi    .setStyleSheet (cp.styleEditInfo) # cp.styleEditInfo
        #box    .setStyleSheet (cp.styleButton) 
        but    .setStyleSheet (cp.styleButton) 
        edi    .setAlignment (QtCore.Qt.AlignLeft)

        width = 60
        but    .setFixedWidth(width)
        edi    .setFixedWidth(250)
        #box    .setFixedWidth(160)

        #self.connect(edi, QtCore.SIGNAL('editingFinished()'),        self.onEdit )
        self.connect(but, QtCore.SIGNAL('clicked()'),                self.onBut  )
        #self.connect(box, QtCore.SIGNAL('currentIndexChanged(int)'), self.onBox  )


    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)
        #try    : del cp.guiintensitymonitors # GUIIntensityMonitors
        #except : pass # silently ignore

    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()

    def onApply(self):
        logger.debug('onApply - is already applied...', __name__)

    def onShow(self):
        logger.debug('onShow - is not implemented yet...', __name__)


    def onCBox(self) :
        for row0, sec_dict in enumerate(self.list_of_dicts) :                        
            for col,(cbx, par) in sec_dict.iteritems() :
                if cbx.hasFocus() : 
                    msg = 'onCBox - set status %s of checkbox in row:%s col:%s' % (cbx.isChecked(), row0+1, col)
                    par.setValue( cbx.isChecked() )
                    logger.info(msg, __name__ )

                    if cp.plotarray_is_on :
                        self.redrawArray(row0)
                    return


    def onBut(self):
        logger.debug('onBut', __name__)
        for row0, sec_dict in enumerate(self.list_of_dicts) :                        
            edi, name = sec_dict[0]
            but, empt = sec_dict[5]
            if but.hasFocus() : 
                msg = 'onBut - click on button %s in row %s, plot for %s' % (str(but.text()), row0+1, name.value())
                logger.info(msg, __name__ )
                self.plotIMon(row0)
                return


    def plotIMon(self,imon):
        logger.debug('plotIMon', __name__)
        arr = self.getArray(imon)
        try :
            cp.plotarray.close()
        except :
            if arr == None : return
            cp.plotarray = PlotArray(None, arr,
                                     ofname=fnm.path_data_mons_plot(),
                                     title=self.titleForIMon(imon))
            cp.plotarray.move(self.parentWidget().pos().__add__(QtCore.QPoint(700,300)))
            cp.plotarray.show()


    def titleForIMon(self,imon):
        return cp.imon_name_list[imon].value() + \
               ':  sum of channels: ' + \
               self.strMaskForIMonChannels(imon) 
        

    def redrawArray(self,imon):
        logger.debug('plotIMon', __name__)
        arr = self.getArray(imon)
        if arr == None : return
        try :
            cp.plotarray.set_array(arr, title=self.titleForIMon(imon))
        except :
            pass


    def boolMaskForIMonChannels(self,imon):
        return [cp.imon_ch1_list[imon].value(),
                cp.imon_ch2_list[imon].value(),
                cp.imon_ch3_list[imon].value(),
                cp.imon_ch4_list[imon].value()]


    def npIntMaskForIMonChannels(self,imon):
        return np.array(self.boolMaskForIMonChannels(imon),dtype=int)


    def strMaskForIMonChannels(self,imon):
        mask = self.boolMaskForIMonChannels(imon)
        str = ''
        for i,v in enumerate(mask) :
            if v : str += '%s+' % (i+1)
        return str.rstrip('+')


    def getArray(self,imon):
        logger.debug('getArray for imon: '+str(imon), __name__)
        arr_all = gu.get_array_from_file(fnm.path_data_scan_monitors_data())
        if arr_all == None : return None
        logger.debug('Array shape: ' + str(arr_all.shape), __name__)

        ibase    = 1+imon*4
        arr_imon = arr_all[:,ibase:ibase+4]
        #print 'arr_imon:\n', arr_imon
        #print 'arr_imon.shape:', arr_imon.shape

        #mask = self.maskForIMonChannels(imon)
        #npmask = np.array(mask,dtype=float)
        npmask = self.npIntMaskForIMonChannels(imon)

        size   = arr_imon.shape[0]
        npcol1 = np.ones(size)

        X,Y = np.meshgrid(npmask,npcol1)
        arr_prod = (arr_imon * X)        
        arr_sum  = arr_prod.sum(1) 
        
        #print 'npmask=', npmask
        #print 'size=', size
        #print X
        #print X.shape
        #print arr_imon
        #print arr_imon.shape
        #print arr_prod
        #print arr_prod.shape
        return arr_sum

#-----------------------------
#
#    def onEdit(self):
#        logger.debug('onEdit', __name__)
#        for fields in self.sect_fields :
#            edi = fields[4]
#            par = fields[7]
#            if edi.isModified() :            
#                edi.setModified(False)
#                par.setValue( str(edi.displayText()) )
#                logger.info('Set parameter = ' + str( par.value()), __name__ )
#
#    def onBox(self):
#        for fields in self.sect_fields :
#            box = fields[3]
#            if box.hasFocus() :
#                tit    = fields[0]
#                method = fields[6]
#                method_selected = box.currentText()
#                method.setValue( method_selected ) 
#                logger.info('onBox for ' + str(tit.text()) + ' - selected method: ' + method_selected, __name__)
#
#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIIntensityMonitors ()
    widget.show()
    app.exec_()

#-----------------------------
