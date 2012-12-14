#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIELogPostingFields...
#
#------------------------------------------------------------------------

"""GUI sets fields for ELog posting"""

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
import GlobalUtils          as     gu
from FileNameManager        import fnm
#from PlotImgSpe             import *
#from GUIFileBrowser         import *


#---------------------
#  Class definition --
#---------------------
class GUIELogPostingFields ( QtGui.QWidget ) :
    """GUI sets fields for ELog posting"""

    def __init__ ( self, parent=None, att_fname=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 530, 30)
        self.setWindowTitle('Fields for ELog posting')
        self.setFrame()

        self.att_input = str(att_fname)

        #self.cbx_use = QtGui.QCheckBox('Use default fields for ELog posting', self)
        #self.cbx_use.setChecked( cp.elog_post_cbx_state.value() )

        self.list_of_rad_types = ['Data', 'Dark', 'Saved', 'Default', 'Edit']
        self.rad_grp = QtGui.QButtonGroup()
        self.list_of_rad = []

        for type in self.list_of_rad_types :
            rad = QtGui.QRadioButton(type)
            self.list_of_rad.append(rad)
            self.rad_grp.addButton(rad)        
            self.connect(rad, QtCore.SIGNAL('clicked()'), self.onRadio ) 

        self.lab_ins = QtGui.QLabel('Ins:')
        self.lab_exp = QtGui.QLabel('Exp:')
        self.lab_run = QtGui.QLabel('Run:')
        self.lab_tag = QtGui.QLabel('Tag:')
        self.lab_res = QtGui.QLabel('Rsp:')
        self.lab_msg = QtGui.QLabel('Msg:')
        self.lab_att = QtGui.QLabel('Att:')

        self.edi_ins = QtGui.QLineEdit( cp.elog_post_ins.value() ) 
        self.edi_exp = QtGui.QLineEdit( cp.elog_post_exp.value() )
        self.edi_run = QtGui.QLineEdit( cp.elog_post_run.value() )
        self.edi_tag = QtGui.QLineEdit( cp.elog_post_tag.value() )
        self.edi_res = QtGui.QLineEdit( cp.elog_post_res.value() )
        self.edi_msg = QtGui.QLineEdit( cp.elog_post_msg.value() )
        self.edi_att = QtGui.QLineEdit( cp.elog_post_att.value() )

        self.setFieldsDefault()

        self.list_of_fields = {
            (self.lab_ins, self.edi_ins, cp.elog_post_ins, self.ins), 
            (self.lab_exp, self.edi_exp, cp.elog_post_exp, self.exp),
            (self.lab_run, self.edi_run, cp.elog_post_run, self.run), 
            (self.lab_tag, self.edi_tag, cp.elog_post_tag, self.tag),
            (self.lab_res, self.edi_res, cp.elog_post_res, self.res),
            (self.lab_msg, self.edi_msg, cp.elog_post_msg, self.msg),
            (self.lab_att, self.edi_att, cp.elog_post_att, self.att) }
                             
        for (label, edi, par, val_loc) in self.list_of_fields :
           self.connect(edi, QtCore.SIGNAL('editingFinished ()'), self.onEdit) 

        #self.edi_path = QtGui.QLineEdit( fnm.path_blam() )        
        #self.edi_path.setReadOnly( True )   
        #self.but_path = QtGui.QPushButton('File:')
        #self.but_plot = QtGui.QPushButton('Plot')
        #self.but_brow = QtGui.QPushButton('Browse')

        self.grid = QtGui.QGridLayout()
        self.grid_row = 0
        #self.grid.addWidget(self.tit_path, self.grid_row,   0)
        #self.grid.addWidget(self.cbx_use, self.grid_row,   0, 1, 8)

        for col,rad in enumerate(self.list_of_rad) :
            self.grid.addWidget(rad, self.grid_row, col*2, 1, 2)

        self.grid.addWidget(self.lab_ins, self.grid_row+1, 0, 1, 2)
        self.grid.addWidget(self.edi_ins, self.grid_row+2, 0, 1, 2)
        self.grid.addWidget(self.lab_exp, self.grid_row+1, 2, 1, 2)
        self.grid.addWidget(self.edi_exp, self.grid_row+2, 2, 1, 2)
        self.grid.addWidget(self.lab_run, self.grid_row+1, 4, 1, 2)
        self.grid.addWidget(self.edi_run, self.grid_row+2, 4, 1, 2)
        self.grid.addWidget(self.lab_tag, self.grid_row+1, 6, 1, 2)
        self.grid.addWidget(self.edi_tag, self.grid_row+2, 6, 1, 2)
        self.grid.addWidget(self.lab_res, self.grid_row+1, 8, 1, 2)
        self.grid.addWidget(self.edi_res, self.grid_row+2, 8, 1, 2)
        self.grid.addWidget(self.lab_att, self.grid_row+3, 0)
        self.grid.addWidget(self.edi_att, self.grid_row+3, 1, 1, 7)
        self.grid.addWidget(self.lab_msg, self.grid_row+4, 0)
        self.grid.addWidget(self.edi_msg, self.grid_row+4, 1, 1, 7)

        #self.grid.addWidget(self.but_path, self.grid_row+1, 0)
        #self.grid.addWidget(self.edi_path, self.grid_row+1, 1)
        #self.grid.addWidget(self.but_plot, self.grid_row+2, 0)
        #self.grid.addWidget(self.but_brow, self.grid_row+2, 1)

        #self.connect(self.cbx_use,  QtCore.SIGNAL('stateChanged(int)'), self.onCBox ) 
        #self.connect(self.but_path, QtCore.SIGNAL('clicked()'), self.on_but_path )
        #self.connect(self.but_plot, QtCore.SIGNAL('clicked()'), self.on_but_plot )

        self.setLayout(self.grid)

        self.showToolTips()

        self.setStyle()
        self.setCheckedRadioButton()
        self.setFields()


    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        #self        .setToolTip('Use this GUI to work with xtc file.')
        #self.edi_path.setToolTip('The path to the blamish mask file')
        #self.but_path.setToolTip('Push this button and select the blamish mask file')
        #self.but_plot.setToolTip('Plot image and spectrum for blamish file')
        #self.but_brow.setToolTip('Browse blamish file')
        #self.cbx_use .setToolTip('Check box \nto set and use \nblamish mask correction')
        self.edi_ins.setToolTip('Instrument')
        self.edi_exp.setToolTip('Experiment')
        self.edi_run.setToolTip('Run number')
        self.edi_tag.setToolTip('Tag of the message.\nCan be used for \nfast search in ELog.')
        self.edi_res.setToolTip('Responce on message number.\nCan be used for accumulation of \nmessages in one branch of ELog.')
        self.edi_msg.setToolTip('Message for posting in ELog')
        self.edi_att.setToolTip('Attached file')
        #self.rad_grp.setToolTip('Fast substitution of \nparameters for posting in ELog.')
        for rad in self.list_of_rad :
            rad.setToolTip('Fast substitution of \nparameters for posting \nin ELog using ' + str(rad.text()))

        
    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):

        self.setMinimumWidth(400)
        self.setStyleSheet(cp.styleBkgd)

        width_edi       =  80
        width_edi_long  = 420
        width_com       =  80
        width_com_short =  34

        #self.edi_path.setStyleSheet (cp.styleEditInfo)
        #self.edi_path.setAlignment  (QtCore.Qt.AlignRight)

        #self.but_path.setStyleSheet (cp.styleButton)
        #self.but_plot.setStyleSheet (cp.styleButton) 
        #self.but_brow.setStyleSheet (cp.styleButton) 
   
        #self.but_path.setFixedWidth(width)
        #self.but_plot.setFixedWidth(width)
        #self.but_brow.setFixedWidth(width)
        #self.cbx_use .setStyleSheet (cp.styleLabel)

        for (label, edi, par, val_loc) in self.list_of_fields :
            label.setStyleSheet(cp.styleLabel)
            label.setFixedWidth(width_com)
            label.setAlignment(QtCore.Qt.AlignLeft)

        self.lab_att.setFixedWidth(width_com_short)
        self.lab_msg.setFixedWidth(width_com_short)

#        .setAlignment(QtCore.Qt.AlignRight)

        for (label, edi, par, val_loc) in self.list_of_fields :
            edi.setFixedWidth(width_edi)

        self.edi_msg.setMinimumWidth(width_edi_long)
        self.edi_att.setMinimumWidth(width_edi_long)



    def setCheckedRadioButton(self):
        for rad in self.list_of_rad :
            if cp.elog_post_rad.value() == str(rad.text()) : rad.setChecked(True)


    def onRadio(self) :
        self.setFields()


    def setFields(self):
        self.is_read_only = True
        for rad in self.list_of_rad :
            if rad.isChecked() :
                rad_txt = str(rad.text()) 
                print 'Set fields for ' + rad_txt
                if   rad_txt == 'Data'    : self.setFieldsData() 
                elif rad_txt == 'Dark'    : self.setFieldsDark()
                elif rad_txt == 'Saved'   : self.setFieldsSaved()
                elif rad_txt == 'Edit'    : self.setFieldsEdit()
                elif rad_txt == 'Default' : self.setFieldsDefault()
                else                      : self.setFieldsDefault()
                
                self.updateFields()
                self.setFieldsStyle()


    def setFieldsStyle(self):
        for (label, edi, par, val_loc)  in self.list_of_fields :
            if edi == self.edi_tag or edi == self.edi_msg or edi == self.edi_res :
                edi.setReadOnly(False)
                edi.setStyleSheet (cp.styleEdit)
            else :
                edi.setReadOnly( self.is_read_only )
                if self.is_read_only : edi.setStyleSheet (cp.styleEditInfo)
                else                 : edi.setStyleSheet (cp.styleEdit)

        self.edi_res.setReadOnly(True)
        self.edi_res.setStyleSheet(cp.styleEditInfo)


    def setFieldsData(self):
        self.ins, self.exp, run_str, run_num = gu.parse_xtc_path(fnm.path_data_xtc())
        self.run = str(run_num)
        #self.att = fnm.path_data_time_plot(self)
        self.tag = cp.elog_post_tag.value()
        self.msg = cp.elog_post_msg.value()
        self.att = self.att_input


    def setFieldsDark(self):
        self.ins, self.exp, run_str, run_num = gu.parse_xtc_path(fnm.path_dark_xtc())
        self.run = str(run_num)
        #self.att = fnm.path_data_time_plot(self)
        self.tag = cp.elog_post_tag.value()
        self.msg = cp.elog_post_msg.value()
        self.att = self.att_input


    def setFieldsEdit(self):
        self.is_read_only = False
        #self.ins = cp.elog_post_ins.value() 
        #self.exp = cp.elog_post_exp.value()
        #self.run = cp.elog_post_run.value()
        #self.tag = cp.elog_post_tag.value()
        #self.res = cp.elog_post_res.value()
        #self.msg = cp.elog_post_msg.value()
        #self.att = cp.elog_post_att.value()
        self.att = self.att_input


    def setFieldsSaved(self):
        self.ins = cp.elog_post_ins.value() 
        self.exp = cp.elog_post_exp.value()
        self.run = cp.elog_post_run.value()
        self.tag = cp.elog_post_tag.value()
        self.res = cp.elog_post_res.value()
        self.msg = cp.elog_post_msg.value()
        #self.att = cp.elog_post_att.value()
        self.att = self.att_input


    def setFieldsDefault(self):
        self.ins = cp.elog_post_ins.value_def() 
        self.exp = cp.elog_post_exp.value_def()
        self.run = cp.elog_post_run.value_def()
        self.tag = cp.elog_post_tag.value_def()
        self.res = cp.elog_post_res.value_def()
        self.msg = cp.elog_post_msg.value_def()
        #self.att = cp.elog_post_att.value_def()
        self.att = self.att_input


    def updateFields(self):
        self.edi_ins.setText( self.ins ) 
        self.edi_exp.setText( self.exp )
        self.edi_run.setText( self.run )
        self.edi_tag.setText( self.tag )
        self.edi_res.setText( self.res )
        self.edi_msg.setText( self.msg )
        self.edi_att.setText( self.att )


    def updateConfigPars(self):
        cp.elog_post_ins.setValue(self.ins) 
        cp.elog_post_exp.setValue(self.exp)
        cp.elog_post_run.setValue(self.run)
        cp.elog_post_tag.setValue(self.tag)
        cp.elog_post_res.setValue(self.res)
        cp.elog_post_msg.setValue(self.msg)
        cp.elog_post_att.setValue(self.att)

    
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

        #try    : cp.plotimgspe.close()
        #except : pass

        #try    : cp.guifilebrowser.close()
        #except : pass
            
        #try    : del cp.guielogpostingfields # GUIELogPostingFields
        #except : pass # silently ignore


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def onEdit(self):
        logger.debug('onEdit', __name__)
        for (label, edi, par, val_loc)  in self.list_of_fields :
            if  edi.isModified() :            
                edi.setModified(False)
                val_loc = str(edi.displayText())
                msg = 'Set the local value of ' + str(label.text()) + ' ' + val_loc
                logger.info(msg, __name__ )
                print msg


    def onCBox(self):
        #if self.cbx_use .hasFocus() :
        par = cp.elog_post_cbx_state
        par.setValue( self.cbx_use.isChecked() )
        msg = 'onCBox - set status of ' + str(par.name()) + ': ' + str(par.value())
        logger.info(msg, __name__ )
        self.setFieldsStyle()


    def getListOfFields(self):
        return self.list_of_fields

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIELogPostingFields ()
    widget.show()
    app.exec_()

#-----------------------------
