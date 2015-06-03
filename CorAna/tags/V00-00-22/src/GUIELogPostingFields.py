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
__version__ = "$Revision$"
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
class LocalParameter () :
    _val=None

    def __init__ ( self, val=None ) :
        self._val = val

    def setValue ( self, val ) :    
        self._val = val

    def getValue (self) :    
        return self._val

    def value (self) :    
        return self._val

#---------------------

class GUIELogPostingFields ( QtGui.QWidget ) :
    """GUI sets fields for ELog posting"""

    ins = LocalParameter ()
    exp = LocalParameter ()
    run = LocalParameter ()
    tag = LocalParameter ()
    res = LocalParameter ()
    msg = LocalParameter ()
    att = LocalParameter ()

    def __init__ ( self, parent=None, att_fname=None ) :
        QtGui.QWidget.__init__(self, parent)
        self.setGeometry(200, 400, 530, 30)
        self.setWindowTitle('Fields for ELog posting')
        self.setFrame()

        self.parent    = parent
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

        self.edi_res.setValidator(QtGui.QIntValidator(0,9000000,self))

        self.setFieldsSaved()

        self.list_of_fields = [
            [self.lab_ins, self.edi_ins, cp.elog_post_ins, self.ins], 
            [self.lab_exp, self.edi_exp, cp.elog_post_exp, self.exp],
            [self.lab_run, self.edi_run, cp.elog_post_run, self.run], 
            [self.lab_tag, self.edi_tag, cp.elog_post_tag, self.tag],
            [self.lab_res, self.edi_res, cp.elog_post_res, self.res],
            [self.lab_msg, self.edi_msg, cp.elog_post_msg, self.msg],
            [self.lab_att, self.edi_att, cp.elog_post_att, self.att] ]

        for [label, edi, par, loc_par] in self.list_of_fields :
           self.connect(edi, QtCore.SIGNAL('editingFinished ()'), self.onEdit) 

        self.grid = QtGui.QGridLayout()
        self.grid_row = 0
        for col,rad in enumerate(self.list_of_rad) :
            self.grid.addWidget(rad, self.grid_row, col*2, 1, 2)

        self.grid.addWidget(self.lab_ins, self.grid_row+1, 1, 1, 2)
        self.grid.addWidget(self.edi_ins, self.grid_row+2, 1, 1, 2)
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
        self.setLayout(self.grid)

        self.showToolTips()
        self.setStyle()
        self.setCheckedRadioButton()
        self.setFields()
        self.setControlLock(True)

    #-------------------
    #  Public methods --
    #-------------------

    def showToolTips(self):
        #self        .setToolTip('Use this GUI to work with xtc file.')
        self.edi_ins.setToolTip('Instrument')
        self.edi_exp.setToolTip('Experiment')
        self.edi_run.setToolTip('Run number')
        self.edi_tag.setToolTip('Tag of the message.\nCan be used for \nfast search in ELog.')
        self.edi_res.setToolTip('Responce on message number.\nCan be used for accumulation of \nmessages in one branch of ELog.')
        self.edi_msg.setToolTip('Message for posting in ELog')
        self.edi_att.setToolTip('Attached file')
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

        width_edi_short =  40
        width_edi       =  80
        width_edi_long  = 420
        width_com       =  80
        width_com_short =  34

        for [label, edi, par, loc_par] in self.list_of_fields :
            label.setStyleSheet(cp.styleLabel)
            label.setAlignment(QtCore.Qt.AlignLeft)
            label.setFixedWidth(width_com)
            edi  .setFixedWidth(width_edi)

        self.lab_att.setFixedWidth(width_com_short)
        self.lab_msg.setFixedWidth(width_com_short)
        self.lab_ins.setFixedWidth(width_edi_short)
        self.lab_run.setFixedWidth(width_edi_short)
        self.edi_ins.setFixedWidth(width_edi_short)
        self.edi_run.setFixedWidth(width_edi_short)
        self.edi_msg.setMinimumWidth(width_edi_long)
        self.edi_att.setMinimumWidth(width_edi_long)


    def setCheckedRadioButton(self):
        for rad in self.list_of_rad :
            if cp.elog_post_rad.value() == str(rad.text()) : rad.setChecked(True)


    def onRadio(self) :
        self.setFields()


    def setFields(self):
        for rad in self.list_of_rad :
            if rad.isChecked() :
                rad_txt = str(rad.text()) 
                cp.elog_post_rad.setValue(rad_txt)
                logger.debug('Set fields for ' + rad_txt, __name__)
                if   rad_txt == 'Data'    : self.setFieldsData() 
                elif rad_txt == 'Dark'    : self.setFieldsDark()
                elif rad_txt == 'Saved'   : self.setFieldsSaved()
                elif rad_txt == 'Edit'    : self.setFieldsEdit()
                elif rad_txt == 'Default' : self.setFieldsDefault()
                else                      : self.setFieldsDefault()
                
                self.updateFields()
                self.setFieldsStyle()
                break;


    def setFieldsStyle(self):
        for [label, edi, par, loc_par]  in self.list_of_fields :
            if edi.isReadOnly() :
                edi.setStyleSheet (cp.styleEditInfo)
            else                :
                edi.setStyleSheet (cp.styleEdit)


    def setFieldsReadOnlySet1(self):
        self.edi_ins.setReadOnly(True)
        self.edi_exp.setReadOnly(True)
        self.edi_run.setReadOnly(True)
        self.edi_tag.setReadOnly(False)
        self.edi_res.setReadOnly(True)
        self.edi_msg.setReadOnly(False)
        self.edi_att.setReadOnly(True)


    def setFieldsReadOnlySet2(self):
        self.edi_ins.setReadOnly(False)
        self.edi_exp.setReadOnly(False)
        self.edi_run.setReadOnly(False)
        self.edi_tag.setReadOnly(False)
        self.edi_res.setReadOnly(False)
        self.edi_msg.setReadOnly(False)
        self.edi_att.setReadOnly(False)


    def setFieldsData(self):
        ins, exp, run, num = gu.parse_xtc_path(fnm.path_data_xtc())
        self.ins.setValue(str(ins))
        self.exp.setValue(str(exp))
        self.run.setValue(str(num))
        self.tag.setValue(cp.elog_post_tag.value())
        self.msg.setValue(cp.elog_post_msg.value())
        self.att.setValue(self.att_input)
        self.setFieldsReadOnlySet1()
        

    def setFieldsDark(self):
        ins, exp, run, num = gu.parse_xtc_path(fnm.path_dark_xtc())
        self.ins.setValue(str(ins))
        self.exp.setValue(str(exp))
        self.run.setValue(str(num))
        self.tag.setValue(cp.elog_post_tag.value())
        self.msg.setValue(cp.elog_post_msg.value())
        self.att.setValue(self.att_input)
        self.setFieldsReadOnlySet1()


    def setFieldsEdit(self):
        self.is_read_only = False
        #self.ins.setValue( cp.elog_post_ins.value() ) 
        #self.exp.setValue( cp.elog_post_exp.value() )
        #self.run.setValue( cp.elog_post_run.value() )
        #self.tag.setValue( cp.elog_post_tag.value() )
        #self.res.setValue( cp.elog_post_res.value() )
        #self.msg.setValue( cp.elog_post_msg.value() )
        #self.att.setValue( cp.elog_post_att.value() )
        self.att.setValue( self.att_input )
        self.setFieldsReadOnlySet2()


    def setFieldsSaved(self):
        self.ins.setValue( cp.elog_post_ins.value() )
        self.exp.setValue( cp.elog_post_exp.value() )
        self.run.setValue( cp.elog_post_run.value() )
        self.tag.setValue( cp.elog_post_tag.value() )
        self.res.setValue( cp.elog_post_res.value() )
        self.msg.setValue( cp.elog_post_msg.value() )
        self.att.setValue( cp.elog_post_att.value() )
        self.att.setValue( self.att_input )
        self.setFieldsReadOnlySet1()


    def setFieldsDefault(self):
        self.ins.setValue( cp.elog_post_ins.value_def() ) 
        self.exp.setValue( cp.elog_post_exp.value_def() )
        self.run.setValue( cp.elog_post_run.value_def() )
        self.tag.setValue( cp.elog_post_tag.value_def() )
        self.res.setValue( cp.elog_post_res.value_def() )
        self.msg.setValue( cp.elog_post_msg.value_def() )
        self.att.setValue( cp.elog_post_att.value_def() )
        self.att.setValue( self.att_input )
        self.setFieldsReadOnlySet1()


    def updateFields(self):
        self.edi_ins.setText( self.ins.value() ) 
        self.edi_exp.setText( self.exp.value() )
        self.edi_run.setText( self.run.value() )
        self.edi_tag.setText( self.tag.value() )
        self.edi_res.setText( self.res.value() )
        self.edi_msg.setText( self.msg.value() )
        self.edi_att.setText( self.att.value() )


    def updateConfigPars(self):
        cp.elog_post_ins.setValue( self.ins.value() ) 
        cp.elog_post_exp.setValue( self.exp.value() )
        cp.elog_post_run.setValue( self.run.value() )
        cp.elog_post_tag.setValue( self.tag.value() )
        cp.elog_post_res.setValue( self.res.value() )
        cp.elog_post_msg.setValue( self.msg.value() )
        cp.elog_post_att.setValue( self.att.value() )

    
    def setParent(self,parent) :
        self.parent = parent

    def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        self.frame.setGeometry(self.rect())

    def moveEvent(self, e):
        pass

    def closeEvent(self, event):
        logger.debug('closeEvent', __name__)


    def onClose(self):
        logger.debug('onClose', __name__)
        self.close()


    def onEdit(self):
        logger.debug('onEdit', __name__)
        for [label, edi, par, loc_par]  in self.list_of_fields :
            if  edi.isModified() :            
                edi.setModified(False)
                loc_par.setValue(str(edi.displayText()))
                msg = 'Set the local value of ' + str(label.text()) +\
                      ' ' + loc_par.value()
                logger.info(msg, __name__ )
                #print msg


    def onCBox(self):
        #if self.cbx_use .hasFocus() :
        par = cp.elog_post_cbx_state
        par.setValue( self.cbx_use.isChecked() )
        msg = 'onCBox - set status of ' + str(par.name()) + ': ' + str(par.value())
        logger.info(msg, __name__ )
        self.setFieldsStyle()


    def getListOfFields(self):
        return self.list_of_fields


    def setControlLock(self, isLocked):
        logger.info('setControlLock state: ' + str(isLocked) , __name__)
        for rad in self.list_of_rad :
            rad.setEnabled(not isLocked)
            #if rad.isChecked() : pass
            #else               : rad.setEnabled(isLocked)

#-----------------------------

if __name__ == "__main__" :

    app = QtGui.QApplication(sys.argv)
    widget = GUIELogPostingFields ()
    widget.show()
    app.exec_()

#-----------------------------
