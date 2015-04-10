#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotG2Buttons...
#
#------------------------------------------------------------------------

"""Buttons for plot

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule
@version $Id: 
@author Mikhail S. Dubrovin
"""

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

from ConfigParametersCorAna import confpars as cp # for icons
from Logger                 import logger
from GUIHelp                import *
from GUIELogPostingDialog   import *

#---------------------
#  Class definition --
#---------------------

#class PlotG2Buttons (QtGui.QMainWindow) :
class PlotG2Buttons (QtGui.QWidget) :
    """Buttons for time records plot"""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, widgimage=None, ofname='./fig.png'):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('GUI of buttons')

        self.setFrame()

        self.parent    = parent
        self.ofname    = ofname

        if widgimage is None :
            self.widgimage          = self
            self.widgimage.iq_begin = 0
            self.widgimage.gridIsOn = True
            self.widgimage.logIsOn  = False
        else :
            self.widgimage = widgimage

        self.but_reset = QtGui.QPushButton('&Reset')
        self.but_help  = QtGui.QPushButton('&Help')
        self.but_save  = QtGui.QPushButton('&Save')
        self.but_elog  = QtGui.QPushButton('&ELog') # u'\u2192 &ELog'
        self.but_quit  = QtGui.QPushButton('&Close')
        self.cbox_grid = QtGui.QCheckBox('&Grid')
        self.cbox_log  = QtGui.QCheckBox('&LogX')
        self.tit_nbins = QtGui.QLabel('1-st plot:')
        self.edi_nbins = QtGui.QLineEdit(self.stringOrNone(self.widgimage.iq_begin))
        self.but_inc   = QtGui.QPushButton(u'\u25B6') # right-head triangle
        self.but_dec   = QtGui.QPushButton(u'\u25C0') # left-head triangle
 
        self.set_buttons()
        self.setIcons()

        self.connect(self.but_help,  QtCore.SIGNAL('clicked()'),          self.on_but_help)
        self.connect(self.but_reset, QtCore.SIGNAL('clicked()'),          self.on_but_reset)
        self.connect(self.but_save,  QtCore.SIGNAL('clicked()'),          self.on_but_save)
        self.connect(self.but_elog,  QtCore.SIGNAL('clicked()'),          self.on_but_elog)
        self.connect(self.but_quit,  QtCore.SIGNAL('clicked()'),          self.on_but_quit)
        self.connect(self.but_inc,   QtCore.SIGNAL('clicked()'),          self.on_but_inc)
        self.connect(self.but_dec,   QtCore.SIGNAL('clicked()'),          self.on_but_dec)
        self.connect(self.edi_nbins, QtCore.SIGNAL('editingFinished ()'), self.on_edit_nbins)
        self.connect(self.cbox_grid, QtCore.SIGNAL('stateChanged(int)'),  self.on_cbox_grid)
        self.connect(self.cbox_log,  QtCore.SIGNAL('stateChanged(int)'),  self.on_cbox_log)

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.but_help)
        self.hbox.addWidget(self.tit_nbins)
        self.hbox.addWidget(self.but_dec)
        self.hbox.addWidget(self.edi_nbins)
        self.hbox.addWidget(self.but_inc)
        self.hbox.addWidget(self.cbox_grid)
        self.hbox.addWidget(self.cbox_log)
        self.hbox.addWidget(self.but_reset)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_save)
        self.hbox.addWidget(self.but_elog)
        self.hbox.addWidget(self.but_quit)
        self.setLayout(self.hbox)

        self.showToolTips()
        self.setStyle()



    def setIcons(self) :
        cp.setIcons()
        self.but_elog .setIcon(cp.icon_mail_forward)
        self.but_save .setIcon(cp.icon_save)
        self.but_quit .setIcon(cp.icon_exit)
        self.but_help .setIcon(cp.icon_help)
        self.but_reset.setIcon(cp.icon_reset)
                

    def showToolTips(self):
        self.but_reset.setToolTip('Reset original view') 
        self.but_quit .setToolTip('Quit this GUI') 
        self.but_save .setToolTip('Save the figure in file') 
        self.but_elog .setToolTip('Send figure to ELog') 
        self.but_help .setToolTip('Click on this button\nand get help') 
        self.cbox_grid.setToolTip('On/Off grid') 
        self.cbox_log .setToolTip('Log/Linear scale') 
        self.edi_nbins.setToolTip('Edit the first plot index: [0:end]') #'+str(self.widgimage.arr_q.shape[0]-1) + ']')


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self           .setFixedHeight(50)
        self.but_inc   .setFixedWidth (30)
        self.but_dec   .setFixedWidth (30)
        self.edi_nbins .setFixedWidth (40)

        self           .setStyleSheet (cp.styleBkgd) 
        self.tit_nbins .setStyleSheet (cp.styleLabel)
        self.but_help  .setStyleSheet (cp.styleButtonGood) 
        self.but_reset .setStyleSheet (cp.styleButton) 
        self.but_save  .setStyleSheet (cp.styleButton) 
        self.but_quit  .setStyleSheet (cp.styleButtonBad) 
        self.but_inc   .setStyleSheet (cp.styleButton)
        self.but_dec   .setStyleSheet (cp.styleButton)
        self.edi_nbins .setStyleSheet (cp.styleEdit)

        self.edi_nbins.setValidator(QtGui.QIntValidator(1,1000,self))


 
    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        #print 'Close application'
        try    : self.guihelp.close()
        except : pass

        try    : self.parent.close()
        except : pass
           

    def on_but_quit(self):
        logger.debug('on_but_quit', __name__ )
        self.close()


    def stringOrNone(self,value):
        if value is None : return 'None'
        else             : return str(value)


    def intOrNone(self,value):
        if value is None : return None
        else             : return int(value)


    def set_buttons(self) :
        self.cbox_grid.setChecked(self.widgimage.gridIsOn)
        self.cbox_log .setChecked(self.widgimage.logIsOn)
        self.edi_nbins.setText(self.stringOrNone(self.widgimage.iq_begin))


    def on_edit_nbins(self):
        self.widgimage.iq_begin = int(self.edi_nbins.displayText())
        logger.info('Set for spectrum the number of bins ='+str(self.widgimage.iq_begin), __name__ )
        self.widgimage.on_draw_in_limits()


    def get_iq_begin(self, iq=0) :
        self.iq_min = 0
        self.iq_max = self.widgimage.arr_q.shape[0]
        nwin_max    = self.widgimage.nwin_max
        if   iq < self.iq_min             : return self.iq_min
        elif self.iq_max <= nwin_max      : return self.iq_min
        elif self.iq_max-iq <= nwin_max   : return self.iq_max-nwin_max
        else                              : return iq


    def change_iq_begin(self, diq):
        logger.debug('change iq by %d'% (diq), __name__ )
        iq = int(self.edi_nbins.displayText())
        iq_begin = self.get_iq_begin(iq + diq)
        if iq_begin == iq :
            logger.info('1-st plot is on limit of allowed interval [%d, %d] and is not changed.'%(self.iq_min,self.iq_max), __name__ )
            return
        self.edi_nbins.setText(str(iq_begin))
        self.widgimage.iq_begin = iq_begin 
        self.widgimage.on_draw_in_limits()

 
    def on_but_inc(self):
        self.change_iq_begin( self.widgimage.nwin_max )


    def on_but_dec(self):
        self.change_iq_begin( -self.widgimage.nwin_max )


    def on_but_reset(self):
        logger.debug('on_but_reset', __name__ )
        self.widgimage.initParameters()
        self.set_buttons()
        self.widgimage.on_draw()


    def on_but_save(self):
        logger.debug('on_but_save', __name__ )
        path = self.ofname
        #dir, fname = os.path.split(path)
        path  = str( QtGui.QFileDialog.getSaveFileName(self,
                                                       caption='Select file to save the plot',
                                                       directory = path,
                                                       filter = '*.png *.eps *pdf *.ps'
                                                       ) )
        if path == '' :
            logger.debug('Saving is cancelled.', __name__ )
            return
        logger.info('Save plot in file: ' + path, __name__ )
        self.widgimage.saveFigure(path)


    def on_but_elog(self):
        logger.info('Send message to ELog:', __name__ )
        path = self.ofname
        logger.info('1. Save plot in file: ' + path, __name__ )
        self.widgimage.saveFigure(path)
        logger.info('2. Submit message with plot to ELog', __name__ )
        wdialog = GUIELogPostingDialog (self, fname=path)
        resp=wdialog.exec_()
         

    def on_cbox_log(self):
        logger.info('On/Off log scale for histogram', __name__ )
        self.widgimage.logIsOn = self.cbox_log.isChecked()
        self.widgimage.on_draw_in_limits()


    def on_cbox_grid(self):
        logger.info('On/Off grid.', __name__ )
        self.widgimage.gridIsOn = self.cbox_grid.isChecked()
        self.widgimage.on_draw_in_limits()


    def on_but_help(self):
        logger.debug('on_but_help - is not implemented yet...', __name__ )
        try :
            self.guihelp.close()
            del self.guihelp
        except :
            self.guihelp = GUIHelp(None,self.help_message())
            #self.guihelp.setMinimumSize(600,150) 
            self.guihelp.setFixedSize(610,130) 
            try   : self.guihelp.move(self.parentWidget().pos().__add__(QtCore.QPoint(250,60))) 
            except: self.guihelp.move(self.pos().__add__(QtCore.QPoint(250,60))) 
            self.guihelp.show()


    def help_message(self):
        msg  = 'Mouse control buttons:'
        msg += '\nLeft/right mouse click on graph or hist window sets min/max limit for both plots.'
        msg += '\nMiddle mouse button click returns default limits for horizontal axis.'
        msg += '\n"Reset" button - resets all default parameters/limits.'
        return msg


    def popup_confirmation_box(self):
        """Pop-up box for help"""
        msg = QtGui.QMessageBox(self, windowTitle='Help for interactive plot',
            text='This is a help',
            #standardButtons=QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
            standardButtons=QtGui.QMessageBox.Close)

        msg.setDefaultButton(msg.Close)
        clicked = msg.exec_()

        #if   clicked == QtGui.QMessageBox.Save :
        #    logger.debug('Saving is requested', __name__)
        #elif clicked == QtGui.QMessageBox.Discard :
        #    logger.debug('Discard is requested', __name__)
        #else :
        #    logger.debug('Cancel is requested', __name__)
        #return clicked

        
#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)

    w = PlotG2Buttons(None)
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
