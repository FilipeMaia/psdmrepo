#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotImgSpeButtons...
#
#------------------------------------------------------------------------

"""Buttons for interactive plot of the image and spectrum for 2-d array

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule
@version $Id$ 
@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

from PyQt4 import QtGui, QtCore

from Logger                 import logger
from GUIHelp                import *
#from GUIELogPostingDialog   import *
import GlobalUtils          as     gu

from GUIRangeIntensity      import *

from FileNameManager        import fnm
from ConfigParametersForApp import cp

from CorAna.ArrFileExchange import *

#---------------------
#  Class definition --
#---------------------

#class PlotImgSpeButtons (QtGui.QMainWindow) :
class PlotImgSpeButtons (QtGui.QWidget) :
    """Buttons for interactive plot of the image and spectrum for 2-d array."""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, widgimage=None, ifname='', ofname='./fig.png', help_msg=None, expand=False, fexmod=False, verb=False):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('GUI of buttons')

        self.setFrame()

        self.parent      = parent
        self.ifname      = ifname
        self.ofname      = ofname
        self.is_expanded = expand
        self.guirange    = None
        self.fexmod      = fexmod
        self.verb        = verb

        self.widgimage = widgimage
        if widgimage is None :
            self.fig = self # need it to pass pars
            self.fig.myNBins    = 40
            self.fig.myGridIsOn = False    
            self.fig.myLogXIsOn = False    
            self.fig.myLogYIsOn = False    
            self.fig.myZmin     = None
            self.fig.myZmax     = None
        else :
            self.fig = widgimage.fig

        if help_msg is None : self.help_msg = self.help_message()
        else                : self.help_msg = help_msg

        tit_more_less  = '&Less' if expand else '&More'
        self.but_more  = QtGui.QPushButton(tit_more_less)
        self.but_reset = QtGui.QPushButton('&Reset')
        self.but_help  = QtGui.QPushButton('&Help')
        self.but_load  = QtGui.QPushButton('Load')
        self.but_diff  = QtGui.QPushButton('Diff')
        self.but_save  = QtGui.QPushButton('&Save')
        self.but_elog  = QtGui.QPushButton('&ELog') #u'\u2192 &ELog'
        self.but_quit  = QtGui.QPushButton('&Close')
        self.cbox_grid = QtGui.QCheckBox('&Grid')
        self.cbox_logx = QtGui.QCheckBox('&X')
        self.cbox_logy = QtGui.QCheckBox('&Y')
        self.tit_log   = QtGui.QLabel('Log:')
        self.tit_nbins = QtGui.QLabel('N bins:')
        self.edi_nbins = QtGui.QLineEdit(self.stringOrNone(self.fig.myNBins))

        self.set_buttons()
        self.setIcons()

        width = 60
        self.edi_nbins.setFixedWidth(width)
        #self.but_reset.setFixedWidth(width)
        #self.but_help .setFixedWidth(width)
        #self.but_save .setFixedWidth(width)
        #self.but_elog .setFixedWidth(width)
        #self.but_quit .setFixedWidth(width)
        self.edi_nbins.setValidator(QtGui.QIntValidator(1,1000,self))
 
        self.but_help .setStyleSheet (cp.styleButtonGood) 
        self.but_more .setStyleSheet (cp.styleButton) 
        self.but_reset.setStyleSheet (cp.styleButton) 
        self.but_load .setStyleSheet (cp.styleButton) 
        self.but_diff .setStyleSheet (cp.styleButton) 
        self.but_save .setStyleSheet (cp.styleButton) 
        self.but_quit .setStyleSheet (cp.styleButtonBad) 

        self.connect(self.but_more,  QtCore.SIGNAL('clicked()'),          self.on_but_more)
        self.connect(self.but_help,  QtCore.SIGNAL('clicked()'),          self.on_but_help)
        self.connect(self.but_reset, QtCore.SIGNAL('clicked()'),          self.on_but_reset)
        self.connect(self.but_load,  QtCore.SIGNAL('clicked()'),          self.on_but_load)
        self.connect(self.but_diff,  QtCore.SIGNAL('clicked()'),          self.on_but_diff)
        self.connect(self.but_save,  QtCore.SIGNAL('clicked()'),          self.on_but_save)
        self.connect(self.but_elog,  QtCore.SIGNAL('clicked()'),          self.on_but_elog)
        self.connect(self.but_quit,  QtCore.SIGNAL('clicked()'),          self.on_but_quit)
        self.connect(self.cbox_grid, QtCore.SIGNAL('stateChanged(int)'),  self.on_cbox_grid)
        self.connect(self.cbox_logx, QtCore.SIGNAL('stateChanged(int)'),  self.on_cbox_logx)
        self.connect(self.cbox_logy, QtCore.SIGNAL('stateChanged(int)'),  self.on_cbox_logy)
        self.connect(self.edi_nbins, QtCore.SIGNAL('editingFinished ()'), self.on_edit_nbins)

        #self.setGridLayout()        
        self.setPanelLayout()        
        self.showToolTips()
        #self.setFixedHeight(50)

        pbits = 377 if self.verb else 0
        self.afe_rd = ArrFileExchange(prefix=self.ifname, rblen=3, print_bits=pbits)


    def setIcons(self) :
        cp.setIcons()
        self.but_elog .setIcon(cp.icon_mail_forward)
        self.but_load .setIcon(cp.icon_browser) # icon_contents)
        self.but_diff .setIcon(cp.icon_minus) # icon_contents)
        self.but_save .setIcon(cp.icon_save)
        self.but_quit .setIcon(cp.icon_exit)
        self.but_help .setIcon(cp.icon_help)
        self.but_reset.setIcon(cp.icon_reset)


    def setPanelLayoutV1(self):
        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.but_help)
        self.hbox.addWidget(self.tit_nbins)
        self.hbox.addWidget(self.edi_nbins)
        self.hbox.addWidget(self.cbox_grid)
        self.hbox.addWidget(self.tit_log)
        self.hbox.addWidget(self.cbox_logx)
        self.hbox.addWidget(self.cbox_logy)
        self.hbox.addWidget(self.but_reset)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_load)
        self.hbox.addWidget(self.but_diff)
        self.hbox.addWidget(self.but_save)
        self.hbox.addWidget(self.but_elog)
        self.hbox.addWidget(self.but_quit)
        self.hbox.addWidget(self.but_more)
        self.setLayout(self.hbox)

        self.setPannel()


    def setPanelLayout(self):
        self.hbox1 = QtGui.QHBoxLayout()
        self.hbox1.addWidget(self.but_help)
        self.hbox1.addWidget(self.tit_nbins)
        self.hbox1.addWidget(self.edi_nbins)
        self.hbox1.addWidget(self.cbox_grid)
        self.hbox1.addWidget(self.tit_log)
        self.hbox1.addWidget(self.cbox_logx)
        self.hbox1.addWidget(self.cbox_logy)
        self.hbox1.addWidget(self.but_reset)
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.but_elog)
        self.hbox1.addWidget(self.but_quit)
        self.hbox1.addWidget(self.but_more)

        self.guirange = GUIRangeIntensity(self, None, None, txt_from='Spec range', txt_to=':')

        self.hbox2 = QtGui.QHBoxLayout()
        self.hbox2.addWidget(self.but_load)
        self.hbox2.addWidget(self.but_diff)
        self.hbox2.addWidget(self.guirange)
        self.hbox2.addStretch(1)
        self.hbox2.addWidget(self.but_save)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.hbox2)
        self.vbox.addLayout(self.hbox1)
        self.vbox.addStretch(1)

        self.setLayout(self.vbox)
        self.setContentsMargins (QtCore.QMargins(0,-5,0,-5))
        self.setPannel()


    def setZMin(self, zmin=None) :
        if self.guirange is None : return
        self.guirange.setParamFrom(zmin)


    def setZMax(self, zmin=None) :
        if self.guirange is None : return
        self.guirange.setParamTo(zmax)


    def setZRange(self, str_from=None, str_to=None) :
        if self.guirange is None : return
        self.guirange.setParams(str_from, str_to)


    def setPannel(self):

        self.but_quit.setVisible(False)
        self.but_elog.setVisible(False)
        #self.but_help.setVisible(False)
        self.but_load.setVisible(self.is_expanded)
        self.but_diff.setVisible(self.is_expanded)
        self.but_save.setVisible(self.is_expanded)
        self.guirange.setVisible(self.is_expanded)
        
        height = 78 if self.is_expanded else 40
        #self.setMinimumHeight(height)
        self.setFixedHeight(height)


    def setGridLayout(self):
        self.grid = QtGui.QGridLayout() 
        self.grid.addWidget(self.but_help,  0, 0)
        self.grid.addWidget(self.tit_nbins, 0, 1)
        self.grid.addWidget(self.edi_nbins, 0, 2)
        self.grid.addWidget(self.cbox_grid, 0, 3)
        self.grid.addWidget(self.tit_log,   0, 4)
        self.grid.addWidget(self.cbox_logx, 0, 5)
        self.grid.addWidget(self.cbox_logy, 0, 6)
        self.grid.addWidget(self.but_reset, 0, 7)
        self.grid.addWidget(self.but_load,  0, 8)
        self.grid.addWidget(self.but_diff,  0, 9)
        self.grid.addWidget(self.but_save,  0, 10)
        self.grid.addWidget(self.but_quit,  0, 11)
        self.grid.addWidget(self.but_more,  0, 12)
        self.setLayout(self.grid)


    def showToolTips(self):
        self.but_reset.setToolTip('Reset original view') 
        self.but_quit .setToolTip('Quit this GUI') 
        self.but_load .setToolTip('Load image from file') 
        self.but_load .setToolTip('Load subtracting image from file') 
        self.but_save .setToolTip('Save the figure in file') 
        self.but_elog .setToolTip('Send figure to ELog') 
        self.but_help .setToolTip('Click on this button\nand get help') 
        self.cbox_grid.setToolTip('On/Off grid') 
        self.cbox_logx.setToolTip('Log/Linear X scale') 
        self.cbox_logy.setToolTip('Log/Linear Y scale') 
        self.edi_nbins.setToolTip('Edit the number of bins\nfor spectrum [1-1000]')


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
        #print 'PlotImgSpeButtons resizeEvent: %s' % str(self.size())


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
        self.cbox_grid.setChecked(self.fig.myGridIsOn)
        self.cbox_logx.setChecked(self.fig.myLogXIsOn)
        self.cbox_logy.setChecked(self.fig.myLogYIsOn)
        self.edi_nbins.setText(self.stringOrNone(self.fig.myNBins))


    def on_edit_nbins(self):
        self.fig.myNBins = int(self.edi_nbins.displayText())
        logger.info('Set for spectrum the number of bins ='+str(self.fig.myNBins), __name__ )
        self.widgimage.processDraw()

 
    def on_but_more(self):
        logger.debug('on_but_more', __name__ )
        if self.is_expanded :
            self.but_more.setText('&More')
            self.is_expanded = False
        else :
            self.but_more.setText('&Less')
            self.is_expanded = True
        self.setPannel()


    def on_but_reset(self):
        logger.debug('on_but_reset', __name__ )
        self.widgimage.initParameters()
        self.set_buttons()
        self.widgimage.on_draw()


    def on_but_load(self):
        logger.debug('on_but_load', __name__ )

        if self.fexmod :
            if self.afe_rd.is_new_arr_available() :
                logger.info('WAIT for image loading', __name__ )
                arr = self.afe_rd.get_arr_latest()             
                self.widgimage.set_image_array_new(arr,
                                               rot_ang_n90 = self.widgimage.rot_ang_n90,
                                               y_is_flip   = self.widgimage.y_is_flip)
                                               #title='Image from %s...' % self.ifname,
                logger.info('Image is loaded', __name__ )
                return
            else :
                logger.info('New image is N/A !', __name__ )
                return
        
        file_filter = 'Files (*.txt *.data *.npy)\nAll files (*)'
        path = gu.get_open_fname_through_dialog_box(self, self.ifname, 'Select file with image', filter=file_filter)
        if path is None or path == '' :
            logger.info('Loading is cancelled...', __name__ )
            return

        self.ifname = path


        arr = gu.get_image_array_from_file(path) # dtype=np.float32)
        if arr is None : return

        #arr = gu.get_array_from_file(path) # dtype=np.float32)
        #print 'arr:\n', arr
        self.widgimage.set_image_array_new(arr,
                                           rot_ang_n90 = self.widgimage.rot_ang_n90,
                                           y_is_flip   = self.widgimage.y_is_flip)



    def on_but_diff(self):
        logger.info('on_but_diff', __name__ )

        list_of_opts = ['Load from WORK directory',
                        'Load from CALIB directory',
                        'Load initial image',
                        'Cancel'
                        ]

        selected = gu.selectFromListInPopupMenu(list_of_opts)
        logger.debug('selected option: %s' % selected, __name__ )

        path0 = self.ifname
        if selected is None              : return
        elif selected == list_of_opts[0] : path0 = fnm.path_dir_work()
        elif selected == list_of_opts[1] : path0 = fnm.path_to_calib_dir()
        elif selected == list_of_opts[2] : path0 = self.ifname
        elif selected == list_of_opts[3] : return
        else                             : return

        file_filter = 'Files (*.txt *.data *.npy)\nAll files (*)'
        path = gu.get_open_fname_through_dialog_box(self, path0, 'Select file to subtract', filter=file_filter)
        if path is None or path == '' :
            logger.info('Loading is cancelled...', __name__ )
            return

        arr_sub = gu.get_image_array_from_file(path) # dtype=np.float32)
        if arr_sub is None : return

        if arr_sub.size != self.widgimage.arr.size :
            msg = 'Subtracted array size: %d is different from current image size: %d. Diff plotting is cancelled.' % \
                  (arr_sub.size, self.widgimage.arr.size)
            logger.warning(msg, __name__ )
            #print msg
            return

        self.widgimage.subtract_from_image_array(arr_sub)



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
         

    def on_cbox_logx(self):
        logger.info('Set log10 for X scale', __name__ )
        self.fig.myLogXIsOn = self.cbox_logx.isChecked()
        self.fig.myZmin    = None
        self.fig.myZmax    = None        
        self.widgimage.processDraw()


    def on_cbox_logy(self):
        logger.info('Set log10 for Y scale', __name__ )
        self.fig.myLogYIsOn = self.cbox_logy.isChecked()
        self.fig.myZmin    = None
        self.fig.myZmax    = None        
        self.widgimage.processDraw()


    def on_cbox_grid(self):
        logger.info('On/Off grid.', __name__ )
        self.fig.myGridIsOn = self.cbox_grid.isChecked()
        self.widgimage.processDraw()


    def on_but_help(self):
        logger.debug('on_but_help - is not implemented yet...', __name__ )
        try :
            self.guihelp.close()
            del self.guihelp
        except :
            self.guihelp = GUIHelp(None, self.help_msg)
            self.guihelp.setMinimumSize(620, 200) 
            self.guihelp.move(self.parentWidget().pos().__add__(QtCore.QPoint(250,60))) 
            self.guihelp.show()


    def help_message(self):
        msg  = 'Mouse control functions:'
        msg += '\nZoom-in image: left mouse click, move, and release in another image position.'
        msg += '\nMiddle mouse button click on image - restores full size image'
        msg += '\nLeft/right mouse click on histogram or color bar - sets min/max amplitude.' 
        msg += '\nMiddle mouse click on histogram or color bar - resets amplitude limits to default.'
        msg += '\n"Reset" button - resets all parameters to default values.'
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

    w = PlotImgSpeButtons(None, is_expanded=True)
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
