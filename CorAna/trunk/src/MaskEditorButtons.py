#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module MaskEditorButtons...
#
#------------------------------------------------------------------------

"""Buttons for interactive plot of the image and spectrum for 2-d array

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule
@version $Id: 
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

from Logger                 import logger
#from GUIHelp                import *
#from GUIELogPostingDialog   import *
#from FileNameManager        import fnm
from ConfigParametersCorAna import confpars as cp

from DragWedge     import *
from DragLine      import *
from DragRectangle import *
from DragCircle    import *
from DragCenter    import *
from DragObjectSet import *

#---------------------
#  Class definition --
#---------------------

#class MaskEditorButtons (QtGui.QMainWindow) :
class MaskEditorButtons (QtGui.QWidget) :
    """Buttons for interactive plot of the image and spectrum for 2-d array."""

    #----------------
    #  Constructor --
    #----------------

    def __init__(self, parent=None, widgimage=None, ofname='./roi-mask.png', mfname='./roi-mask'):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('GUI of buttons')

        self.setFrame()

        self.parent    = parent
        self.ofname    = ofname
        self.mfname    = mfname
        self.widgimage = widgimage

        if widgimage != None :
            self.fig  = self.widgimage.fig
            self.axes = self.widgimage.axim
            self.set_lines      = DragObjectSet(self.fig, self.axes, DragLine,      useKeyboard=False)
            self.set_wedges     = DragObjectSet(self.fig, self.axes, DragWedge,     useKeyboard=False)
            self.set_rectangles = DragObjectSet(self.fig, self.axes, DragRectangle, useKeyboard=False)
            self.set_circles    = DragObjectSet(self.fig, self.axes, DragCircle,    useKeyboard=False)
            self.set_centers    = DragObjectSet(self.fig, self.axes, DragCenter,    useKeyboard=False, is_single_obj=True)
            self.disconnect_all()
            
        self.list_of_modes = ['Zoom', 'Add', 'Move', 'Select', 'Remove']
        self.list_of_forms = ['Line', 'Rectangle', 'Circle', 'Center', 'Wedge']
        self.list_of_buts  = []

        self.current_mode = self.list_of_modes[0]
        #self.current_form = self.list_of_forms[0]
        self.current_form = None

        self.tit_modes = QtGui.QLabel('Modes:')
        self.tit_forms = QtGui.QLabel('Forms:')

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.tit_modes)

        for mode in self.list_of_modes :
            but = QtGui.QPushButton(mode)
            self.list_of_buts.append(but)
            self.vbox.addWidget(but)
            self.connect(but, QtCore.SIGNAL('clicked()'), self.on_but)

        self.vbox.addStretch(1)
        self.vbox.addWidget(self.tit_forms)

        for form in self.list_of_forms :
            but = QtGui.QPushButton(form)
            self.list_of_buts.append(but)
            self.vbox.addWidget(but)
            self.connect(but, QtCore.SIGNAL('clicked()'), self.on_but)

        self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.showToolTips()
        self.setStyle()


    def showToolTips(self):
        #self.but_remove.setToolTip('Remove') 
        pass

    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)

    def setStyle(self):
        self           .setFixedWidth(90)
        #self           .setStyleSheet (cp.styleBkgd) 
        self.tit_modes .setStyleSheet (cp.styleLabel)
        self.tit_forms .setStyleSheet (cp.styleLabel)
        self.tit_modes .setAlignment  (QtCore.Qt.AlignCenter)
        self.tit_forms .setAlignment  (QtCore.Qt.AlignCenter)

        self.setButtonStyle()


    def setButtonStyle(self):
        for but in self.list_of_buts :
            but_text = str(but.text()) 
            if   but_text == self.current_mode : but.setStyleSheet (cp.styleButtonGood)
            elif but_text == self.current_form : but.setStyleSheet (cp.styleButtonGood)
            else                               : but.setStyleSheet (cp.styleButton)
 
        #self.but_help  .setStyleSheet (cp.styleButtonGood) 
        #self.but_quit  .setStyleSheet (cp.styleButtonBad) 

        #self.edi_nbins.setValidator(QtGui.QIntValidator(1,1000,self))


    def resizeEvent(self, e):
        self.frame.setGeometry(self.rect())


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        #print 'Close application'
        try    : self.guihelp.close()
        except : pass

        #try    : self.parent.close()
        #except : pass
           

    def on_but_quit(self):
        logger.debug('on_but_quit', __name__ )
        self.close()


    #def set_buttons(self) :
    #    self.cbox_grid.setChecked(self.fig.myGridIsOn)
    #    self.cbox_log .setChecked(self.fig.myLogIsOn)
    #    self.edi_nbins.setText(self.stringOrNone(self.fig.myNBins))


    def get_pushed_but(self):
        for but in self.list_of_buts :
            if but.hasFocus() : return but

 
    def disconnect_all(self):
        self.set_lines     .disconnect_objs()    
        self.set_rectangles.disconnect_objs()    
        self.set_circles   .disconnect_objs()    
        self.set_centers   .disconnect_objs()    
        self.set_wedges    .disconnect_objs()    


    def on_but(self):
        but = self.get_pushed_but()
        msg = 'on_but ' + str(but.text())
        logger.debug(msg, __name__ )
        print msg

        but_text = str(but.text())
        if but_text in self.list_of_modes :
            self.current_mode = but_text
            self.fig.my_mode  = but_text

            if self.current_mode == 'Zoom' :
                self.widgimage.connectZoomMode()
                self.disconnect_all()
                self.current_form = None
            else :
                self.widgimage.disconnectZoomMode()

        if but_text in self.list_of_forms :

            if   self.current_form == 'Line'      : self.set_lines     .disconnect_objs()    
            elif self.current_form == 'Rectangle' : self.set_rectangles.disconnect_objs()    
            elif self.current_form == 'Circle'    : self.set_circles   .disconnect_objs()    
            elif self.current_form == 'Center'    : self.set_centers   .disconnect_objs()    
            elif self.current_form == 'Wedge'     : self.set_wedges    .disconnect_objs()    
            else                                  : pass

            self.current_form = but_text

            if   self.current_form == 'Line'      : self.set_lines     .connect_objs()    
            elif self.current_form == 'Rectangle' : self.set_rectangles.connect_objs()    
            elif self.current_form == 'Circle'    : self.set_circles   .connect_objs()    
            elif self.current_form == 'Center'    : self.set_centers   .connect_objs()    
            elif self.current_form == 'Wedge'     : self.set_wedges    .connect_objs()    
            else                                  : pass

        self.setButtonStyle()
 

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
                                                       filter = '*.png, *.eps, *pdf, *.ps'
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
         


    #def on_but_help(self):
    #    logger.debug('on_but_help - is not implemented yet...', __name__ )
    #    try :
    #        self.guihelp.close()
    #        del self.guihelp
    #    except :
    #        self.guihelp = GUIHelp(None,self.help_message())
    #        self.guihelp.setFixedSize(620,160) 
    #        self.guihelp.move(self.parentWidget().pos().__add__(QtCore.QPoint(250,60))) 
    #        self.guihelp.show()


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

    w = MaskEditorButtons(None)
    w.move(QtCore.QPoint(50,50))
    w.show()

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
