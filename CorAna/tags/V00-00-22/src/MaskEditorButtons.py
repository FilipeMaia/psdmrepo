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
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import time
import numpy as np

from PyQt4 import QtGui, QtCore

from Logger                 import logger
#from GUIHelp                import *
#from GUIELogPostingDialog   import *
#from FileNameManager        import fnm
from ConfigParametersCorAna import confpars as cp
import GlobalUtils          as     gu
from CorAna.ArrFileExchange import *

from DragWedge     import *
from DragLine      import *
from DragRectangle import *
from DragCircle    import *
from DragCenter    import *
from DragPolygon   import *
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

    def __init__(self, parent=None, widgimage=None, ifname=None, ofname='./roi-mask.png', mfname='./roi-mask', \
                 xyc=None, lw=2, col='b', picker=5, verb=False, ccd_rot_n90=0, y_is_flip=False, fexmod=False):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('GUI of buttons')

        self.setFrame()

        self.parent      = parent
        self.ifname      = ifname
        self.ofname      = ofname
        self.mfname      = mfname
        self.fexmod      = fexmod

        self.verb        = verb
        self.ccd_rot_n90 = ccd_rot_n90
        self.y_is_flip   = y_is_flip

        
        mfroot, mfext = os.path.splitext(self.mfname)
        self.mfname_img  = ifname
        self.mfname_mask = self.mfname + '.txt' if mfext == '' else self.mfname
        self.mfname_objs = mfroot + '-shape-objs.txt'

        self.widgimage = widgimage


        if self.widgimage is not None :
            self.fig        = self.widgimage.fig
            self.axes       = self.widgimage.get_axim()

            if xyc is not None : self.fig.my_xyc = xyc
            else               : self.fig.my_xyc = self.widgimage.get_xy_img_center()

            if self.verb : print 'Image center: ', self.fig.my_xyc

            self.set_lines      = DragObjectSet(self.fig, self.axes, DragLine,      useKeyboard=False, lw=lw, col=col, picker=picker)
            self.set_wedges     = DragObjectSet(self.fig, self.axes, DragWedge,     useKeyboard=False, lw=lw, col=col, picker=picker, use_xyc=True)
            self.set_rectangles = DragObjectSet(self.fig, self.axes, DragRectangle, useKeyboard=False, lw=lw, col=col, picker=picker)
            self.set_circles    = DragObjectSet(self.fig, self.axes, DragCircle,    useKeyboard=False, lw=lw, col=col, picker=picker)
            #self.set_centers    = DragObjectSet(self.fig, self.axes, DragCenter,    useKeyboard=False, lw=lw, col=col, picker=picker, is_single_obj=True)
            self.set_polygons   = DragObjectSet(self.fig, self.axes, DragPolygon,   useKeyboard=False, lw=lw, col=col, picker=picker)
            self.disconnect_all()

        else :              # for self-testing mode only
            self.fig = self # in order to get rid of crash... 

            
        self.list_of_modes   = ['Zoom', 'Add', 'Move', 'Select', 'Remove']
        self.list_of_forms   = ['Rectangle', 'Wedge', 'Circle', 'Line', 'Polygon'] #, 'Center'] 
        #self.list_of_io_tits = ['Load Image', 'Load Forms', 'Save Forms', 'Save Mask', 'Save Inv-M', 'Print Forms', 'Clear Forms', 'Excg-Load', 'Excg-Save']
        self.list_of_io_tits = ['Load Image', 'Load Forms', 'Save Forms', 'Save Mask', 'Save Inv-M', 'Print Forms', 'Clear Forms']
        self.list_of_io_tips = ['Load image for \ndisplay from file',
                                'Load forms of masked \nregions from file',
                                'Save forms of masked \nregions in text file',
                                'Save mask as a 2D array \nof ones and zeros in text file',
                                'Save inversed-mask as a 2D array\n of ones and zeros in text file',
                                'Prints parameters of \ncurrently entered forms',
                                'Clear all forms from the image'
                                ]
                                #'Load image from file with pre-defined name',
                                #'Save mask in file with pre-defined name'
        #self.list_of_fnames  = [self.mfname_img, self.mfname_objs, self.mfname_objs, self.mfname_mask, self.mfname_mask, None, None, self.mfname_img, self.mfname_mask]
        self.list_of_fnames  = [self.mfname_img, self.mfname_objs, self.mfname_objs, self.mfname_mask, self.mfname_mask, None, None]

        zoom_tip_msg = 'Zoom mode for image and spactrom.' + \
                       '\nZoom-in image: left mouse button click-drug-release.' + \
                       '\nZoom-in spectrum: left/right mouse button click for min/max limit.' + \
                       '\nReset to full size: middle mouse button click.'

        self.list_of_buts    = []
        self.list_of_io_buts = []

        self.fig.my_mode = self.list_of_modes[0]
        #self.current_form = self.list_of_forms[0]
        self.current_form = None

        self.tit_modes  = QtGui.QLabel('Modes:')
        self.tit_forms  = QtGui.QLabel('Forms:')
        self.tit_io     = QtGui.QLabel('I/O:')
        self.tit_status = QtGui.QLabel('Status:')
        self.lab_status = QtGui.QPushButton('Good')
        #self.lab_status = QtGui.QLabel('Good')
        #self.lab_status = QtGui.QTextEdit()

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.tit_forms)
        for form in self.list_of_forms :
            but = QtGui.QPushButton(form)
            self.list_of_buts.append(but)
            self.vbox.addWidget(but)
            self.connect(but, QtCore.SIGNAL('clicked()'), self.on_but)
            but.setToolTip('Select form to draw on image \nas ROI or masked region')

        self.vbox.addStretch(1)
        self.vbox.addWidget(self.tit_modes)
        for mode in self.list_of_modes :
            but = QtGui.QPushButton(mode)
            self.list_of_buts.append(but)
            self.vbox.addWidget(but)
            self.connect(but, QtCore.SIGNAL('clicked()'), self.on_but)
            if   mode == 'Zoom'   : but.setToolTip(zoom_tip_msg)
            elif mode == 'Select' : but.setToolTip('Select forms for inversed masking. \nSelected forms are marked by different color')
            else                  : but.setToolTip('Select mode of manipulation with form')

        self.vbox.addStretch(1)
        self.vbox.addWidget(self.tit_io)
        for io_tit, io_tip in zip(self.list_of_io_tits, self.list_of_io_tips) :
            but = QtGui.QPushButton(io_tit)
            self.list_of_io_buts.append(but)
            self.vbox.addWidget(but)
            self.connect(but, QtCore.SIGNAL('clicked()'), self.on_io_but)
            but.setToolTip(io_tip)

        self.vbox.addStretch(1)
        self.vbox.addWidget(self.tit_status)
        self.vbox.addWidget(self.lab_status)

        self.setLayout(self.vbox)

        self.showToolTips()
        self.setStyle()

        self.setStatus()

        pbits = 377 if self.verb else 0
        self.afe_rd = ArrFileExchange(prefix=self.ifname, rblen=3, print_bits=pbits)
        self.afe_wr = ArrFileExchange(prefix=self.mfname, rblen=3, print_bits=pbits)


    #def updateCenter(self,x,y):
    #    print 'updateCenter, x,y=', x, y 


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
        self           .setFixedWidth(100)
        #self           .setStyleSheet (cp.styleBkgd) 
        self.tit_modes .setStyleSheet (cp.styleLabel)
        self.tit_forms .setStyleSheet (cp.styleLabel)
        self.tit_io    .setStyleSheet (cp.styleLabel)
        self.tit_status.setStyleSheet (cp.styleLabel)
        self.tit_modes .setAlignment  (QtCore.Qt.AlignCenter)
        self.tit_forms .setAlignment  (QtCore.Qt.AlignCenter)
        self.tit_io    .setAlignment  (QtCore.Qt.AlignCenter)
        self.tit_status.setAlignment  (QtCore.Qt.AlignCenter)

        self.lab_status.setFixedHeight(50)

        self.setButtonStyle()


    def setButtonStyle(self):
        for but in self.list_of_buts :
            but_text = str(but.text()) 
            if   but_text == self.fig.my_mode  : but.setStyleSheet (cp.styleButtonGood)
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
        self.set_wedges    .disconnect_objs()    
        self.set_polygons  .disconnect_objs()    
        #self.set_centers   .disconnect_objs()    


    def connect_all(self):
        self.set_lines     .connect_objs()    
        self.set_rectangles.connect_objs()    
        self.set_circles   .connect_objs()    
        self.set_wedges    .connect_objs()    
        self.set_polygons  .connect_objs()    
        #self.set_centers   .connect_objs()    


    def disconnect_form(self, form='Line'):
        if   form == 'Line'      : self.set_lines     .disconnect_objs()    
        elif form == 'Rectangle' : self.set_rectangles.disconnect_objs()    
        elif form == 'Circle'    : self.set_circles   .disconnect_objs()    
        elif form == 'Wedge'     : self.set_wedges    .disconnect_objs()    
        elif form == 'Polygon'   : self.set_polygons  .disconnect_objs()    
        #elif form == 'Center'    : self.set_centers   .disconnect_objs()    
        else                     : pass


    def connect_form(self, form='Line'):
        if   form == 'Line'      : self.set_lines     .connect_objs()    
        elif form == 'Rectangle' : self.set_rectangles.connect_objs()    
        elif form == 'Circle'    : self.set_circles   .connect_objs()    
        elif form == 'Wedge'     : self.set_wedges    .connect_objs()    
        elif form == 'Polygon'   : self.set_polygons  .connect_objs()    
        #elif form == 'Center'    : self.set_centers   .connect_objs()    
        else                     : pass


    def add_obj(self, str_of_pars) :
        """Add object when load the forms from file"""
        obj_type = str_of_pars.split(' ',1)[0]
        if   obj_type == 'Line'      : self.set_lines     .add_obj_for_str_of_pars(str_of_pars)    
        elif obj_type == 'Rectangle' : self.set_rectangles.add_obj_for_str_of_pars(str_of_pars)    
        elif obj_type == 'Circle'    : self.set_circles   .add_obj_for_str_of_pars(str_of_pars)    
        elif obj_type == 'Wedge'     : self.set_wedges    .add_obj_for_str_of_pars(str_of_pars)    
        elif obj_type == 'Polygon'   : self.set_polygons  .add_obj_for_str_of_pars(str_of_pars)
        #elif obj_type == 'Center'    : self.set_centers   .add_obj_for_str_of_pars(str_of_pars)    
        else                         : pass


    #def on_but_old(self):
    def on_but(self):
        but = self.get_pushed_but()
        msg = 'on_but ' + str(but.text())
        logger.debug(msg, __name__ )
        #if self.verb : print msg
        but_text = str(but.text())

        #Set MODE
        if but_text in self.list_of_modes :
            self.fig.my_mode  = but_text # Sets mode for Drag objects

            if self.fig.my_mode == 'Zoom' :
                self.widgimage.connectZoomMode()
                self.disconnect_all()
                self.current_form = None
            else :
                self.widgimage.disconnectZoomMode()

        #Set FORM
        if but_text in self.list_of_forms :
            self.disconnect_all()
            self.current_form = but_text
            self.connect_form(self.current_form)    # Connect objects for NEW form

        self.setButtonStyle()
        self.setStatus()



    def on_but_new(self):
    #def on_but(self):
        but = self.get_pushed_but()
        msg = 'on_but ' + str(but.text())
        logger.debug(msg, __name__ )
        #if self.verb : print msg

        #Set MODE
        but_text = str(but.text())
        if but_text in self.list_of_modes :
            self.fig.my_mode  = but_text # Sets mode for Drag objects

            if self.fig.my_mode == 'Zoom' :
                self.widgimage.connectZoomMode()
                self.disconnect_all()
                self.current_form = None

            elif self.fig.my_mode == 'Add' :
                self.widgimage.disconnectZoomMode()
                self.disconnect_all()
                #self.current_form = None

            else :
                self.widgimage.disconnectZoomMode()
                self.connect_all()
                self.current_form = None

        #Set FORM
        if but_text in self.list_of_forms :
            if self.current_form is None : self.disconnect_all()
            else : self.disconnect_form(self.current_form) # Disconnect objects for OLD form
            self.current_form = but_text
            self.connect_form(self.current_form)    # Connect objects for NEW form

        self.setButtonStyle()
        self.setStatus()


    def get_pushed_io_but(self):
        for ind, but in enumerate(self.list_of_io_buts) :
            if but.hasFocus() : return but, ind, self.list_of_fnames[ind]


    def on_io_but(self):
        but, ind, fname = self.get_pushed_io_but()
        but_text = str(but.text())
        msg = but_text + ', default file name: ' + str(fname) 
        logger.debug(msg, __name__ )
        if self.verb : print msg

        if fname is None : path0 = '.'
        else             : path0 = fname


        if but_text == self.list_of_io_tits[0] : # 'Load Img'
            if self.fexmod :
                self.exchange_image_load()
                return
            
            self.setStatus(1, 'Waiting\nfor input...')
            path = gu.get_open_fname_through_dialog_box(self, path0, but_text, filter='*.txt *.npy')
            if path is None :
                self.setStatus()
                return
            self.setStatus(2, 'WAIT!\nLoad image')
            arr = gu.get_array_from_file(path)             
            self.parent.set_image_array_new(arr, title='Image from '+path )
            self.setStatus(0, 'Image \nloaded')


        if but_text == self.list_of_io_tits[1] : # 'Load Forms'
            self.setStatus(1, 'Waiting\nfor input')
            path = gu.get_open_fname_through_dialog_box(self, path0, but_text, filter='*.txt')
            if path is None :
                self.setStatus()
                return
            msg='Load shaping-objects for mask from file: ' + path 
            logger.debug(msg, __name__ )
            if self.verb : print msg
            #text = gu.get_text_file_content(path)
            self.setStatus(2,'WAIT\nLoad forms')
            f=open(path,'r')
            for str_of_pars in f :
                self.add_obj(str_of_pars.rstrip('\n'))
            f.close() 
            self.setStatus(0, 'Forms\nloaded')


        if but_text == self.list_of_io_tits[2] : # 'Save Forms'
            #self.parent.set_image_array_new( get_array2d_for_test(), title='New array' )
            if self.list_of_objs_for_mask_is_empty() : return
            self.setStatus(1, 'Waiting\nfor input...')
            path = gu.get_save_fname_through_dialog_box(self, path0, but_text, filter='*.txt')
            if path is None :
                self.setStatus()
                return
            msg='Save shaping-objects for mask in file: ' + path 
            logger.debug(msg, __name__ )
            self.setStatus(2, 'WAIT!\nSave forms')
            f=open(path,'w')
            for obj in self.get_list_of_objs_for_mask() :
                str_of_pars = obj.get_str_of_pars()
                if self.verb : print str_of_pars
                f.write(str_of_pars + '\n')
            f.close() 
            self.setStatus(0, 'Forms\nsaved')


        if but_text == self.list_of_io_tits[3] : # 'Save Mask'
            if self.list_of_objs_for_mask_is_empty() :
                print 'WARNING: Empty mask is NOT saved!'
                self.setStatus(2, 'Empty mask\nNOT saved!')
                return
            self.setStatus(2, 'WAIT!\nMask is\nprocessing')
            self.enforceStatusRepaint()
            #print 'WAIT for mask processing'
            mask_total = self.get_mask_total()
            self.parent.set_image_array_new(mask_total, title='Mask')

            if self.fexmod :
                self.exchange_mask_save(mask_total)
                return            

            self.setStatus(1, 'Waiting\nfor input...')
            path = gu.get_save_fname_through_dialog_box(self, path0, but_text, filter='*.txt *.npy')
            if path is None :
                self.setStatus()
                return

            ext = os.path.splitext(path)[1]
            if ext == '.npy' : np.save(path, mask_total)
            else             : np.savetxt(path, mask_total, fmt='%1i', delimiter=' ')
            self.setStatus(0, 'Mask\nis saved')


        if but_text == self.list_of_io_tits[4] : # 'Save Inv-Mask'
            if self.list_of_objs_for_mask_is_empty() : 
                print 'WARNING: Empty mask is NOT saved!'
                self.setStatus(2, 'Empty mask\nNOT saved!')
                return
            self.setStatus(2, 'Wait!\nInv-mask is\nprocessing')
            self.enforceStatusRepaint()
            mask_total = ~self.get_mask_total()
            self.parent.set_image_array_new(mask_total, title='Inverse Mask')

            if self.fexmod :
                self.exchange_mask_save(mask_total)
                return            

            self.setStatus(1, 'Waiting\nfor input')
            path = gu.get_save_fname_through_dialog_box(self, path0, but_text, filter='*.txt *.npy')
            if path is None : 
                self.setStatus()
                return
            
            ext = os.path.splitext(path)[1]
            if ext == '.npy' : np.save(path, mask_total)
            else             : np.savetxt(path, mask_total, fmt='%1i', delimiter=' ')
            self.setStatus(0, 'Mask\nis ready')


        if but_text == self.list_of_io_tits[5] : # 'Print Forms'
            #self.parent.set_image_array_new( get_array2d_for_test(), title='New array' )

            msg = '\nForm parameters for composition of the mask'
            if self.verb : print msg
            logger.info(msg, __name__ )

            for obj in self.get_list_of_objs_for_mask() :
                str_of_pars = obj.get_str_of_pars()
                if self.verb : print str_of_pars
                logger.info(str_of_pars)


        if but_text == self.list_of_io_tits[6] : # 'Clear Forms'
            self.setStatus(2, 'WAIT!\nremoving\nforms')
            self.set_lines     .remove_all_objs_from_img_by_call()    
            self.set_rectangles.remove_all_objs_from_img_by_call()        
            self.set_circles   .remove_all_objs_from_img_by_call()        
            self.set_wedges    .remove_all_objs_from_img_by_call()        
            self.set_polygons  .remove_all_objs_from_img_by_call()    
            self.setStatus(0, 'Forms\nremoved')


    def exchange_image_load(self):       
        if self.afe_rd.is_new_arr_available() :
            self.setStatus(2, 'WAIT!\nLoad image')
            arr = self.afe_rd.get_arr_latest()             
            self.parent.set_image_array_new(arr, title='Image from %s...' % self.ifname )
            self.setStatus(0, 'Image is\nloaded')
        else :
            self.setStatus(1, 'New image\nis N/A !')
            return

  
    def exchange_mask_save(self, mask):       
        self.setStatus(2, 'WAIT!\nSave mask')
        self.afe_wr.save_arr(mask)
        self.setStatus(0, 'Mask\nis saved')


    def get_mask_total(self):       
        shape = self.widgimage.get_img_shape()
        if self.verb : print 'get_img_shape():', shape

        self.mask_total = None
        for i, obj in enumerate(self.get_list_of_objs_for_mask()) :
            if obj.isSelected : continue # Loop over ROI-type objects
            if self.mask_total is None : self.mask_total = obj.get_obj_mask(shape)
            else                       : self.mask_total = np.logical_or(self.mask_total, obj.get_obj_mask(shape))
            msg = 'mask for ROI-type object %i is ready...' % (i)
            logger.info(msg, __name__ )
            
        for i, obj in enumerate(self.get_list_of_objs_for_mask()) :
            if not obj.isSelected : continue # Loop over inversed objects
            if self.mask_total is None : self.mask_total = obj.get_obj_mask(shape)
            else                       : self.mask_total = np.logical_and(self.mask_total, obj.get_obj_mask(shape))
            msg = 'mask for inversed object %i is ready...' % (i)            
            logger.info(msg, __name__ )

        if self.y_is_flip : self.mask_total = np.flipud(gu.arr_rot_n90(self.mask_total, self.ccd_rot_n90))
        else :
            rot_ang = self.ccd_rot_n90
            if self.ccd_rot_n90 ==  90 : rot_ang=270
            if self.ccd_rot_n90 == 270 : rot_ang=90
            self.mask_total = gu.arr_rot_n90(self.mask_total, rot_ang)

        return self.mask_total


    def list_of_objs_for_mask_is_empty(self):       
        if len(self.get_list_of_objs_for_mask()) == 0 :
            logger.warning('List of objects for mask IS EMPTY!', __name__ )            
            return True
        else :
            return False


    def get_list_of_objs_for_mask(self):       
        return self.set_rectangles.get_list_of_objs() \
              +self.set_wedges    .get_list_of_objs() \
              +self.set_circles   .get_list_of_objs() \
              +self.set_lines     .get_list_of_objs() \
              +self.set_polygons  .get_list_of_objs()
            #+self.set_centers


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


    def setStatus(self, ind=None, msg='') :

        if ind is not None and msg == '' :
            self.stat_msg = self.fig.my_mode
            if self.current_form is not None : self.stat_msg += '\n' + str(self.current_form)
            self.stat_ind = 0

            if self.fig.my_mode == 'Zoom' and self.current_form is not None :
                self.stat_ind = 2
                self.stat_msg = 'What to do\nwith\n' + self.current_form + '?'

            elif self.fig.my_mode != 'Zoom' and self.current_form is None :
                self.stat_ind = 2
                self.stat_msg = self.fig.my_mode + '\nwhat form?'

        else :
            self.stat_ind = ind
            self.stat_msg = msg

        #self.lab_status.clear()
        if   self.stat_ind == 0 : self.lab_status.setStyleSheet(cp.styleButtonGood) # cp.styleStatusGood)
        elif self.stat_ind == 1 : self.lab_status.setStyleSheet(cp.styleButtonWarning) # cp.styleStatusWarning)
        elif self.stat_ind == 2 : self.lab_status.setStyleSheet(cp.styleButtonBad) # cp.styleStatusAlarm)
        self.lab_status.setText(self.stat_msg)


    def enforceStatusRepaint(self) :
        self.lab_status.repaint()
        self.repaint()

        #time.sleep(1)
        #print 'ind :', ind
        #print 'msg :', self.lab_status.text()


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
    sys.exit ('Exit test')

#-----------------------------
