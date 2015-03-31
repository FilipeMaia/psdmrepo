
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIMaskEditor...
#
#------------------------------------------------------------------------

"""Renders the main GUI for the CalibManager.

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
import numpy as np
from PSCalib.GeometryAccess import GeometryAccess, img_from_pixel_arrays


# For self-run debugging:
if __name__ == "__main__" :
    import matplotlib
    matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)

#import matplotlib
#if matplotlib.get_backend() != 'Qt4Agg' : matplotlib.use('Qt4Agg')

from PyQt4 import QtGui, QtCore
#import time   # for sleep(sec)

#-----------------------------
# Imports for other modules --
#-----------------------------

from CalibManager.Frame                  import Frame
from CalibManager.ConfigParametersForApp import cp
from CalibManager.Logger                 import logger
import CalibManager.GlobalUtils          as     gu
from CalibManager.PlotImgSpe             import *
from CalibManager.GUIFileBrowser         import *
from CalibManager.FileNameManager        import fnm

from CorAna.MaskEditor import MaskEditor

#---------------------
#  Class definition --
#---------------------
#class GUIMaskEditor ( QtGui.QWidget ) :
class GUIMaskEditor ( Frame ) : 
    """GUI for ROI mask processing.

    @see BaseClass
    @see OtherClass
    """
    def __init__ (self, parent=None, app=None) :

        self.name = 'GUIMaskEditor'
        self.myapp = app
        #QtGui.QWidget.__init__(self, parent)
        Frame.__init__(self, parent, mlw=1)


        self.fname_geometry         = cp.fname_geometry     
        self.fname_roi_img_nda      = cp.fname_roi_img_nda  
        self.fname_roi_img          = cp.fname_roi_img 
        self.fname_roi_mask_img     = cp.fname_roi_mask_img 
        self.fname_roi_mask_nda     = cp.fname_roi_mask_nda
        self.fname_roi_mask_nda_tst = cp.fname_roi_mask_nda_tst
        self.sensor_mask_cbits      = cp.sensor_mask_cbits


        self.img_arr = None

        #self.setFrame()

        cp.setIcons()

        self.setGeometry(10, 25, 800, 360)
        self.setWindowTitle('Mask Editor Control')
        #self.setWindowIcon(cp.icon_monitor)
        self.palette = QtGui.QPalette()
        self.resetColorIsSet = False


        self.lab_status = QtGui.QLabel('Status: ')
        self.lab_mask_cbits = QtGui.QLabel('Mask control bits:')

        self.but_geometry         = QtGui.QPushButton( ' 1. Select geometry file' )
        self.but_roi_img_nda      = QtGui.QPushButton( ' 2. Select ndarray for image file' )
        self.but_roi_img          = QtGui.QPushButton( 'Image' )
        self.but_reco_image       = QtGui.QPushButton( ' 3. Reconstruct image from ndarray')
        self.but_roi_mask_img     = QtGui.QPushButton( 'Mask image' )
        self.but_mask_editor      = QtGui.QPushButton( ' 4. Open Mask Editor')
        self.but_roi_mask_nda     = QtGui.QPushButton( 'Mask ndarray' )
        self.but_roi_convert      = QtGui.QPushButton( ' 5. Convert mask image to ndarray')

        self.but_plot             = QtGui.QPushButton( 'Plot')
        self.but_view             = QtGui.QPushButton( 'View')

        self.edi_geometry     = QtGui.QLineEdit ( self.fname_geometry.value() )
        self.edi_roi_img_nda  = QtGui.QLineEdit ( self.fname_roi_img_nda.value() )
        self.edi_roi_img      = QtGui.QLineEdit ( self.fname_roi_img.value() )
        self.edi_roi_mask_img = QtGui.QLineEdit ( self.fname_roi_mask_img.value() )
        self.edi_roi_mask_nda = QtGui.QLineEdit ( self.fname_roi_mask_nda.value() )
        self.edi_mask_cbits   = QtGui.QLineEdit ( str(self.sensor_mask_cbits.value()) )
        self.edi_mask_cbits.setValidator(QtGui.QDoubleValidator(0,0177777,3,self))
 
 
        self.grid = QtGui.QGridLayout()
        self.grid_row = 0
        self.grid.addWidget(self.but_geometry,      self.grid_row,     0, 1, 4)
        self.grid.addWidget(self.edi_geometry,      self.grid_row,     4, 1, 6)

        self.grid.addWidget(self.but_roi_img_nda,   self.grid_row+1,   0, 1, 4)
        self.grid.addWidget(self.edi_roi_img_nda,   self.grid_row+1,   4, 1, 6)

        self.grid.addWidget(self.but_reco_image,    self.grid_row+2,   0, 1, 4)
        self.grid.addWidget(self.edi_roi_img,       self.grid_row+2,   4, 1, 5)
        self.grid.addWidget(self.but_roi_img,       self.grid_row+2,   9)

        self.grid.addWidget(self.but_mask_editor,   self.grid_row+3,   0, 1, 4)
        self.grid.addWidget(self.edi_roi_mask_img,  self.grid_row+3,   4, 1, 5)
        self.grid.addWidget(self.but_roi_mask_img,  self.grid_row+3,   9)

        self.grid.addWidget(self.but_roi_convert,   self.grid_row+4,   0, 1, 4)
        self.grid.addWidget(self.edi_roi_mask_nda,  self.grid_row+4,   4, 1, 5)
        self.grid.addWidget(self.but_roi_mask_nda,  self.grid_row+4,   9)

        self.grid.addWidget(self.lab_mask_cbits,    self.grid_row+8,   0)
        self.grid.addWidget(self.edi_mask_cbits,    self.grid_row+8,   1)
        self.grid.addWidget(self.but_plot,          self.grid_row+8,   8)
        self.grid.addWidget(self.but_view,          self.grid_row+8,   9)

        self.hboxS = QtGui.QHBoxLayout()
        self.hboxS.addWidget(self.lab_status)
        self.hboxS.addStretch(1)     

        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addLayout(self.grid)
        self.vbox.addStretch(1)
        self.vbox.addLayout( self.hboxS ) 
        self.setLayout(self.vbox)

        self.connect( self.edi_mask_cbits, QtCore.SIGNAL('editingFinished()'), self.on_edi_mask_cbits )
        self.connect( self.but_geometry    ,     QtCore.SIGNAL('clicked()'), self.on_but_geometry     ) 
        self.connect( self.but_roi_img_nda ,     QtCore.SIGNAL('clicked()'), self.on_but_roi_img_nda  ) 
        self.connect( self.but_roi_img     ,     QtCore.SIGNAL('clicked()'), self.on_but_roi_img      ) 
        self.connect( self.but_roi_mask_img,     QtCore.SIGNAL('clicked()'), self.on_but_roi_mask_img ) 
        self.connect( self.but_roi_mask_nda,     QtCore.SIGNAL('clicked()'), self.on_but_roi_mask_nda ) 
        self.connect( self.but_reco_image  ,     QtCore.SIGNAL('clicked()'), self.on_but_reco_image   ) 
        self.connect( self.but_mask_editor ,     QtCore.SIGNAL('clicked()'), self.on_but_mask_editor  ) 
        self.connect( self.but_roi_convert ,     QtCore.SIGNAL('clicked()'), self.on_but_roi_convert  ) 
        self.connect( self.but_plot        ,     QtCore.SIGNAL('clicked()'), self.on_but_plot         ) 
        self.connect( self.but_view        ,     QtCore.SIGNAL('clicked()'), self.on_but_view         ) 
 
        self.showToolTips()
        self.setStyle()

        self.setStatus(0)

        cp.guimaskeditor = self
        self.move(10,25)
        
        #print 'End of init'
        
    #-------------------
    # Private methods --
    #-------------------

    def showToolTips(self):
        #pass
        self.edi_roi_img.setToolTip('Path to the file with image data') 
        self.but_roi_img.setToolTip('Open file browser dialog window \nand select the file with image data') 
        self.but_mask_editor.setToolTip('Open/Close Mask Editor window')


#    def setFrame(self):
#        self.frame = QtGui.QFrame(self)
#        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken ) #Box, Panel | Sunken, Raised 
#        self.frame.setLineWidth(0)
#        self.frame.setMidLineWidth(1)
#        self.frame.setGeometry(self.rect())
#        #self.frame.setVisible(False)


    def setStyle(self):

        #self.setMinimumSize(725,360)
        self.setFixedSize(750,360)
        #self.setMaximumWidth(800)
        #self.lab_status.setMinimumWidth(600) 

        self.                setStyleSheet(cp.styleBkgd)
        #self.but_mask_editor.setStyleSheet(cp.styleButton)
        #self.but_mask_editor.setFixedWidth(200)
        #self.but_mask_editor.setMinimumHeight(60)
        #self.but_mask_editor.setMinimumSize(180,40)
        #self.but_roi_convert.setMinimumSize(180,40)
        #self.but_reco_image .setMinimumSize(180,40)

        self.but_geometry    .setStyleSheet(cp.styleButtonLeft)
        self.but_roi_img_nda .setStyleSheet(cp.styleButtonLeft)
        self.but_roi_img     .setStyleSheet(cp.styleButton)
        self.but_roi_mask_img.setStyleSheet(cp.styleButton)
        self.but_roi_mask_nda.setStyleSheet(cp.styleButton)
        self.but_reco_image  .setStyleSheet(cp.styleButtonLeft)
        self.but_mask_editor .setStyleSheet(cp.styleButtonLeft)
        self.but_roi_convert .setStyleSheet(cp.styleButtonLeft)

        #self.but_plot        .setFixedWidth(100)    
        #self.but_view        .setFixedWidth(100)    
        self.but_plot        .setIcon(cp.icon_monitor)
        self.but_view        .setIcon(cp.icon_table) # cp.icon_logviewer)

        #self.edi_roi_img.setFixedWidth(400)

        self.edi_geometry    .setReadOnly(True)
        self.edi_roi_img_nda .setReadOnly(True)
        self.edi_roi_img     .setReadOnly(True)
        self.edi_roi_mask_img.setReadOnly(True)
        self.edi_roi_mask_nda.setReadOnly(True)

        self.edi_geometry    .setEnabled(False)
        self.edi_roi_img_nda .setEnabled(False)
        self.edi_roi_img     .setEnabled(False)
        self.edi_roi_mask_img.setEnabled(False)
        self.edi_roi_mask_nda.setEnabled(False)

        self.edi_mask_cbits.setFixedWidth(60)   

        self.lab_mask_cbits.setStyleSheet(cp.styleLabel) 
        
        #self.edi_geometry    .setStyleSheet(cp.styleEditInfo)
        #self.edi_roi_img_nda .setStyleSheet(cp.styleEditInfo)
        #self.edi_roi_img     .setStyleSheet(cp.styleEditInfo)
        #self.edi_roi_mask_img.setStyleSheet(cp.styleEditInfo)
        #self.edi_roi_mask_nda.setStyleSheet(cp.styleEditInfo)

        #self.butFBrowser.setVisible(False)
        #self.butSave.setText('')
        #self.butExit.setText('')
        #self.butExit.setFlat(True)


    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self.name) 
        #self.frame.setGeometry(self.rect())
        pass


    def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        pass


    def closeEvent(self, event):
        logger.debug('closeEvent', self.name)

        #try    : cp.guimain.close()
        #except : pass

        #try    : del cp.guimain
        #except : pass

        try    : cp.maskeditor.close()
        except : pass

        try    : cp.plotimgspe.close()
        except : pass

        cp.guimaskeditor = None


    def onExit(self):
        logger.debug('onExit', self.name)
        self.close()


    def on_but_geometry(self):
        logger.info('Select the "geometry" file', __name__)
        self.set_file_name(self.edi_geometry, self.fname_geometry, mode='open')


    def on_but_roi_img_nda(self):
        logger.info('Select file with ndarray for image', __name__)
        self.set_file_name(self.edi_roi_img_nda, self.fname_roi_img_nda, mode='open')


    def on_but_roi_img(self):
        logger.info('Set/select the file with image', __name__)
        #prefix = self.fname_prefix.value()
        #filter = 'Text files (' + prefix + '*.txt ' + prefix + '*.dat)\nAll files (*)'
        self.set_file_name(self.edi_roi_img, self.fname_roi_img) #, filter)


    def on_but_roi_mask_img(self):
        logger.info('Set/select the file for image of the mask', __name__)
        self.set_file_name(self.edi_roi_mask_img, self.fname_roi_mask_img)


    def on_but_roi_mask_nda(self):
        logger.info('Set/select the file for mask ndarray', __name__)
        self.set_file_name(self.edi_roi_mask_nda, self.fname_roi_mask_nda)


    def set_file_name(self, edi, par, mode='save', filter='*.txt *.npy *.dat *.data\nAll files (*)'):
        #logger.debug('set_file_name', __name__)

        self.setStatus(1, 'Waiting for input of the file name...')
        
        path = str( edi.displayText() )
        dname, fname = os.path.split(path)
        msg = 'Current dir: %s   file: %s' % (dname, fname)
        logger.info(msg, __name__)
        
        path = str( QtGui.QFileDialog.getSaveFileName(self, 'Save file', path, filter=filter) ) \
               if mode == 'save' else \
               str( QtGui.QFileDialog.getOpenFileName(self, 'Open file', path, filter=filter) )

        dname, fname = os.path.split(path)

        if dname == '' or fname == '' :
            logger.info('Input directiry name or file name is empty... use default values', __name__)
            return
        else :
            edi.setText(path)
            par.setValue(path)
            logger.info('Selected file:\n' + path, __name__)

        self.setStatus(0)


    def openFileWithImageArray(self) :
         path = self.fname_roi_img.value()
         self.img_arr = gu.get_image_array_from_file(path) # , dtype=np.float32)
         #print 'openFileWithImageArray: self.arr.shape:', self.img_arr.shape
         #print self.img_arr


    def dictOfMaskEditorPars (self):       
        pars = {'parent' : None, 
                'arr'    : self.img_arr, 
                'xyc'    : None, # xyc=(opts.xc,opts.yc)
                'ifname' : self.fname_roi_img.value(), 
                'ofname' : 'plot.png', 
                'mfname' : self.fname_roi_mask_img.value(),
                'title'  : 'Mask Editor', 
                'lw'     : 1, 
                'col'    : 'k',
                'picker' : 8,
                'verb'   : True,
                'ccd_rot': 0, 
                'updown' : False}

        #print 'Start MaskEditor with input parameters:'
        #for k,v in pars.items():
        #    print '%9s : %s' % (k,v)

        return pars


    def on_but_mask_editor  (self):       
        logger.debug('onLogger', self.name)
        self.setStatus(1, 'Work with mask editor... DO NOT FORGET TO SAVE MASK!!!')

        mfname = self.fname_roi_mask_img.value()

        try    :
            if cp.maskeditor is not None and not os.path.exists(mfname) :
                msg = 'File %s was not saved!\nContinue anyway?' % os.path.basename(mfname)
                logger.warning(msg, __name__)
                status = gu.confirm_or_cancel_dialog_box(parent=self, text=msg, title='Confirm or cancel')
                if not status : return

            cp.maskeditor.close()
            del cp.maskeditor
            cp.maskeditor = None
            self.but_mask_editor.setStyleSheet(cp.styleButtonLeft)
            self.but_mask_editor.setText(' 4. Open Mask Editor')

        except :
            self.but_mask_editor.setStyleSheet(cp.styleButtonGood)
            self.but_mask_editor.setText('Close Mask Editor')

            self.openFileWithImageArray()

            pars = self.dictOfMaskEditorPars ()
            cp.maskeditor = MaskEditor(**pars)
            cp.maskeditor.move(self.pos().__add__(QtCore.QPoint(820,-7))) # open window with offset w.r.t. parent
            cp.maskeditor.show()

        self.setStatus(0)


    def reco_image_from_ndarray (self, gfname, afname):

        #mcbits = self.sensor_mask_cbits.value() 

        msg = 'Reconstruct image from\n      geometry: %s\n     and ndarray: %s' % \
             ( gfname, afname )
        logger.info(msg, __name__)

        geometry = GeometryAccess(gfname, 0)
        iX, iY = geometry.get_pixel_coord_indexes()

        afext = '' if afname is None else os.path.splitext(afname)[1]

        nda = np.ones(iX.shape, dtype=np.uint16) if afname is None else \
              np.load(afname) if afext == '.npy' else \
              np.loadtxt(afname) #, dtype=np.uint16)
        nda.shape = iX.shape 

        #if mcbits :  nda *= geometry.get_pixel_mask(mbits=mcbits)
 
        return img_from_pixel_arrays(iX, iY, W=nda) 


    def on_but_reco_image (self):

        self.setStatus(1, 'Reconstruct image from ndarray')

        gfname = self.fname_geometry.value()
        afname = self.fname_roi_img_nda.value()
        ofname = self.fname_roi_img.value()

        msg = 'Reconstruct image from\n      geometry: %s\n         ndarray: %s\n and save it in: %s' % \
             ( gfname, afname, ofname )

        img = self.reco_image_from_ndarray(gfname, afname)

        ofext = '' if ofname is None else os.path.splitext(ofname)[1]

        if ofext == '.npy' : np.save(ofname, img)
        else               : np.savetxt(ofname, img, fmt='%d', delimiter=' ')

        msg = 'Image is saved in the file %s' % ofname
        logger.info(msg, __name__)

        self.setStatus(0)


    def on_but_roi_convert (self):

        self.setStatus(1, 'Convert image to ndarray')

        mcbits = self.sensor_mask_cbits.value() 
        gfname = self.fname_geometry.value()
        ifname = self.fname_roi_mask_img.value()
        ofname = self.fname_roi_mask_nda.value()
        tfname = self.fname_roi_mask_nda_tst.value()
        
        msg = '\n  Convert ROI mask image: %s\n      to ndarray: %s\n    using geometry: %s' % \
             ( ifname, ofname, gfname )
        logger.info(msg, __name__)

        geometry = GeometryAccess(gfname, 0)
        iX, iY = geometry.get_pixel_coord_indexes()
        msg = 'Pixel index array iX, iY shapes: %s,  %s' % (str(iX.shape), str(iY.shape))
        logger.info(msg, __name__)

        ifext = os.path.splitext(ifname)[1]
        ofext = os.path.splitext(ofname)[1]

        mask_roi = np.load(ifname) if ifext == '.npy' else np.loadtxt(ifname, dtype=np.uint16)

        mask_nda = np.array( [mask_roi[r,c] for r,c in zip(iX, iY)] ) # 155 msec
        if mcbits : mask_nda *= geometry.get_pixel_mask(mbits=mcbits)

        img_mask_test = img_from_pixel_arrays(iX, iY, W=mask_nda) 

        if ofext == '.npy' : np.save(ofname, mask_nda)
        else               :
            mask_nda.shape = [iX.size/iX.shape[-1],iX.shape[-1]]
            logger.info('Mask ndarray is re-shape for saving in txt to 2-d: %s' % str(mask_nda.shape),  __name__) 
            np.savetxt(ofname, mask_nda, fmt='%d', delimiter=' ')

        logger.info('Mask ndarray is saved in the file %s' % ofname, __name__)

        self.setStatus(1, 'Test: reconstruct image from mask ndarray...')

        tfext = os.path.splitext(tfname)[1]
        
        if tfext == '.npy' : np.save(tfname, img_mask_test)
        else               : np.savetxt(tfname, img_mask_test, fmt='%d', delimiter=' ')
        logger.info('Test-image generated from mask ndarray is saved in file %s' % tfname, __name__)

        self.setStatus(0)


    def select_file_for_plot(self):

        list = [ self.fname_roi_img.value() \
               , self.fname_roi_mask_img.value() \
               , self.fname_roi_mask_nda.value() \
               , self.fname_roi_mask_nda_tst.value() \
                 ]
        fname = gu.selectFromListInPopupMenu(list)
        if fname is None :
            return None

        if not os.path.exists(fname) :
            logger.warning('File %s does not exist. There is nothing to plot...' % fname, __name__)
            return None

        logger.info('Selected file for plot: %s' % fname, __name__)

        return fname


    def on_but_plot(self):
        try :
            cp.plotimgspe.close()
            try    : del cp.plotimgspe
            except : pass

        except :

            ifname = self.select_file_for_plot()
            if ifname is None :
                return
            
            self.setStatus(1, 'Plot image from file %s' % os.path.basename(ifname))

            ofname = os.path.join(fnm.path_dir_work(),'image.png')
            tit = 'Plot for %s' % os.path.basename(ifname)            
            cp.plotimgspe = PlotImgSpe(None, ifname=ifname, ofname=ofname, title=tit, is_expanded=False)
            cp.plotimgspe.move(cp.guimain.pos().__add__(QtCore.QPoint(720,120)))
            cp.plotimgspe.show()

        self.setStatus(0)


    def on_but_view(self):
        logger.info('on_but_view', __name__)

        try    :
            cp.guifilebrowser.close()

        except :            
            list_of_fnames = [ self.fname_geometry.value() \
                             , self.fname_roi_img_nda.value() \
                             , self.fname_roi_img.value() \
                             , self.fname_roi_mask_img.value() \
                             , self.fname_roi_mask_nda.value() \
                             , self.fname_roi_mask_nda_tst.value() \
                               ] 

            cp.guifilebrowser = GUIFileBrowser(None, list_of_fnames, list_of_fnames[0])
            cp.guifilebrowser.move(self.pos().__add__(QtCore.QPoint(880,40))) # open window with offset w.r.t. parent
            cp.guifilebrowser.show()



    def on_edi_mask_cbits(self):
        str_value = str( self.edi_mask_cbits.displayText() )
        self.sensor_mask_cbits.setValue(float(str_value))  
        logger.info('Set sensor mask control bitword: %s' % str_value, __name__ )


    def setStatus(self, status_index=0, msg='Waiting for the next command'):
        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0 : self.lab_status.setStyleSheet(cp.styleStatusGood)
        if status_index == 1 : self.lab_status.setStyleSheet(cp.styleStatusWarning)
        if status_index == 2 : self.lab_status.setStyleSheet(cp.styleStatusAlarm)

        #self.lab_status.setText('Status: ' + list_of_states[status_index] + msg)
        self.lab_status.setText(msg)

#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    ex  = GUIMaskEditor()
    ex.show()
    app.exec_()
#-----------------------------
