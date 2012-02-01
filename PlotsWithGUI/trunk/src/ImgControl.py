#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgControl...
#
#------------------------------------------------------------------------

"""This class contains methods for transmission of control signals between GUI and presenter

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id:$

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

import ImgConfigParameters as gicp
import ImgFigureManager    as imgfm

#---------------------
#  Class definition --
#---------------------
class ImgControl :
    """This class contains methods for transmission of control signals between GUI and presenter 
    """

    def __init__ ( self ) :
        self.icp         = gicp.giconfpars.addImgConfigPars( self )
        self.icp.control = self

    def get_icp( self ) :
        return self.icp
        #return gicp.giconfpars.getImgConfigPars( self )

    def get_wimg( self ) :
        return self.icp.wimg

    def get_wgui( self ) :
        return self.icp.wgui

    def get_idrawout( self ) :
        return self.icp.idrawout


    def get_idrawontop( self ) :
        return self.icp.idrawontop


#---------------------------------------
# Signals from ImgGUISpectrum.py
    def signal_to_control( self, form, mode ) :
        self.icp.modeCurrent = mode  # None, Add, Select, Remove, etc
        self.icp.formCurrent = form  # None, Line, Rect, Circle, etc. 
        #self.icp.typeCurrent        # None, Spectrum, Profile, ProjX, Zoom, etc.; Is already set.
        self.set_signal_info()

        # HERE WE NEED TO SET MODE ONLY
        #if form == self.icp.formRect :
        #    if mode == self.icp.modeAdd:     self.get_wimg().    add_rect( self.icp.typeCurrent )
        #    if mode == self.icp.modeSelect:  self.get_wimg(). select_rect()
        #    if mode == self.icp.modeOverlay: self.get_wimg().overlay_rect()
        #    if mode == self.icp.modeNone:    self.get_wimg().   none_rect()
        #    if mode == self.icp.modeRemove:  self.get_wimg(). remove_rect()

    def set_signal_info( self, mode=None ) :
        if mode != None : self.icp.modeCurrent = mode
        self.get_wimg().fig.my_mode = self.icp.modeCurrent # Is used in Drag.py
        self.print_signal_info()        

    def print_signal_info( self ) :
        print 'Current signals: type, form, mode :', self.icp.typeCurrent, self.icp.formCurrent, self.icp.modeCurrent

#---------------------------------------

    def signal_draw( self ) :
        print 'signal_draw, type=', self.icp.typeCurrent

#---------------------------------------
# Signals from ImgGUIPlayer.py

    def signal_get_event_previous( self ) :
        self.reset_event_for_test()
        
    def signal_get_event_current( self ) :
        self.reset_event_for_test()

    def signal_get_event_next( self ) :
        self.reset_event_for_test()

    def reset_event_for_test( self ) :
        self.set_image_array( self.get_array2d_for_test() )  # on_draw() is called automatically

    def signal_grid_onoff(self):
        #self.icp.gridIsOn = self.cbox_onoff.isChecked()
        self.get_wimg().grid_onoff(self.icp.gridIsOn)

    def signal_save( self ) :
        print 'signal_save'
        self.icp.saveImgConfigPars()

    def signal_print( self ) :
        print 'signal_print'
        self.icp.printImgConfigPars()

    def signal_quit( self ) :
        print 'signal_quit'
        imgfm.ifm.close_fig()
        self.close()

#---------------------------------------
# Signals from ImgWidget which require immediate actions


    def signal_mouse_on_image_press( self, event ) :
        """Signal receiver from ImgWidget"""
        #print 'signal_mouse_press'
        self.get_idrawontop().on_mouse_press(event) 


    def signal_mouse_on_image_release( self, event ) :
        """Signal receiver from ImgWidget"""
        #print 'signal_mouse_release'
        self.get_idrawontop().on_mouse_release(event) 
        self.get_idrawout().draw_outside()

#---------------------------------------
# Signals from ImgDrawOnTop

    def signal_obj_will_be_removed(self, obj) :
        print 'ImgControl : signal_obj_will_be_removed(...)'
        obj.print_pars()

#---------------------------------------
