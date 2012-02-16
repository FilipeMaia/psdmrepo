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


#---------------------------------------
# Signals from ImgGUIMode.py

    def signal_to_control( self, form, mode ) :
        self.icp.modeCurrent = mode  # None, Add, Move, Select, Remove, etc
        self.icp.formCurrent = form  # None, Line, Rect, Circle, etc. 
        #self.icp.typeCurrent        # None, Spectrum, Profile, ProjX, Zoom, etc.; Is already set.
        self.set_signal_info()

    def set_signal_info( self, mode=None ) :
        if mode != None : self.icp.modeCurrent = mode
        self.get_wimg().fig.my_mode = self.icp.modeCurrent # Is used in Drag.py
        self.print_signal_info()        

    def print_signal_info( self ) :
        print 'Current signals: type, form, mode :', self.icp.typeCurrent, self.icp.formCurrent, self.icp.modeCurrent

    def signal_draw( self ) :
        print 'signal_draw, type=', self.icp.typeCurrent

    def signal_grid_onoff(self):
        #self.icp.gridIsOn = self.cbox_onoff.isChecked()
        self.get_wimg().grid_onoff(self.icp.gridIsOn)

    def signal_log_onoff(self):
        print 'ImgControl : signal_log_onoff(...), log=', self.icp.logIsOn

#---------------------------------------
# Signals from ImgGUIPlayer.py

    def signal_get_event_previous( self ) :
        self.reset_event_for_test()
        
    def signal_get_event_current( self ) :
        self.reset_event_for_test()

    def signal_get_event_next( self ) :
        self.reset_event_for_test()

    def reset_event_for_test( self ) :
        #self.set_image_array( self.get_array2d_for_test() )  # on_draw() is called automatically
        self.set_image_array( self.get_array2d_with_ring_for_test() )  # on_draw() is called automatically
        self.get_idrawontop().set_all_objs_need_in_redraw()
        self.get_idrawout().draw_outside()

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
        self.get_idrawout().remove_outside_plot_for_obj(obj)


    def signal_center_is_reset_on_click(self) :
        print 'ImgControl : signal_center_is_reset_on_click(). New center x,y =', self.icp.x_center, self.icp.y_center
        # self.get_idrawontop().draw_wedgs() # DOES NOT WORK
        self.get_wgui().setTabBarCenter()
        self.get_idrawontop().update_list_of_objs( self.icp.list_of_wedgs )
        self.get_idrawontop().set_all_objs_need_in_redraw()
        self.get_wimg().on_draw()

#---------------------------------------
# Signals from ImgGUICenter.py
    def signal_center_is_reset_in_gui(self) :
        print 'ImgControl : signal_center_is_reset_in_gui(). New center x,y =', self.icp.x_center, self.icp.y_center
        self.get_idrawontop().set_center_position_from_icp()
        self.get_idrawontop().update_list_of_objs( self.icp.list_of_wedgs )
        self.get_idrawontop().set_all_objs_need_in_redraw()
        self.get_wimg().on_draw()
        
#---------------------------------------
# Signals from ImgDrawOutside

    def signal_and_close_fig(self, number) :
        print 'ImgControl : signal_and_close_fig(...), figure number =', number
        imgfm.ifm.close_fig(number)

#---------------------------------------
# Signals from ImgFigureManager
    def signal_outside_fig_is_closing_by_call(self, fig) :
        """This method will be called for the program request to close figure
        """
        print 'ImgControl : signal_outside_fig_is_closing_by_call(...), fig.number =', fig.number 
        #fig.my_object.print_pars()
        if self.icp.modeCurrent == self.icp.modeSelect : return
        if self.icp.modeCurrent != self.icp.modeRemove : return 
        #and fig.my_object.isSelected : return
        self.get_idrawontop().remove_object(fig.my_object) # Remove object from the main plot


    def signal_outside_fig_is_closing_by_click(self, fig) :
        """This method will be called for the click on X
        """
        print 'ImgControl : signal_outside_fig_is_closing_by_click(...), fig.number =', fig.number
        #fig.my_object.print_pars()
        self.get_idrawontop().remove_object(fig.my_object) # Remove object from the main plot


    def signal_figure_is_selected(self, fig) :
        """This method will be called when mouse click on figure.
           CLICK SHOULD BE INSIDE THE CANVAS REGION...
           DOES NOT WORK AT CLICK ON FRAME ....
        """
        print 'ImgControl : signal_figure_is_selected(...), fig number =', fig.number, ' for object:'
        obj = fig.my_object

        if obj != None :
            obj.print_pars()
           #self.get_idrawontop().select_deselect_object_by_call()   
            obj.select_deselect_object_by_call(color='w')   

#---------------------------------------
