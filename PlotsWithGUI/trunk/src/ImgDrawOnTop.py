#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgDrawOnTop...
#
#------------------------------------------------------------------------

"""Additional graphics on top of 2-d image

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

import DragCircle          as dragc
import DragRectangle       as dragr
#import ImgConfigParameters as icp

#---------------------
#  Class definition --
#---------------------

class ImgDrawOnTop :
    """Additional graphics on top of 2-d image"""


    def __init__(self, icp=None):
        self.icp               = icp
        self.icp.idrawontop    = self

        self.rectsFromInputAreCreated = False

        #self.icp.list_of_rects = [] # moved to icp
        print 'ImgDrawOnTop : init' 


    def get_control(self) :
        return self.icp.control


    def get_config_pars(self) :
        return self.icp


    def get_wimg(self) :
        return self.icp.wimg


    def get_fig(self) :
        return self.icp.wimg.fig


    def get_axes(self) :
        return self.icp.wimg.fig.myaxesSIm


    def get_idrawout( self ) :
        return self.icp.idrawout


    def on_mouse_press(self, event) : # is called from ImgControl
        print 'ImgDrawOnTop : on_mouse_press(...), mode=', self.get_fig().my_mode
        #if self.get_fig().my_mode  == 'Add' :
        if self.icp.modeCurrent == self.icp.modeAdd :
            if event.button != 1 : return     # if other than Left mouse button

            if self.icp.formCurrent == self.icp.formRect :
                obj = dragr.DragRectangle()

            dragr.add_obj_to_axes(obj, self.get_axes(), self.icp.list_of_rects)
            obj.on_press(event)               # Initialize object by the mouse drag

            obj.myType  = self.icp.typeCurrent              # set attribute
            obj.myIndex = self.icp.list_of_rects.index(obj) # index in the list


    def on_mouse_release(self, event) : # is called from ImgControl
        """ImgDrawOnTop : on_mouse_release(...) is called from ImgControl
        """
        self.update_list_of_objs()


    def add_rect(self, type=None, selected=False, xy=None, width=1, height=1):
        print 'ImgDrawOnTop : add_rect(...) from program call'
        obj = dragr.DragRectangle(xy=xy, width=width, height=height, color='r') 
        dragr.add_obj_to_axes(obj, self.get_axes(), self.icp.list_of_rects)
        obj.myType     = type
        obj.isSelected = selected
        obj.myIndex=self.icp.list_of_rects.index(obj) # index in the list
        print 'obj.myIndex=', obj.myIndex





#???????????????????????????????????????????

    def redraw_rect(self, obj):
        obj.disconnect() # disconnect object from canvas signals
        #obj.remove()     # remove object from axes
        if obj in self.icp.list_of_rects :
            self.icp.list_of_rects.remove(obj)
        #obj.add_to_axes(self.get_fig().myaxesSIm)
        #obj.connect()
        #obj.set_select_deselect_color()
        dragr.add_obj_to_axes(obj, self.get_axes(), self.icp.list_of_rects)
        obj.myIndex=self.icp.list_of_rects.index(obj) # index in the list
        print 'B) obj.myIndex=', obj.myIndex


    def set_all_objs_need_in_redraw(self, list_of_obj):
        for obj in list_of_obj :
            obj.needsInRedraw = True


    def none_rect(self):
        pass # self.update_list_of_rects()


    def overlay_rect(self):
        pass


    def select_rect(self):
        pass


    def remove_rect(self):
        pass


    def send_signal_and_remove_object_from_list(self, obj, list_of_objs, remove_type) :
        print 'Object is removing from the list. ACT HERE !!!, remove_type=', remove_type
        obj.print_pars()
        #====================== REMOVE OBJECT =================================================
        # THIS IS A PLACE TO REMOVE EVERYTHING ASSOCIATED WITH OBJECT AFTER CLICK OR PROGRAM CALL
        # SIGNAL ABOUT REMOVAL SHOULD BE SENT FROM HERE
        if remove_type == 'Call' : obj.remove_object_from_img() # <<<========= PROGRAM CALL TO REMOVE THE OBJECT FROM IMG
        self.get_control().signal_obj_will_be_removed(obj)      # <<<========= SIGNAL FOR ImgControl
        list_of_objs.remove(obj)                                # <<<========= REMOVE OBJECT FROM THE LIST
        #======================================================================================


    def update_list_of_objs(self):
        self.update_list_of_rects()
        #self.update_list_of_lines()
        #self.update_list_of_circles()


    def update_list_of_rects(self):
        initial_list_of_objs = list(self.icp.list_of_rects) # COPY list
        for obj in initial_list_of_objs :
            #print 'update_list_of_rects: obj.isRemoved=', obj.isRemoved
            if obj.isRemoved :
                #self.icp.list_of_rects.remove(obj)
                self.send_signal_and_remove_object_from_list(obj, self.icp.list_of_rects, 'Click')


    def draw_rects(self):

        if self.rectsFromInputAreCreated :
            initial_list_of_rects = list(self.icp.list_of_rects)
            for obj in initial_list_of_rects :
                self.redraw_rect(obj)
        else:
            for rectPars in self.icp.listOfRectInputParameters :
                #print rectPars
                t,s,x,y,w,h = rectPars
                if t == self.icp.typeCurrent :
                    self.add_rect(type=t, selected=s, xy=(x,y), width=w, height=h)

        self.rectsFromInputAreCreated = True


    def draw_on_top(self):
        print 'draw_on_top()'
        self.draw_rects()

#-----------------------------
# is called from ImgFigureManager.

    def remove_object(self, obj):
        print 'ImgDrawOnTop : remove_object'
        if obj in self.icp.list_of_rects :
            obj.remove_object_from_img()
            self.icp.list_of_rects.remove(obj) 
            self.icp.control.set_signal_info(mode=self.icp.modeNone)

#-----------------------------
# Test
#

def main():
    w = ImgDrawOnTop(None)

#-----------------------------

if __name__ == "__main__" :
    main()

#-----------------------------
