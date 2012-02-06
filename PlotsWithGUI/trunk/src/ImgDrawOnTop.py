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

import Drag                as drag
import DragCircle          as dragc                             # <===== DEPENDS ON SHAPE
import DragRectangle       as dragr                             # <===== DEPENDS ON SHAPE
import DragLine            as dragl                             # <===== DEPENDS ON SHAPE

#---------------------
#  Class definition --
#---------------------

class ImgDrawOnTop :
    """Additional graphics on top of 2-d image"""

    def __init__(self, icp=None):
        self.icp            = icp
        self.icp.idrawontop = self

        self.rectsFromInputAreCreated = False                   # <===== DEPENDS ON SHAPE
        self.circsFromInputAreCreated = False                   # <===== DEPENDS ON SHAPE
        self.linesFromInputAreCreated = False                   # <===== DEPENDS ON SHAPE

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

            self.add_obj_on_click(event, dragr.DragRectangle, self.icp.formRect,   self.get_axes(), self.icp.list_of_rects) # <===== DEPENDS ON SHAPE
            self.add_obj_on_click(event, dragl.DragLine,      self.icp.formLine,   self.get_axes(), self.icp.list_of_lines) # <===== DEPENDS ON SHAPE
            self.add_obj_on_click(event, dragc.DragCircle,    self.icp.formCircle, self.get_axes(), self.icp.list_of_circs) # <===== DEPENDS ON SHAPE


    def on_mouse_release(self, event) : # is called from ImgControl
        """ImgDrawOnTop : on_mouse_release(...) is called from ImgControl
        """
        self.update_list_of_all_objs()


    def update_list_of_all_objs(self):
        self.update_list_of_objs( self.icp.list_of_rects )                   # <===== DEPENDS ON SHAPE
        self.update_list_of_objs( self.icp.list_of_lines )                   # <===== DEPENDS ON SHAPE
        self.update_list_of_objs( self.icp.list_of_circs )                   # <===== DEPENDS ON SHAPE


    def update_list_of_objs(self, list_of_objs):
        initial_list_of_objs = list(list_of_objs) # COPY list
        for obj in initial_list_of_objs :
            if obj.isRemoved :
                self.send_signal_and_remove_object_from_list(obj, list_of_objs, 'Click')


    def set_all_objs_need_in_redraw(self):
        #if self.icp.typeCurrent != self.icp.typeSpectrum : return
        drag.set_list_need_in_redraw(self.icp.list_of_rects)                 # <===== DEPENDS ON SHAPE
        drag.set_list_need_in_redraw(self.icp.list_of_lines)                 # <===== DEPENDS ON SHAPE
        drag.set_list_need_in_redraw(self.icp.list_of_circs)                 # <===== DEPENDS ON SHAPE



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


    def add_obj_on_click(self, event, DragObject, form, axes, list_of_objs) :
        """Creates a new rect object on mouse click
        """
        if self.icp.formCurrent != form : return                            # check that the form is correct 
        obj = DragObject()                                                  # create default object
        drag.add_obj_to_axes(obj, axes, list_of_objs)                       # add object to axes and in the list
        obj.on_press(event)                                                 # Initialize object by the mouse drag
        obj.myType     = self.icp.typeCurrent                               # set attribute
        obj.myIndex    = list_of_objs.index(obj)                            # set attribute
        obj.isSelected = False                                              # set attribute


    def add_obj_on_call(self, obj, axes, list_of_objs, type=None, selected=False) :  
        """Creates a new rect object at program call
        """
        drag.add_obj_to_axes(obj, self.get_axes(), list_of_objs)            # add object to axes and in the list
        obj.myType     = type                                               # set attribute
        obj.myIndex    = list_of_objs.index(obj)                            # set attribute
        obj.isSelected = selected                                           # set attribute


    def draw_rects(self) :                                     # <===== DEPENDS ON SHAPE INSIDE
        """Draw rects on top of the main image plot
        """
        axes         = self.get_axes()
        list_of_objs = self.icp.list_of_rects

        if not self.rectsFromInputAreCreated :
            self.rectsFromInputAreCreated = True
            for objPars in self.icp.listOfRectInputParameters :
                #print objPars
                t,s,x,y,w,h,lw,col = objPars
                if t == self.icp.typeCurrent :
                    obj = dragr.DragRectangle(xy=(x,y), width=w, height=h, color='r')
                    self.add_obj_on_call(obj, axes, list_of_objs, type=t, selected=s)
        else:
            drag.redraw_objs_from_list(axes, list_of_objs)


    def draw_lines(self) :                                      # <===== DEPENDS ON SHAPE INSIDE
        """Draw lines on top of the main image plot
        """
        axes         = self.get_axes()
        list_of_objs = self.icp.list_of_lines

        if not self.linesFromInputAreCreated :
            self.linesFromInputAreCreated = True
#            for objPars in self.icp.listOfLineInputParameters :
#                #print objPars
#                t,s,x1,x2,y1,y2,w,h = objPars
#                if t == self.icp.typeCurrent :
#                    obj = DragLine((x1,x2), (y1,y2), linewidth=2, color='r')
#                    self.add_obj_on_call(obj, axes, list_of_objs, type=t, selected=s)
        else:
            drag.redraw_objs_from_list(axes, list_of_objs)




    def draw_circs(self) :                                      # <===== DEPENDS ON SHAPE INSIDE
        """Draw circs on top of the main image plot
        """
        axes         = self.get_axes()
        list_of_objs = self.icp.list_of_circs

        if not self.circsFromInputAreCreated :
            self.circsFromInputAreCreated = True
#            for objPars in self.icp.listOfCircleInputParameters :
#                #print objPars
#                t,s,x0,y0,r = objPars
#                if t == self.icp.typeCurrent :
#                    obj = DragCircle((x0, y0), radius=r, color='r')
#                    self.add_obj_on_call(obj, axes, list_of_objs, type=t, selected=s)
        else:
            drag.redraw_objs_from_list(axes, list_of_objs)



    def draw_on_top(self):
        print 'draw_on_top()'
        self.draw_rects()                                                   # <===== DEPENDS ON SHAPE
        self.draw_lines()                                                   # <===== DEPENDS ON SHAPE
        self.draw_circs()                                                   # <===== DEPENDS ON SHAPE

#-----------------------------
# is called from ImgFigureManager -> ImgControl ->

    def remove_object(self, obj):
        drag.remove_object_from_img_and_list(obj, self.icp.list_of_rects)       # <===== DEPENDS ON SHAPE 
        drag.remove_object_from_img_and_list(obj, self.icp.list_of_lines)       # <===== DEPENDS ON SHAPE 
        drag.remove_object_from_img_and_list(obj, self.icp.list_of_circs)       # <===== DEPENDS ON SHAPE 

#-----------------------------
# Test
#

def main():
    w = ImgDrawOnTop(None)

#-----------------------------

if __name__ == "__main__" :
    main()

#-----------------------------
