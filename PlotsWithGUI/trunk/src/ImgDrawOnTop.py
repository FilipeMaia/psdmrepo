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
import DragWedge           as dragw                             # <===== DEPENDS ON SHAPE
import DragCenter          as drags                             # <===== DEPENDS ON SHAPE

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
        self.wedgsFromInputAreCreated = False                   # <===== DEPENDS ON SHAPE
        self.centsFromInputAreCreated = False                   # <===== DEPENDS ON SHAPE

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
            self.add_obj_on_click(event, drags.DragCenter,    self.icp.formCenter, self.get_axes(), self.icp.list_of_cents) # <===== DEPENDS ON SHAPE
            self.add_obj_on_click(event, dragw.DragWedge,     self.icp.formWedge,  self.get_axes(), self.icp.list_of_wedgs) # <===== DEPENDS ON SHAPE


    def on_mouse_release(self, event) : # is called from ImgControl
        """ImgDrawOnTop : on_mouse_release(...) is called from ImgControl
        """
        self.update_list_of_all_objs()


    def update_list_of_all_objs(self) :
        self.update_list_of_centers()                                        # <===== DEPENDS ON SHAPE
        self.update_list_of_objs( self.icp.list_of_rects )                   # <===== DEPENDS ON SHAPE
        self.update_list_of_objs( self.icp.list_of_lines )                   # <===== DEPENDS ON SHAPE
        self.update_list_of_objs( self.icp.list_of_circs )                   # <===== DEPENDS ON SHAPE
        self.update_list_of_objs( self.icp.list_of_wedgs )                   # <===== DEPENDS ON SHAPE


    def update_list_of_objs(self, list_of_objs):
        initial_list_of_objs = list(list_of_objs) # COPY list
        for obj in initial_list_of_objs :
            if obj.isRemoved :
                self.send_signal_and_remove_object_from_list(obj, list_of_objs, 'Click')
                continue

            if obj.myType == self.icp.typeProjRP : # formWedge :
                obj.set_center((self.icp.x_center, self.icp.y_center))



    def update_list_of_centers(self) :
        list_of_objs = self.icp.list_of_cents
        initial_list_of_objs = list(list_of_objs) # COPY list
        list_len = len(list_of_objs)
        obj_last = list_of_objs[list_len-1]
        # Remove all center objects except the last one:
        for obj in initial_list_of_objs :
            if obj == obj_last :
                (xc,yc,xerr,yerr,lw,col,s,t,r) = obj.get_list_of_center_pars()

                if xc != self.icp.x_center or yc != self.icp.y_center :
                    self.icp.x_center = xc
                    self.icp.y_center = yc
                    self.get_control().signal_center_is_reset_on_click()
                    
            else :
                obj.remove_object_from_img()
                list_of_objs.remove(obj)


    def set_center_position_from_icp(self) :
        obj = self.icp.list_of_cents[0]
        obj.reset_center_position(self.icp.x_center, self.icp.y_center)




    def set_all_objs_need_in_redraw(self):
        #if self.icp.typeCurrent != self.icp.typeSpectrum : return
        drag.set_list_need_in_redraw(self.icp.list_of_rects)                 # <===== DEPENDS ON SHAPE
        drag.set_list_need_in_redraw(self.icp.list_of_lines)                 # <===== DEPENDS ON SHAPE
        drag.set_list_need_in_redraw(self.icp.list_of_circs)                 # <===== DEPENDS ON SHAPE
        drag.set_list_need_in_redraw(self.icp.list_of_wedgs)                 # <===== DEPENDS ON SHAPE
        drag.set_list_need_in_redraw(self.icp.list_of_cents)                 # <===== DEPENDS ON SHAPE


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
        """Creates a new object on mouse click
        """
        if self.icp.formCurrent != form : return                            # check that the form is correct 

        center = (self.icp.x_center, self.icp.y_center)                     #FOR OBJECTS WHICH NEED IN CENTER
        if form == self.icp.formWedge :

            print 'Initialize Wedge for center=', center

            obj = DragObject(xy=center)                                     # create semi-default object
            
        else :
            obj = DragObject()                                              # create default object

        drag.add_obj_to_axes(obj, axes, list_of_objs)                       # add object to axes and in the list
        obj.on_press(event)                                                 # Initialize object by the mouse drag
        obj.myType     = self.icp.typeCurrent                               # set attribute
        obj.myIndex    = list_of_objs.index(obj)                            # set attribute
        obj.isSelected = False                                              # set attribute        
        
        if  obj.myType == self.icp.typeProjRP  :
            obj.n_rings    = self.icp.n_rings                                   # set attribute
            obj.n_sects    = self.icp.n_sects                                   # set attribute

        if  obj.myType == self.icp.typeProjXY  :
            obj.nx_slices  = self.icp.nx_slices                                 # set attribute
            obj.ny_slices  = self.icp.ny_slices                                 # set attribute




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
                t,s,x,y,w,h,lw,col,nx,ny = objPars
                #if t == self.icp.typeCurrent :
                obj = dragr.DragRectangle(xy=(x,y), width=w, height=h, color='r')
                obj.nx_slices = nx
                obj.ny_slices = ny
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
            for objPars in self.icp.listOfLineInputParameters :
                #print objPars
                t,s,x1,x2,y1,y2,lw,col = objPars
                #if t == self.icp.typeCurrent :
                obj = dragl.DragLine((x1,x2), (y1,y2), linewidth=2, color='r')
                self.add_obj_on_call(obj, axes, list_of_objs, type=t, selected=s)
        else:
            drag.redraw_objs_from_list(axes, list_of_objs)



    def draw_circs(self) :                                      # <===== DEPENDS ON SHAPE INSIDE
        """Draw circs on top of the main image plot
        """
        axes         = self.get_axes()
        list_of_objs = self.icp.list_of_circs

        if not self.circsFromInputAreCreated :
            self.circsFromInputAreCreated = True
            for objPars in self.icp.listOfCircInputParameters :
                #print objPars
                t,s,x0,y0,r,lw,col = objPars
                #if t == self.icp.typeCurrent :
                obj = dragc.DragCircle((x0, y0), radius=r, color='r')
                self.add_obj_on_call(obj, axes, list_of_objs, type=t, selected=s)
        else:
            drag.redraw_objs_from_list(axes, list_of_objs)



    def draw_wedgs(self) :                                     # <===== DEPENDS ON SHAPE INSIDE
        """Draw wedges on top of the main image plot
        """
        axes         = self.get_axes()
        list_of_objs = self.icp.list_of_wedgs

        if not self.wedgsFromInputAreCreated :
            self.wedgsFromInputAreCreated = True
            for objPars in self.icp.listOfWedgInputParameters :
                #print objPars
                t,s,x,y,r,w,t1,t2,lw,col,nr,np = objPars
                #if t == self.icp.typeCurrent :
                obj = dragw.DragWedge(xy=(x,y), radius=r, width=w, theta1=t1, theta2=t2, color='r')
                obj.n_rings = nr
                obj.n_sects = np
                obj.set_center((self.icp.x_center, self.icp.y_center))
                self.add_obj_on_call(obj, axes, list_of_objs, type=t, selected=s)
        else:
            drag.redraw_objs_from_list(axes, list_of_objs)



    def draw_cents(self) :                                      # <===== DEPENDS ON SHAPE INSIDE
        """Draw centers on top of the main image plot
        """
        axes         = self.get_axes()
        list_of_objs = self.icp.list_of_cents

        if not self.centsFromInputAreCreated :
            self.centsFromInputAreCreated = True
            for objPars in self.icp.listOfCentInputParameters :
                #print objPars
                t,s,xc,yc,xe,ye,lw,col = objPars
                #if t == self.icp.typeCurrent :
                obj = drags.DragCenter(xc, yc, xe, ye, linewidth=2, color='g')
                self.add_obj_on_call(obj, axes, list_of_objs, type=t, selected=s)
        else:
            drag.redraw_objs_from_list(axes, list_of_objs)


    def draw_on_top(self):
        print 'draw_on_top()'
        self.draw_cents()                                                   # <===== DEPENDS ON SHAPE
        self.draw_rects()                                                   # <===== DEPENDS ON SHAPE
        self.draw_lines()                                                   # <===== DEPENDS ON SHAPE
        self.draw_circs()                                                   # <===== DEPENDS ON SHAPE
        self.draw_wedgs()                                                   # <===== DEPENDS ON SHAPE

#-----------------------------
# is called from ImgFigureManager -> ImgControl ->

    def remove_object(self, obj):
        drag.remove_object_from_img_and_list(obj, self.icp.list_of_rects)       # <===== DEPENDS ON SHAPE 
        drag.remove_object_from_img_and_list(obj, self.icp.list_of_lines)       # <===== DEPENDS ON SHAPE 
        drag.remove_object_from_img_and_list(obj, self.icp.list_of_circs)       # <===== DEPENDS ON SHAPE 
        drag.remove_object_from_img_and_list(obj, self.icp.list_of_wedgs)       # <===== DEPENDS ON SHAPE 
        drag.remove_object_from_img_and_list(obj, self.icp.list_of_cents)       # <===== DEPENDS ON SHAPE 

#-----------------------------
# Test
#

def main():
    w = ImgDrawOnTop(None)

#-----------------------------

if __name__ == "__main__" :
    main()

#-----------------------------
