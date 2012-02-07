#!/usr/bin/env python
#----------------------------------
import sys

import numpy as np
#import copy
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import math # cos(x), sin(x), radians(x), degrees()
#import copy

#import Drag as drag
from Drag import *


class DragCircle( Drag, patches.Circle ) :  #patches.CirclePolygon

    def __init__(self, xy=None, radius=1, linewidth=2, linestyle='solid', color='b', picker=5) :

        Drag.__init__(self, linewidth, color, linestyle)

        if  xy == None : # Default line initialization
            xy0=(0,0)
            r0=1
            #patches.CirclePolygon.__init__(self, xy0, linewidth=linewidth, color=color, fill=False, picker=picker)
            patches.Circle.__init__(self, xy0, radius=r0, linewidth=linewidth, color=color, fill=False, picker=picker)
            #self.isInitialized = False
            self.set_isInitialized(False)
        else :
            patches.Circle.__init__(self, xy, radius=radius, linewidth=linewidth, color=color, fill=False, picker=picker)
            #self.isInitialized = True
            self.set_isInitialized(True)

        self.set_picker(picker)
        self.myPicker = picker
        self.press    = None


    #def contains(self, event):
    #    clickxy  = event.xdata, event.ydata
    #    xy0 = self.center
    #    r0  = self.get_radius()
    #    if abs(self.distance(clickxy,xy0) - r0) < self.myPicker :
    #        return True
    #    else :
    #        return False

    def get_list_of_circ_pars(self) :
        xc,yc = self.center
        x,y = (int(xc), int(yc))
        r0  = int( self.get_radius() )
        lw  = int( self.get_linewidth() ) 
        col =      self.get_edgecolor() 
        s   =      self.isSelected
        t   =      self.myType
        r   =      self.isRemoved
        return (x,y,r0,lw,col,s,t,r)


    def print_pars(self) :
        x,y,r0,lw,col,s,t,r = self.get_list_of_circ_pars()
        print 'x,y,r0,lw,col,s,t,r =', x,y,r0,lw,col,s,t,r


    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.axes: return

        clickxy = event.xdata, event.ydata
        #print 'clickxy =',clickxy 

        if self.isInitialized :
            #contains, attrd = self.contains(event)
            #if not contains: return
            #print 'event contains circle',
            r0  = self.get_radius()
            xy0 = self.center
            if abs(self.distance(clickxy,xy0) - r0) > self.myPicker : return

            vertindex = 1
            self.press = xy0, clickxy, vertindex, r0

            #----Remove object at click on middle mouse botton
            if event.button is 2 : # for middle mouse button
                self.remove_object_from_img() # Remove object from image
                return

        else : # if the object position is not defined yet:
            vertindex = 0
            r0  = 0
            xy0 = [event.xdata, event.ydata]
            self.press = xy0, clickxy, vertindex, r0
            self.center = xy0

        self.on_press_graphic_manipulations()


    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.axes: return

        #print 'event on_moution', self.get_xydata()
        currentxy = event.xdata, event.ydata

        xy0, clickxy, vertindex, r0 = self.press
        xy = [xy0[0], xy0[1]]

        if event.button is 3 and self.isInitialized : # for left mouse button
            xy[0] += (currentxy[0] - clickxy[0])
            xy[1] += (currentxy[1] - clickxy[1])
            self.center = xy[0], xy[1]

        if event.button is 1 or not self.isInitialized: # for right mouse button
            dx = currentxy[0] - xy0[0]
            dy = currentxy[1] - xy0[1]
            r = math.sqrt(dx*dx + dy*dy)
            self.set_radius(r)

        self.on_motion_graphic_manipulations()


    def on_release(self, event):
    #    'on release we reset the press data'
        self.on_release_graphic_manipulations()
        self.press = None


#-----------------------------
#-----------------------------
#-----------------------------
# Test
#-----------------------------
#-----------------------------
#-----------------------------

from TestDragObject import *

class TestDragCircle(TestDragObject) : 

    def __init__(self, fig, axes) :
        TestDragObject.__init__(self, fig, axes)


    # THE ONLY OBJECT-SPECIFIC METHOD, IN ADDITION TO class TestDragObject
    def on_mouse_press(self, event) :
        """Responds on mouse signals and do the object initialization for the mode Add
        """
        #xy = event.xdata, event.ydata
        #print 'TestDragRectangle : on_mouse_press(...), xy =', xy        
        if self.fig.my_mode  == 'Add' :
            if event.button != 1 : return # if other than Left mouse button
            #print 'mode=', self.fig.my_mode
            obj = DragCircle() # Creates the DragRectangle object with 1st vertex in xy
            #obj = DragCircle(xy) # Creates the DragRectangle object with 1st vertex in xy
            add_obj_to_axes(obj, self.axes, self.list_of_objs)      # <<<==========================
            obj.on_press(event)                                     # <<<==========================


#-----------------------------
#-----------------------------
# ===>>> Moved to Drag.py
#-----------------------------
# def add_obj_to_axes(obj, axes, list_of_objs) :
# def get_array2d_for_test() :
# def generate_test_image() :
#-----------------------------
#-----------------------------

def generate_list_of_objects_for_axes(axes, axesImage) :
    """Produce the list of random objects (Circle) for test purpose.
    1. Generates initial list of random objects
    2. Add them to the figure axes
    3. Connect with signals.
    4. Returns the list of created objects.
    """
    xmin,xmax,ymin,ymax =  axesImage.get_extent() 
    #print 'xmin,xmax,ymin,ymax = ', xmin,xmax,ymin,ymax

    nobj = 10
    x = xmin+(xmax-xmin)*np.random.rand(nobj)
    y = ymin+(ymax-ymin)*np.random.rand(nobj)
    r = (ymin-ymax)/3*np.random.rand(nobj)

    obj_list = []
    # Add objects with initialization through the parameters
    for indobj in range(nobj) :
        obj = DragCircle((x[indobj], y[indobj]), radius=r[indobj], color='g')
        add_obj_to_axes(obj, axes, obj_list)      # <<<==========================

    return obj_list

#-----------------------------

def main_full_test():
    """Full test of the class DragRectangle, using the class TestDragRectangle
       1. make a 2-d plot
       2. make a list of random objects and add them to the plot
       3. use the class TestDragRectangle to switch between modes for full test of the class DragRectangle
    """
    fig, axes, axesImage = generate_test_image()

    list_of_objs = generate_list_of_objects_for_axes(axes, axesImage)

    t = TestDragCircle(fig, axes)
    t .set_list_of_objs(list_of_objs)

    plt.get_current_fig_manager().window.move(50, 10)
    plt.show()

#-----------------------------

def main_simple_test():
    """Simple test of the class DragRectangle.
       1. make a 2-d plot
       2. make a list of random objects and add them to the plot
       3. add one more object with initialization at 1st click-and-drag of mouse-left button
    """
    fig, axes, axesImage = generate_test_image()

    list_of_objs = generate_list_of_objects_for_axes(axes, axesImage)

    #Add one more object
    obj = DragCircle() # call W/O parameters => object will be initialized at first mouse click
    add_obj_to_axes(obj, axes, list_of_objs)

    plt.get_current_fig_manager().window.move(50, 10)
    plt.show()

#-----------------------------

if __name__ == "__main__" :

    #main_simple_test()
    main_full_test()
    sys.exit ('End of test')

#-----------------------------
