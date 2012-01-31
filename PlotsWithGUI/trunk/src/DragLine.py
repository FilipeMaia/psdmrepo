#!/usr/bin/env python
#----------------------------------
import sys

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.lines  as lines
import math # cos(x), sin(x), radians(x), degrees()
#import copy
from Drag import *

class DragLine( Drag, lines.Line2D ) :

    def __init__(self, x=None, y=None, linewidth=2, color='b', picker=5, linestyle='-') :

        Drag.__init__(self, linewidth, color, linestyle)

        if x == None or y == None : # Default line initialization
            x0=y0=(0,0)
            lines.Line2D.__init__(self, x0, y0, linewidth=linewidth, color=color, picker=picker)
            self.isInitialized = False
        else :
            lines.Line2D.__init__(self,  x,  y, linewidth=linewidth, color=color, picker=picker)
            self.isInitialized = True

        self.set_pickradius(picker)
        self.press    = None
        self.myPicker = picker


    #def set_dragged_obj_properties(self):
    #    self.set_color    ('k')
    #    self.set_linewidth(1)
    #    self.set_linestyle('--') #'--', ':'

    def get_list_of_line_pars(self) :
        x = self.get_xdata()
        y = self.get_ydata()
        s = self.isSelected
        r = self.isRemoved
        t = self.myType
        return (x,y,s,t,r)


    def print_pars(self) :
        x,y,s,t,r = self.get_list_of_line_pars()
        r = self.isRemoved
        print 'x,y,s,t,r =', x,y,s,t,r


    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.axes: return

        clickxy = event.xdata, event.ydata
        #print 'clickxy =',clickxy

        if self.isInitialized :
            contains, attrd = self.contains(event)
            #contains = self.contains(event)
            if not contains: return

            xy0 = self.get_xydata()
            #print 'event contains: xy0[0], xy0[1]', xy0[0], xy0[1]
            if self.distance(clickxy,xy0[0]) < self.distance(clickxy,xy0[1]) :
                vertindex = 0
            else :
                vertindex = 1
            self.press = xy0, clickxy, vertindex

            #----Remove object at click on middle mouse botton
            if event.button is 2 : # for middle mouse button
                self.remove_object_from_img() # Remove object from image
                return

        else : # if the line position is not defined yet:
            vertindex = 1
            xy0 = [event.xdata, event.ydata], [event.xdata, event.ydata]
            self.press = xy0, clickxy, vertindex

        self.on_press_graphic_manipulations()


    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.axes: return

        #print 'event on_moution', self.get_xydata()
        currentxy = event.xdata, event.ydata
        #print 'self.onmoutionxy =',currentxy 

        xy0, clickxy, vertindex = self.press
        dx = currentxy[0] - clickxy[0]
        dy = currentxy[1] - clickxy[1]

        #xy = xy0.copy() # copy
        #xy = [xy0[0][0], xy0[0][1]], [xy0[1][0], xy0[1][1]]
        xy = copy.deepcopy(xy0)

        #print 'xy0=', xy0
        #print 'xy =', xy 

        xy[vertindex][0] += dx
        xy[vertindex][1] += dy

        self.set_data([[xy[0][0], xy[1][0]],[xy[0][1], xy[1][1]]])

        self.on_motion_graphic_manipulations()


    def on_release(self, event):
        'on release we reset the press data'
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

class TestDragLine(TestDragObject) : 

    def __init__(self, fig, axes) :
        TestDragObject.__init__(self, fig, axes)


    # THE ONLY OBJECT-SPECIFIC METHOD, IN ADDITION TO class TestDragObject
    def on_mouse_press(self, event) :
        """Responds on mouse signals and do the object initialization for the mode Add
        """
        #x = (event.xdata, event.xdata+1)
        #y = (event.ydata, event.ydata+1)
        #print 'TestDragRectangle : on_mouse_press(...), xy =', xy        
        if self.fig.my_mode  == 'Add' :
            if event.button != 1 : return # if other than Left mouse button
            #print 'mode=', self.fig.my_mode
            obj = DragLine() # Creates the DragRectangle object with 1st vertex in xy
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
    """Produce the list of random objects (Rectangles) for test purpose.
    1. Generates initial list of random objects
    2. Add them to the figure axes
    3. Connect with signals.
    4. Returns the list of created objects.
    """
    xmin,xmax,ymin,ymax =  axesImage.get_extent() 
    #print 'xmin,xmax,ymin,ymax = ', xmin,xmax,ymin,ymax

    nobj = 10
    x = (xmin,xmax)
    y = ymin+(ymax-ymin)*np.random.rand(nobj,2)

    obj_list = []

    # Add objects with initialization through the parameters
    for indobj in range(nobj) :
        obj = DragLine(x, y[indobj], linewidth=2, color='g', picker=5, linestyle='-')
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

    t = TestDragLine(fig, axes)
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
    obj = DragLine() # call W/O parameters => object will be initialized at first mouse click
    add_obj_to_axes(obj, axes, list_of_objs)

    plt.get_current_fig_manager().window.move(50, 10)
    plt.show()

#-----------------------------

if __name__ == "__main__" :

    #main_simple_test()
    main_full_test()
    sys.exit ('End of test')

#-----------------------------


