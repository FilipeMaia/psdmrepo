#!/usr/bin/env python
#----------------------------------

import sys

import numpy as np
#import copy
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import math # cos(x), sin(x), radians(x), degrees()
#import copy
from Drag import *

class DragRectangle( Drag, patches.Rectangle ) : 

    def __init__(self, xy=None, width=1, height=1, linewidth=2, linestyle='solid', color='b', picker=2) :

        Drag.__init__(self, linewidth, color, linestyle)

        if  xy == None : # Default line initialization
            xy0=(0,0)
            patches.Rectangle.__init__(self, xy0, width, height, linewidth=linewidth, color=color, fill=False, picker=picker)
            self.isInitialized = False
            print "DragRectangle initialization w/o parameters."

        else :
            patches.Rectangle.__init__(self, xy,  width, height, linewidth=linewidth, color=color, fill=False, picker=picker)
            self.isInitialized = True

        self.set_picker(picker)
        self.myPicker = picker
        self.press    = None # Is used to transmit local information between press and release button


    def get_list_of_rect_pars(self) :
        x0 = int( self.get_x() )
        y0 = int( self.get_y() )
        w0 = int( self.get_width () )
        h0 = int( self.get_height() )
        x  = min(x0,x0+w0)
        y  = min(y0,y0+h0)
        h  = abs(h0)
        w  = abs(w0)
        s  =      self.isSelected
        t  =      self.myType
        return (x,y,w,h,s,t)


    def print_pars(self) :
        x,y,w,h,s,t = self.get_list_of_rect_pars()
        r = self.isRemoved
        print 'x,y,w,h,s,t,r =', x,y,w,h,s,t,r


    def my_contains(self, event):
        x,y = event.xdata, event.ydata
        x0  = self.get_x()
        y0  = self.get_y()
        w0  = self.get_width()
        h0  = self.get_height()
        r   = self.myPicker

        xmin = min(x0, x0+w0)
        xmax = max(x0, x0+w0)
        ymin = min(y0, y0+h0)
        ymax = max(y0, y0+h0)

        if x > xmin-r and x < xmax+r and y > ymin-r and y < ymax+r :
            self.inLargeBox = True
        else :
            self.inLargeBox  = False

        if x > xmin+r and x < xmax-r and y > ymin+r and y < ymax-r :
            self.inSmallBox = True
        else :
            self.inSmallBox = False

        #print 'self.inLargeBox =',self.inLargeBox ,'  self.inSmallBox =', self.inSmallBox

        if self.inLargeBox and not self.inSmallBox:
            return True
        else :
            return False


    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.axes: return

        clickxy = event.xdata, event.ydata
        #print 'clickxy =',clickxy 

        if self.isInitialized :
            #contains, attrd = self.contains(event)
            contains = self.my_contains(event)
            if not contains: return
            #print 'event contains object',

            xy0 = self.get_xy()
            x0  = self.get_x()
            y0  = self.get_y()
            w0  = self.get_width()
            h0  = self.get_height()

            vertindex = 5 # click xy is already contained around rect area within self.myPicker 

            if  self.max_deviation(clickxy,(x0,y0))        < self.myPicker :
                vertindex = 0
            elif self.max_deviation(clickxy,(x0+w0,y0))    < self.myPicker :
                vertindex = 1
            elif self.max_deviation(clickxy,(x0+w0,y0+h0)) < self.myPicker :
                vertindex = 2
            elif self.max_deviation(clickxy,(x0,y0+h0))    < self.myPicker :
                vertindex = 3

            #print 'vertindex=',vertindex

            self.press = xy0, w0, h0, clickxy, vertindex

            #if event.button is 2 : # for middle mouse button
            #    print 'Remove the rect now...'
            #    self.remove_from_axes()

        else : # if the object position is not defined yet:
            vertindex = 0
            xy0 = clickxy
            w0  = 1
            h0  = 1
            self.press = xy0, w0, h0, clickxy, vertindex

        self.on_press_graphic_manipulations()


    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.axes: return

        #print 'event on_moution', self.get_xydata()
        currentxy = event.xdata, event.ydata

        xy0, w0, h0, clickxy, vertindex = self.press
        xy = [xy0[0], xy0[1]]

        dx = currentxy[0]-clickxy[0]
        dy = currentxy[1]-clickxy[1] 

        if self.isInitialized : # for left mouse button

            if   vertindex == 5 : #side
                self.set_xy( (xy0[0] + dx,  xy0[1] + dy) )
            elif vertindex == 0 :
                self.set_xy( (xy0[0] + dx,  xy0[1] + dy) )
                self.set_width (w0 - dx)
                self.set_height(h0 - dy)
            elif vertindex == 1 :
                self.set_y( xy0[1] + dy)
                self.set_width (w0 + dx)
                self.set_height(h0 - dy)
            elif vertindex == 2 :
                self.set_width (w0 + dx)
                self.set_height(h0 + dy)
            elif vertindex == 3 :
                self.set_x( xy0[0] + dx)
                self.set_width (w0 - dx)
                self.set_height(h0 + dy)

        else :
            self.set_width (dx)
            self.set_height(dy)
            self.set_xy(xy0)


        self.on_motion_graphic_manipulations()


    def on_release(self, event):
        'on release we reset the press data'
        self.on_release_graphic_manipulations()

        if self.press is None: return
        if event.button is 2 : # for middle mouse button
            self.remove_object_from_img() # Remove object from image for test

        self.press = None


#-----------------------------
#-----------------------------
#-----------------------------
# Test
#-----------------------------
#-----------------------------
#-----------------------------

class TestDragRectangle : 

    def __init__(self, fig, axes) :

        self.fig          = fig
        self.axes         = axes
        self.fig.my_mode  = None # This is used to transmit signals
        self.list_of_objs = []
        self.needInUpdate = False

        self.fig.canvas.mpl_connect('key_press_event',      self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event',   self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event',  self.on_mouse_motion)
        self.print_mode_keys()


    def set_list_of_objs(self, list_of_objs) :
        self.list_of_objs = list_of_objs

        
    def print_mode_keys(self) :
        """Prints the hint for mode selection using keyboard
        """
        print '\n\nUse keyboard to select the mode: \nA=ADD, \nM=MARK, \nD=REMOVE SELECTED, \nR=REMOVE, \nN=NONE, \nW=PRINT list'


    def on_key_press(self, event) :
        """Responds on keyboard signals and switch the fig.my_mode
           It also prints the list of objects for key W.
        """
        if   event.key == 'a': self.fig.my_mode  = 'Add'
        elif event.key == 'n': self.fig.my_mode  = 'None'
        elif event.key == 'r': self.fig.my_mode  = 'Remove'
        elif event.key == 'm': self.fig.my_mode  = 'Select'
        elif event.key == 'w': self.print_list_of_objs()
        elif event.key == 'd': self.test_remove_selected_objs_from_img_by_call()
        else                 : self.print_mode_keys()
        print '\nCurrent mode:', self.fig.my_mode


    def on_mouse_press(self, event) :
        """Responds on mouse signals and do the object initialization for the mode Add
        """
        xy = event.xdata, event.ydata
        #print 'TestDragRectangle : on_mouse_press(...), xy =', xy        
        if self.fig.my_mode  == 'Add' :
            if event.button != 1 : return # if other than Left mouse button
            #print 'mode=', self.fig.my_mode
            obj = DragRectangle(xy) # Creates the DragRectangle object with 1st vertex in xy
            add_obj_to_axes(obj, self.axes, self.list_of_objs)      # <<<==========================
            obj.on_press(event)                                     # <<<==========================


    def on_mouse_release(self, event) :
        self.needInUpdate = True # Works in on_mouse_motion(...)

    def on_mouse_motion(self, event) :
        """HAVE TO USE IT, because I need to update the list after the button release loop over all objects...
        """
        #print 'mouse is moved.. do something...', event.xdata, event.ydata
        if self.needInUpdate :
            self.update_list_of_objs()
            self.needInUpdate = False

    
    def update_list_of_objs(self) :
        print 'update_list_of_objs()'
        for obj in self.list_of_objs :
            if obj.isRemoved :

                #====================== REMOVE OBJECT BY CLICK ON MOUSE ===============================
                # THIS IS A PLACE TO REMOVE EVERYTHING ASSOCIATED WITH OBJECT AFTER CLICK ON MOUSE
                print 'Object ', self.list_of_objs.index(obj), 'is removing from the list. ACT HERE !!!'
                self.list_of_objs.remove(obj)
                #======================================================================================


    def test_remove_selected_objs_from_img_by_call(self) :
        """Loop over list of objects and remove selected from the image USING CALL from code.
        """
        print 'test_remove_selected_objs_from_img_by_call()'
        for obj in self.list_of_objs :
            if obj.isSelected :
                print 'Object', self.list_of_objs.index(obj), 'is selected and will be removed in this test'

                #====================== REMOVE OBJECT BY PROGRAM CALL =================================
                # THIS IS A PLACE TO REMOVE EVERYTHING ASSOCIATED WITH OBJECT AFTER PROGRAM CALL
                print 'Object ', self.list_of_objs.index(obj), 'is removing from the list. ACT HERE !!!'
                obj.remove_object_from_img()  # <<<========= PROGRAM CALL TO REMOVE THE OBJECT FROM IMG
                self.list_of_objs.remove(obj) # <<<========= REMOVE OBJECT FROM THE LIST
                #======================================================================================

        self.needInUpdate = False


    def print_list_of_objs(self) :
        """Prints the list of objects with its parameters.
        """
        print 'Print list of', len(self.list_of_objs), 'objects'
        for obj in self.list_of_objs :
            print 'ind=', self.list_of_objs.index(obj),':',
            obj.print_pars()

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
    x = xmin+(xmax-xmin)*np.random.rand(nobj)
    y = ymax+(ymin-ymax)*np.random.rand(nobj)
    w = (xmax-xmin)/3*np.random.rand(nobj)
    h = (ymin-ymax)/3*np.random.rand(nobj)
    #print ' x=',x, ' y=',y, ' w=',w, ' h=',h

    obj_list = []

    # Add objects with initialization through the parameters
    for indobj in range(nobj) :
        obj = DragRectangle((x[indobj],y[indobj]), w[indobj], h[indobj], color='g')
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

    t = TestDragRectangle(fig, axes)
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
    obj = DragRectangle() # call W/O parameters => object will be initialized at first mouse click
    add_obj_to_axes(obj, axes, list_of_objs)

    plt.get_current_fig_manager().window.move(50, 10)
    plt.show()

#-----------------------------

if __name__ == "__main__" :

    #main_simple_test()
    main_full_test()
    sys.exit ('End of test')

#-----------------------------
