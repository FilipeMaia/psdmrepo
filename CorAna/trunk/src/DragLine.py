#!/usr/bin/env python
#----------------------------------
import sys

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.lines  as lines
import math # cos(x), sin(x), radians(x), degrees()
from Drag import *

class DragLine( Drag, lines.Line2D ) :

    def __init__(self, x=None, y=None, linewidth=2, color='b', picker=5, linestyle='-', str_of_pars=None) :

        Drag.__init__(self, linewidth, color, linestyle, my_type='Line')

        if str_of_pars != None :
            x1,x2,y1,y2,lw,col,s,t,r = self.parse_str_of_pars(str_of_pars)
            self.isSelected    = s
            self.myType        = t
            self.isRemoved     = r
            self.isInitialized = True
            lines.Line2D.__init__(self, (x1,x2), (y1,y2), linewidth=lw, color=col, picker=picker)

        elif x == None or y == None : # Default line initialization
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

    def get_list_of_pars(self) :
        x1  = int( self.get_xdata()[0] )
        x2  = int( self.get_xdata()[1] )
        y1  = int( self.get_ydata()[0] )
        y2  = int( self.get_ydata()[1] )
        lw  = int( self.get_linewidth() ) 
        #col =      self.get_color() 
        col = self.myCurrentColor
        s   = self.isSelected
        t   = self.myType
        r   = self.isRemoved
        return (x1,x2,y1,y2,lw,col,s,t,r)

    def parse_str_of_pars(self, str_of_pars) :
        pars = str_of_pars.split()
        #print 'pars:', pars
        t   = pars[0]
        x1  = float(pars[1])
        x2  = float(pars[2])
        y1  = float(pars[3])
        y2  = float(pars[4])
        lw  = int(pars[5])
        col = str(pars[6])
        s   = self.dicBool[pars[7].lower()]
        r   = self.dicBool[pars[8].lower()]
        return (x1,x2,y1,y2,lw,col,s,t,r)


    def get_str_of_pars(self) :
        x1,x2,y1,y2,lw,col,s,t,r = self.get_list_of_pars()
        return '%s %7.2f %7.2f %7.2f %7.2f %d %s %s %s' % (t,x1,x2,y1,y2,lw,col,s,r)


    def print_pars(self) :
        print 't,x1,x2,y1,y2,lw,col,s,r =', self.get_str_of_pars()


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

        xy0, clickxy, vertindex = self.press
        dx = currentxy[0] - clickxy[0]
        dy = currentxy[1] - clickxy[1]

        xy = copy.deepcopy(xy0)

        if event.button is 3 and self.isInitialized : # for left mouse button
            xy[0][0] += dx
            xy[0][1] += dy
            xy[1][0] += dx
            xy[1][1] += dy

        if event.button is 1 or not self.isInitialized: # for right mouse button
            xy[vertindex][0] += dx
            xy[vertindex][1] += dy

        self.set_data([[xy[0][0], xy[1][0]],[xy[0][1], xy[1][1]]])

        self.on_motion_graphic_manipulations()


    def on_release(self, event):
        'on release we reset the press data'
        self.on_release_graphic_manipulations()
        #if self.press != None : self.print_pars()
        if self.press != None : self.maskIsAvailable = False        
        self.press = None

    #def get_poly_verts(self):
    #    """Creates a set of (closed) poly vertices for mask"""
    #    print 'Is not implemented for DragLine - does not make sense...'
    #    return None

#-----------------------------

    def get_obj_mask(self, shape):
        """Re-implementation of this method from Drag: standard method for points in polygon has no sence for line"""
        if not self.maskIsAvailable :
            self.mask = self.get_mask_for_line(shape)
            self.maskIsAvailable = True
        if self.isSelected : return ~self.mask # inversed mask
        else               : return  self.mask # mask


    def get_mask_for_line(self, shape):
        x1,x2,y1,y2,lw,col,s,t,r = self.get_list_of_pars()

        if abs(x1-x2) > abs(y1-y2) : npix = int(abs(x1-x2)+1)
        else                       : npix = int(abs(y1-y2)+1)

        xarr = np.array( np.linspace(x1, x2, npix, endpoint=True), dtype=np.int32() )
        yarr = np.array( np.linspace(y1, y2, npix, endpoint=True), dtype=np.int32() )

        mask = np.zeros(shape, dtype=np.bool_())
        for c,r in zip(xarr,yarr) : mask[r,c]=True 
        return mask

#-----------------------------
#-----------------------------
#-----------------------------
# Test
#-----------------------------
#-----------------------------
#-----------------------------

from DragObjectSet import *

#-----------------------------
#-----------------------------
 
def generate_list_of_objects(img_extent) :
    """Produce the list of initial random objects for test purpose.
    """
    xmin,xmax,ymin,ymax = img_extent 
    print 'xmin,xmax,ymin,ymax = ', xmin,xmax,ymin,ymax

    nobj = 10
    x = (xmin,xmax)
    y = ymin+(ymax-ymin)*np.random.rand(nobj,2)

    obj_list = []
    for ind in range(nobj) :
        obj = DragLine(x, y[ind], linewidth=2, color='g', picker=5, linestyle='-')
        obj_list.append(obj)

    return obj_list

#-----------------------------

def main_full_test():
    """Full test of the class DragRectangle, using the class DragObjectSet
       1. make a 2D plot
       2. make a list of random objects and add them to the plot
       3. use the class DragObjectSet to switch between Add/Move/Remove modes for full test of the DragLine
    """
    fig, axes, imsh = generate_test_image()
    list_of_objs = generate_list_of_objects(imsh.get_extent())

    t = DragObjectSet(fig, axes, DragLine, useKeyboard=True)
    t.set_list_of_objs(list_of_objs)

    plt.get_current_fig_manager().window.geometry('+50+10') # move(50, 10)
    plt.show()

#-----------------------------

def main_simple_test():
    """Simple test of the class DragRectangle.
       1. make a 2-d plot
       2. make a list of random objects and add them to the plot
       3. add one more object with initialization at 1st click-and-drag of mouse-left button
    """
    fig, axes, imsh = generate_test_image()
    list_of_objs = generate_list_of_objects(imsh.get_extent())

    #Add one more object
    obj = DragLine() # call W/O parameters => object will be initialized at first mouse click
    add_obj_to_axes(obj, axes, list_of_objs)

    #plt.get_current_fig_manager().window.move(50, 10)
    plt.get_current_fig_manager().window.geometry('+50+10')
    plt.show()

#-----------------------------

if __name__ == "__main__" :

    #main_simple_test()
    main_full_test()
    sys.exit ('End of test')

#-----------------------------


