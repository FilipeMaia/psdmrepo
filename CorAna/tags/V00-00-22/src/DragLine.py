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

    def __init__(self, x=None, y=None, linewidth=1, color='b', picker=8, linestyle='-', str_of_pars=None) :

        Drag.__init__(self, linewidth, color, linestyle, my_type='Line')

        if str_of_pars is not None :      # Initialization from input file
            x1,x2,y1,y2,lw,col,s,t,r = self.parse_str_of_pars(str_of_pars)
            self.isSelected    = s
            self.myType        = t
            self.isRemoved     = r
            self.isInitialized = True
            lines.Line2D.__init__(self, (x1,x2), (y1,y2), linewidth=lw, color=col, picker=picker)

        elif x is None or y is None : # Default initialization using mouse 
            lines.Line2D.__init__(self, (0,1), (0,1), linewidth=linewidth, color=color, picker=picker)
            self.isInitialized = False
        else :                        # Initialization from program call
            lines.Line2D.__init__(self,  x,  y, linewidth=linewidth, color=color, picker=picker)
            self.isInitialized = True

        self.set_pickradius(picker)
        self.press    = None
        self.myPicker = picker
        self.vtx      = 0


    def get_list_of_pars(self) :
        xarr, yarr = self.get_data()
        lw  = int( self.get_linewidth() ) 
        #col =      self.get_color() 
        col = self.myCurrentColor
        s   = self.isSelected
        t   = self.myType
        r   = self.isRemoved
        return (xarr[0],xarr[1],yarr[0],yarr[1],lw,col,s,t,r)


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


    #def obj_contains_cursor(self, event):
    #    return False


    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.axes: return

        clickxy = event.xdata, event.ydata
        #print 'clickxy =', event.xdata, event.ydata

        if self.isInitialized :
            contains, attrd = self.contains(event)
            if not contains: return

            xarr, yarr = self.get_data()
            self.dx_prev, self.dy_prev = 0, 0

            # find vertex closest to the click
            self.dist_min=10000
            for ind, xy in enumerate(zip(xarr,yarr)) :
                dist = self.max_deviation(clickxy, xy)
                if dist < self.dist_min :
                    self.dist_min = dist
                    self.vtx = ind

            self.press = clickxy, (xarr[self.vtx], yarr[self.vtx])

            #----Remove object at click on middle mouse botton
            if event.button is 2 : # for middle mouse button
                self.remove_object_from_img() # Remove object from image
                return

        else : # if the line position is not defined yet:

            if event.button is 2 : # Ignore middle button at initialization
                return

            if self.vtx == 0 :
                self.xarr = [clickxy[0]]
                self.yarr = [clickxy[1]]
            
            # Try to prevent double-clicks
            #if self.vtx>1 and clickxy == (self.xarr[self.vtx-1], self.yarr[self.vtx-1]) : return

            self.vtx += 1
            self.xarr.append(clickxy[0]+1) # take it twise, will be changed...
            self.yarr.append(clickxy[1]+1)

            self.press = clickxy, clickxy

        self.on_press_graphic_manipulations()


    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if event.inaxes != self.axes: return
        if self.press is None: return

        #print 'event on_moution', self.get_xydata()
        currentxy = event.xdata, event.ydata

        (x0, y0), (xv, yv) = self.press

        dx = currentxy[0] - x0
        dy = currentxy[1] - y0

        # Move/edit line
        if self.isInitialized : 
            # Move a single vertex
            if event.button is 1 : # for left mouse button
                self.xarr[self.vtx] = xv + dx
                self.yarr[self.vtx] = yv + dy

            # Move entire lane
            if event.button is 3 : # for right mouse button
                for vi in range(len(self.xarr)) :
                    self.xarr[vi] += dx - self.dx_prev
                    self.yarr[vi] += dy - self.dy_prev

                self.dx_prev, self.dy_prev = dx, dy
            
 
        # Draw continuation of the line to the next (moving) point
        else : # not initialized 
            self.xarr[self.vtx] = currentxy[0]
            self.yarr[self.vtx] = currentxy[1]
            #print 'self.xarr:',self.xarr

        self.set_data(self.xarr, self.yarr)
        self.on_motion_graphic_manipulations()


    def on_release(self, event):
        'on release we reset the press data'

        if not self.isInitialized and event.button is 2 : # Ignore middle button at initialization
            return

        self.on_release_graphic_manipulations()
        #if self.press is not None : self.print_pars()
        if self.press is not None : self.maskIsAvailable = False        
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


