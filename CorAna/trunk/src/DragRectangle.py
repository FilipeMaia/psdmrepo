#!/usr/bin/env python
#----------------------------------

import sys
import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import math # cos(x), sin(x), radians(x), degrees()

from Drag import *

class DragRectangle( Drag, patches.Rectangle ) : 

    def __init__(self, xy=None, width=1, height=1, linewidth=2, linestyle='solid', color='b', picker=8, fill=False, str_of_pars=None) :

        Drag.__init__(self, linewidth, color, linestyle, my_type='Rectangle')

        if str_of_pars is not None :
            x,y,w,h,lw,col,s,t,r = self.parse_str_of_pars(str_of_pars)
            patches.Rectangle.__init__(self, (x,y), w, h, linewidth=lw, color=col, fill=fill, picker=picker)
            self.isSelected    = s
            self.myType        = t
            self.isRemoved     = r
            self.isInitialized = True

        elif xy is None : # Default line initialization
            xy0=(0,0)
            patches.Rectangle.__init__(self, xy0, width, height, linewidth=linewidth, color=color, fill=fill, picker=picker)
            self.isInitialized = False
            #print "DragRectangle initialization w/o parameters."

        else :
            patches.Rectangle.__init__(self, xy,  width, height, linewidth=linewidth, color=color, fill=fill, picker=picker)
            self.isInitialized = True

        self.set_picker(picker)
        self.myPicker  = picker
        self.press     = None # Is used to transmit local information between press and release button

        
    def get_list_of_pars(self) :
        x0 = int( self.get_x() )
        y0 = int( self.get_y() )
        w0 = int( self.get_width () )
        h0 = int( self.get_height() )
        lw = int( self.get_linewidth() ) 
        col= self.myCurrentColor
        x  = min(x0,x0+w0)
        y  = min(y0,y0+h0)
        h  = abs(h0)
        w  = abs(w0)
        s  = self.isSelected
        t  = self.myType
        r  = self.isRemoved
        return (x,y,w,h,lw,col,s,t,r)


    def parse_str_of_pars(self, str_of_pars) :
        pars = str_of_pars.split()
        #print 'pars:', pars
        t   = pars[0]
        x   = float(pars[1])
        y   = float(pars[2])
        w   = float(pars[3])
        h   = float(pars[4])
        lw  = int(pars[5])
        col = str(pars[6])
        s   = self.dicBool[pars[7].lower()]
        r   = self.dicBool[pars[8].lower()]
        #print 'Parsed pars: %s %7.2f %7.2f %7.2f %7.2f %d %s %s %s' % (t,x,y,w,h,lw,col,s,r)
        return (x,y,w,h,lw,col,s,t,r)


    def get_str_of_pars(self) :
        x,y,w,h,lw,col,s,t,r = self.get_list_of_pars()
        return '%s %7.2f %7.2f %7.2f %7.2f %d %s %s %s' % (t,x,y,w,h,lw,col,s,r)


    def print_pars(self) :
        print 't,x,y,w,h,lw,col,s,r =', self.get_str_of_pars()


    def obj_contains_cursor(self, event): # Overrides method in Drag
        if not self.isInitialized   : return False
        if event.inaxes != self.axes: return False
        return self.my_contains(event)


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
            w2  = w0/2
            h2  = h0/2

            self.list_of_verts = [(x0,y0),    (x0+w0,y0),    (x0,y0+h0), (x0+w0,y0+h0), 
                                  (x0,y0+h2), (x0+w0,y0+h2), (x0+w2,y0), (x0+w2,y0+h0)]

# Numeration in vertindex the vertices and sides of the wedge
#
#        x0                x0+w0
#        
# y0+h0  3--------8--------4
#        |                 |
#        |                 |
#        5                 6
#        |                 |
#        |                 |
# y0     1--------7--------2


            vertindex = 10 # click xy is already contained around rect area within self.myPicker 

            self.dist_min = 1000
            for i, vert in enumerate(self.list_of_verts) :
                dist = self.max_deviation(clickxy,vert)
                if dist < self.dist_min :
                    vertindex = i+1
                    self.dist_min = dist

            #print 'vertindex=',vertindex

            self.press = xy0, w0, h0, clickxy, vertindex

            #----Remove object at click on middle mouse botton
            if event.button is 2 : # for middle mouse button
                self.remove_object_from_img() # Remove object from image
                return

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

        if event.button is 3 and self.isInitialized : # move on right mouse button

            #if   vertindex == 10 : #side
            self.set_xy( (xy0[0] + dx,  xy0[1] + dy) )


        elif event.button is 1 and self.isInitialized : # change size for left mouse button

            if   vertindex == 1 :
                self.set_xy( (xy0[0] + dx,  xy0[1] + dy) )
                self.set_width (w0 - dx)
                self.set_height(h0 - dy)
            elif vertindex == 2 :
                self.set_y( xy0[1] + dy)
                self.set_width (w0 + dx)
                self.set_height(h0 - dy)
            elif vertindex == 3 :
                self.set_x( xy0[0] + dx)
                self.set_width (w0 - dx)
                self.set_height(h0 + dy)
            elif vertindex == 4 :
                self.set_width (w0 + dx)
                self.set_height(h0 + dy)
            elif vertindex == 5 :
                self.set_x( xy0[0] + dx)
                self.set_width (w0 - dx)
            elif vertindex == 6 :
                self.set_width (w0 + dx)
            elif vertindex == 7 :
                self.set_y( xy0[1] + dy)
                self.set_height(h0 - dy)
            elif vertindex == 8 :
                self.set_height(h0 + dy)

        elif event.button is 1 and not self.isInitialized : # add new rect

            self.set_width (dx)
            self.set_height(dy)
            self.set_xy(xy0)

        self.on_motion_graphic_manipulations()


    def on_release(self, event):
        'on release we reset the press data'
        self.on_release_graphic_manipulations()
        #if self.press is not None : self.print_pars()
        if self.press is not None : self.maskIsAvailable = False        
        self.press = None


    def get_poly_verts(self):
        """Creates a set of (closed) poly vertices for mask"""
        x,y,w,h,lw,col,s,t,r = self.get_list_of_pars()
        return [(x,y), (x+w,y), (x+w,y+h), (x,y+h), (x,y)] 

#-----------------------------

    def get_obj_mask(self, shape):
        """Re-implementation of this method from Drag: standard method for points in polygon is very slow"""
        if not self.maskIsAvailable :
            self.mask = self.get_mask_for_rectangle(shape)
            self.maskIsAvailable = True
        if self.isSelected : return ~self.mask # inversed mask
        else               : return  self.mask # mask


    def get_mask_for_rectangle(self, shape):
        x0,y0,w,h,lw,col,s,t,r = self.get_list_of_pars()
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        return np.select([x<x0, x>x0+w, y<y0, y>y0+h], [False, False, False, False], default=True)

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
    1. Generates initial list of random objects
    2. Add them to the figure axes
    3. Connect with signals.
    4. Returns the list of created objects.
    """

    xmin,xmax,ymin,ymax = img_extent 
    print 'xmin,xmax,ymin,ymax = ', xmin,xmax,ymin,ymax

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
        obj_list.append(obj)

    return obj_list

#-----------------------------

def main_full_test():
    """Full test of the class DragRectangle, using the class DragObjectSet
       1. make a 2D plot
       2. make a list of random objects and add them to the plot
       3. use the class DragObjectSet to switch between modes for full test of the class DragRectangle
    """
    fig, axes, imsh = generate_test_image()
    list_of_objs = generate_list_of_objects(imsh.get_extent())

    t = DragObjectSet(fig, axes, DragRectangle, useKeyboard=True)
    t .set_list_of_objs(list_of_objs)

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
    obj = DragRectangle() # call W/O parameters => object will be initialized at first mouse click
    add_obj_to_axes(obj, axes, list_of_objs)

    plt.get_current_fig_manager().window.geometry('+50+10') # move(50, 10)
    plt.show()

#-----------------------------

if __name__ == "__main__" :

    #main_simple_test()
    main_full_test()
    sys.exit ('End of test')

#-----------------------------
