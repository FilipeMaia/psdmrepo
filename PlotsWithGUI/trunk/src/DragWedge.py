#!/usr/bin/env python
#----------------------------------
import sys

import numpy as np
#import copy
import matplotlib.pyplot  as plt
import matplotlib.lines   as lines

import math # cos(x), sin(x), radians(x), degrees(), atan2(y,x)
#import copy
#import FastArrayTransformation as fat

#import Drag as drag
from Drag import *


class DragWedge( Drag, lines.Line2D ) :  #patches.CirclePolygon

    def __init__(self, xy=None, radius=10, theta1=0, theta2=70, width=15, linewidth=2, linestyle='solid', color='b', picker=5) :
        """Draw a wedge centered at x, y center with radius r that sweeps theta1 to theta2 (in degrees) in positive angle direction.
        If width is given, then a partial wedge is drawn from inner radius r - width to outer radius r."""
        Drag.__init__(self, linewidth, color, linestyle)

        if  xy == None : # Default line initialization
            xc,yc=(10,10)
            self.isInitialized = False
        else :
            xc,yc=xy
            self.isInitialized = True

        self.set_wedge_pars(xy, radius, width, theta1, theta2)
        xarr, yarr = self.get_xy_arrays_for_wedge(xc,yc,radius,width,theta1,theta2)
        lines.Line2D.__init__(self, xarr, yarr, linewidth=linewidth, color=color, picker=picker)

        self.set_picker(picker)
        self.myPicker = picker
        self.press    = None


    def set_wedge_pars(self, xy, radius, width, theta1, theta2) :
        self.center = xy
        self.radius = radius
        self.width  = width
        self.theta1 = theta1
        self.theta2 = theta2
        #self.set_xy_arrays_for_wedge()


    def set_center(self, center) :
        self.center = center
        self.set_xy_arrays_for_wedge()


    def set_center(self, radius) :
        self.radius = radius
        self.set_xy_arrays_for_wedge()


    def set_width(self, width) :
        self.width = width
        self.set_xy_arrays_for_wedge()


    def set_theta1(self, theta1) :
        self.theta1 = theta1
        self.set_xy_arrays_for_wedge()


    def set_theta2(self, theta2) :
        self.theta2 = theta2
        self.set_xy_arrays_for_wedge()


    def set_xy_arrays_for_wedge(self) :
        xc,yc,radius,width,theta1,theta2 = self.center[0],self.center[1],self.radius,self.width,self.theta1,self.theta2
        xarr, yarr = self.get_xy_arrays_for_wedge(xc,yc,radius,width,theta1,theta2)

        self.set_xdata(xarr)
        self.set_ydata(yarr)


    def get_xy_arrays_for_wedge(self,xc,yc,radius,width,theta1,theta2) :      
        """Return arrays of X and Y polyline coordinates which define the shape of the wedge"""

        t1_rad = math.radians(theta1)
        t2_rad = math.radians(theta2)

        Npoints = int(abs(theta2-theta1)/3)+1

        TarrF = np.linspace(t1_rad, t2_rad, num=Npoints, endpoint=True)
        TarrB = np.array(np.flipud(TarrF))

        rmin = radius - width
        xarrF = xc + radius * np.cos(TarrF)
        yarrF = yc + radius * np.sin(TarrF)
        xarrB = xc + rmin   * np.cos(TarrB)
        yarrB = yc + rmin   * np.sin(TarrB)

        xarr  = np.hstack([xarrF,xarrB,xarrF[0]])
        yarr  = np.hstack([yarrF,yarrB,yarrF[0]])

        return xarr, yarr #.flatten(), yarr.flatten()


    def get_list_of_wedge_pars(self) :
        x,y = self.center
        r   =      self.radius
        w   =      self.width
        t1  =      self.theta1
        t2  =      self.theta2
        lw  = int( self.get_linewidth() ) 
        col =      self.get_color() 
        s   =      self.isSelected
        t   =      self.myType
        rem =      self.isRemoved
        return (x,y,r,w,t1,t2,lw,col,s,t,rem)


    def print_pars(self) :
        x,y,r,w,t1,t2,lw,col,s,t,rem = self.get_list_of_wedge_pars()
        print 'x,y,r,w,t1,t2,lw,col,s,t,rem =', x,y,r,w,t1,t2,lw,col,s,t,rem


    def bring_theta_in_range(self, theta, range=(-180, 180) ) :
        """Recursive method which brings the input theta in the specified range.
           The default range value corresponds to the range of math.atan2(y,x) : [-pi, pi]
        """
        theta_min, theta_max = range
        theta_corr = theta
        if   theta <  theta_min : theta_corr += 360            
        elif theta >= theta_max : theta_corr -= 360
        else : return theta
        return self.bring_theta_in_range(theta_corr)


    def my_contains(self, click_r, click_theta, theta1, theta2, dt):
        x,y,r,w,t1,t2,lw,col,s,t,rem = self.get_list_of_wedge_pars()
        psize = self.myPicker

        #dt = math.degrees( psize / r ) # picker size (degree) in angular direction
        #click_r = self.distance( clickxy, (x,y) )
        #click_theta = math.degrees( math.atan2(click_y-y, click_x-x) ) # Click angle in the range [-180,180]
        #theta1 = self.bring_theta_in_range(t1)
        #theta2 = self.bring_theta_in_range(t2)

        # Check for RADIAL direction
        if click_r < (r + psize) and click_r > (r - w - psize) : 
            self.inWideRing = True
        else :
            self.inWideRing = False
            
        if click_r < (r - psize) and click_r > (r - w + psize) : 
            self.inNarrowRing = True
        else :
            self.inNarrowRing = False

        # Check for ANGULAR direction
        if abs(t2-t1) > 360 :
            self.isRing = True
            if self.inWideRing and not self.inNarrowRing : return True
            else                                         : return False
        else :
            self.isRing = False

        if theta2 > theta1 : # normal, positive direction of the arc

            if click_theta > theta1 - dt and click_theta < theta2 + dt :
                self.inWideSector = True
            else :
                self.inWideSector = False
                
            if click_theta > theta1 + dt and click_theta < theta2 - dt :
                self.inNarrowSector = True
            else :
                self.inNarrowSector = False
                
        else :   # opposite, negative direction of the arc 

            if click_theta > theta1 - dt or click_theta < theta2 + dt :
                self.inWideSector = True
            else :
                self.inWideSector = False
                
            if click_theta > theta1 + dt or click_theta < theta2 - dt :
                self.inNarrowSector = True
            else :
                self.inNarrowSector = False

        if self.inWideRing and self.inWideSector and not (self.inNarrowRing and self.inNarrowSector) :
                return True

        return False


    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.axes: return

        clickxy = click_x, click_y = event.xdata, event.ydata

        if self.isInitialized :

            x,y,r,w,t1,t2,lw,col,s,t,rem = self.get_list_of_wedge_pars()
            psize = self.myPicker
            dt = math.degrees( psize / r ) # picker size (degree) in angular direction
            
            click_r = self.distance( clickxy, (x,y) )
            click_theta = math.degrees( math.atan2(click_y-y, click_x-x) ) # Click angle in the range [-180,180]
            theta1 = self.bring_theta_in_range(t1)
            theta2 = self.bring_theta_in_range(t2)
            
            print 'x,y,r,w,t1,t2,dt,click_theta,_r = ', x,y,r,w,t1,t2,dt, click_theta, click_r

            if not self.my_contains(click_r, click_theta, theta1, theta2, dt) : return
            #if not self.contains(event) : return

            #self.set_center(clickxy) 

            r0   = self.radius
            w0   = self.width
            xy0  = self.center

            clkAtRmin   = False
            clkAtRmax   = False
            clkAtTheta1 = False
            clkAtTheta2 = False

            if abs(click_r - r0)      < psize : clkAtRmax = True                
            if abs(click_r - (r0-w0)) < psize : clkAtRmin = True
            if click_theta > theta1-dt and click_theta < theta1+dt : clkAtTheta1 = True
            if click_theta > theta2-dt and click_theta < theta2+dt : clkAtTheta2 = True

# Numeration in vertindex the vertices and sides of the wedge
#
#        t1                t2
#        
# Rmin   3--------8--------4
#        |                 |
#        |                 |
#        5                 6
#        |                 |
#        |                 |
# Rmax   1--------7--------2

            vertindex = 0  # undefined 
            if   clkAtRmax :
                vertindex = 7                
                if clkAtTheta1 : vertindex = 1
                if clkAtTheta2 : vertindex = 2

            elif clkAtRmin :
                vertindex = 8                
                if clkAtTheta1 : vertindex = 3
                if clkAtTheta2 : vertindex = 4

            elif clkAtTheta1   : vertindex = 5 
            elif clkAtTheta2   : vertindex = 6 

            print 'vertindex= ',vertindex

            self.press = xy0, clickxy, click_r, click_theta, r0, w0, theta1, theta2, vertindex

            #----Remove object at click on middle mouse botton
            if event.button is 2 : # for middle mouse button
                self.remove_object_from_img() # Remove object from image
                return

        else : # if the object position is not defined yet:

            vertindex = 1
            x0,y0 = xy0 = self.center = (40,60)
            click_r = self.distance( clickxy, xy0 )
            click_theta = math.degrees( math.atan2(click_y-y0, click_x-x0) ) # Click angle in the range [-180,180]
            theta1 = click_theta
            theta2 = theta1 + 1 

            self.center = xy0
            self.radius = click_r
            self.theta1 = click_theta

            self.press  = xy0, clickxy, click_r, click_theta, self.radius, self.width, theta1, theta2, vertindex

        self.on_press_graphic_manipulations()


    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.axes: return

        #print 'event on_moution', self.get_xydata()
        current_x, current_y = current_xy = event.xdata , event.ydata

        xy0, click_xy, click_r, click_t, radius, width, theta1, theta2, vertindex = self.press
        x0, y0 = xy0
        click_x, click_y = click_xy

        current_r = self.distance( current_xy, xy0 )
        current_t = math.degrees( math.atan2(current_y-y0, current_x-x0) ) # Click angle in the range [-180,180]

        dx = current_x - click_x
        dy = current_y - click_y
        dr = current_r - click_r
        dt = current_t - click_t

        if self.isInitialized : # for left mouse button

            if   vertindex == 0 : #side
                print 'WORNING, NON-POSSIBLE vertindex =', vertindex

            elif vertindex == 1 :
                self.theta1 = theta1 + dt
                self.radius = radius + dr
                self.width  = width  + dr

            elif vertindex == 2 :
                self.theta2 = theta2 + dt
                self.radius = radius + dr
                self.width  = width  + dr

            elif vertindex == 3 :
                self.theta1 = theta1 + dt
                self.width  = width  - dr

            elif vertindex == 4 :
                self.theta2 = theta2 + dt
                self.width  = width  - dr

            elif vertindex == 5 :
                self.theta1 = theta1 + dt

            elif vertindex == 6 :
                self.theta2 = theta2 + dt

            elif vertindex == 7 :
                self.radius = radius + dr
                self.width  = width  + dr

            elif vertindex == 8 :
                self.width  = width  - dr

        else :
            print 'current_r, current_theta =', current_r, current_theta
            self.width  = click_r - current_r
            self.theta2 = current_t

        self.set_xy_arrays_for_wedge()

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

class TestDragWedge(TestDragObject) : 

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
            obj = DragWedge() # Creates the DragRectangle object with 1st vertex in xy
            #obj = DragWedge(xy) # Creates the DragRectangle object with 1st vertex in xy
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

    nobj = 5
    x = xmin+(xmax-xmin)*np.random.rand(nobj)
    y = ymin+(ymax-ymin)*np.random.rand(nobj)
    r = (ymin-ymax)/3*np.random.rand(nobj) + 30

    obj_list = []
    # Add objects with initialization through the parameters
    for indobj in range(nobj) :
        obj = DragWedge((x[indobj], y[indobj]), radius=r[indobj], color='g')
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

    t = TestDragWedge(fig, axes)
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
    obj = DragWedge() # call W/O parameters => object will be initialized at first mouse click
    add_obj_to_axes(obj, axes, list_of_objs)

    plt.get_current_fig_manager().window.move(50, 10)
    plt.show()

#-----------------------------

if __name__ == "__main__" :

    #main_simple_test()
    main_full_test()
    sys.exit ('End of test')

#-----------------------------
