#!/usr/bin/env python
#----------------------------------

import sys
import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.lines   as lines
import math # cos(x), sin(x), radians(x), degrees()
from Drag import *

class DragCenter( Drag, lines.Line2D ) : # lines.Line2D ) : 

    def __init__(self, x=None, y=None, xerr=5, yerr=5, linewidth=1, color='b', picker=10, linestyle='-', str_of_pars=None) :

        Drag.__init__(self, linewidth, color, linestyle, my_type='Center')

        if str_of_pars is not None : 
            xc,yc,xe,ye,lw,col,s,t,r = self.parse_str_of_pars(str_of_pars)
            self.isSelected    = s
            self.myType        = t
            self.isRemoved     = r
            self.isInitialized = True
            xarr, yarr = self.get_xy_arrays_for_center_sign(xc,yx,xe,ye)
            lines.Line2D.__init__(self,  xarr,  yarr, linewidth=lw, color=col, picker=picker)

        elif x is None or y is None : # Default line initialization
            xc,yc=(10,10)
            xarr, yarr = self.get_xy_arrays_for_center_sign(xc,yc,xerr,yerr)
            lines.Line2D.__init__(self, xarr, yarr, linewidth=linewidth, color=color, picker=picker)
            self.isInitialized = False
            print "DragCenter initialization w/o parameters."

        else :
            #xc, yc = x, y
            xarr, yarr = self.get_xy_arrays_for_center_sign(x,y,xerr,yerr)
            lines.Line2D.__init__(self,  xarr,  yarr, linewidth=linewidth, color=color, picker=picker)
            self.isInitialized = True

        self.set_picker(picker)
        self.myPicker  = picker
        self.press     = None # Is used to transmit local information between press and release button


    def get_xy_arrays_for_center_sign(self,xc,yc,xerr,yerr) :      
        """Return arrays of X and Y polyline coordinates which define the shape of the center sign"""
        gapx, gapy = 0.1*xerr, 0.1*yerr
        xarr = (xc, xc+xerr, xc-xerr, xc-gapx, xc     , xc     )
        yarr = (yc, yc     , yc     , yc+gapy, yc+yerr, yc-yerr)
        return xarr, yarr


    def reset_center_position(self,x,y) :
        """Reset the center position keeping all other properties"""
        xc,yc,xerr,yerr,lw,col,s,t,r = self.get_list_of_pars()
        self.set_xy_arrays_for_center_sign(x,y,xerr,yerr)        


    def set_xy_arrays_for_center_sign(self,xc,yc,xerr,yerr) :      
        xarr, yarr = self.get_xy_arrays_for_center_sign(xc,yc,xerr,yerr) 
        #self.set_data(xarr, yarr)
        self.set_xdata(xarr)
        self.set_ydata(yarr)


    def get_xy_center(self) :      
        return int(self.get_xdata()[0]), int(self.get_ydata()[0])


    def get_list_of_pars(self) :
        xarr = self.get_xdata()
        yarr = self.get_ydata()
        xc   = int( xarr[0] )
        yc   = int( yarr[0] )
        xerr = int( xarr[1] - xarr[0] )
        yerr = int( yarr[0] - yarr[5] )
        lw   = int( self.get_linewidth() ) 
        #col  =      self.get_color() 
        col  = self.myCurrentColor
        s    = self.isSelected
        t    = self.myType
        r    = self.isRemoved
        return (xc,yc,xerr,yerr,lw,col,s,t,r)


    def parse_str_of_pars(self, str_of_pars) :
        pars = str_of_pars.split()
        #print 'pars:', pars
        t   = pars[0]
        xc  = float(pars[1])
        yc  = float(pars[2])
        xe  = float(pars[3])
        ye  = float(pars[4])
        lw  = int(pars[5])
        col = str(pars[6])
        s   = self.dicBool[pars[7].lower()]
        r   = self.dicBool[pars[8].lower()]
        return (xc,yc,xe,ye,lw,col,s,t,r)


    def get_str_of_pars(self) :
        xc,yc,xerr,yerr,lw,col,s,t,r = self.get_list_of_pars()
        return '%s %7.2f %7.2f %7.2f %7.2f %d %s %s %s' % (t,xc,yc,xerr,yerr,lw,col,s,r)


    def print_pars(self) :
        xc,yc,xerr,yerr,lw,col,s,t,r = self.get_list_of_pars()
        print 't,xc,yc,xerr,yerr,lw,col,s,t,r =', self.get_str_of_pars()


    def obj_contains_cursor(self, event): # Overrides method in Drag
        if not self.isInitialized   : return False
        if event.inaxes != self.axes: return False
        return self.my_contains(event)


    def my_contains(self, event):
        x,y = event.xdata, event.ydata
        xc,yc,xerr,yerr,lw,col,s,t,r = self.get_list_of_pars()
        r   = self.myPicker

        xmin = xc-xerr
        xmax = xc+xerr
        ymin = yc-yerr
        ymax = yc+yerr

        if x > xmin-r and x < xmax+r and y > yc-r and y < yc+r :
            self.inHorBox = True
        else :
            self.inHorBox  = False

        if x > xc-r and x < xc+r and y > ymin-r and y < ymax+r :
            self.inVerBox = True
        else :
            self.inVerBox = False

        print 'self.inHorBox =',self.inHorBox ,'  self.inVerBox =', self.inVerBox

        if self.inHorBox or self.inVerBox:
            return True
        else :
            return False


    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.axes: return

        clickxy = event.xdata, event.ydata
        #print 'clickxy =',clickxy 

        xc,yc,xerr,yerr,lw,col,s,t,r = self.get_list_of_pars()

        if self.isInitialized :
            #contains, attrd = self.contains(event)
            contains = self.my_contains(event)
            if not contains: return
            #print 'event contains object',

            xarr = self.get_xdata()
            yarr = self.get_ydata()

            vertindex = 0 # assumes that the click xy is close to center
            ind       = 0

            self.dist_min = 1000
            for vertex in (1,2,4,5) :
                ind += 1
                dist = self.max_deviation(clickxy,(xarr[vertex],yarr[vertex]))
                if dist < self.dist_min : # self.myPicker
                    vertindex = ind #vertex
                    self.dist_min = dist

            print 'vertindex=', vertindex

            self.press = xc,yc,xerr,yerr, clickxy, vertindex

            #----Remove object at click on middle mouse botton
            if event.button is 2 : # for middle mouse button
                self.remove_object_from_img() # Remove object from image
                return

        else : # if the object position is not defined yet:
            vertindex = 0
            self.press = xc,yc,xerr,yerr, clickxy, vertindex

        self.on_press_graphic_manipulations()


    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.axes: return

        #print 'event on_moution', self.get_xydata()
        currentxy = event.xdata, event.ydata

        xc, yc, xerr, yerr, clickxy, vertindex = self.press
        xcnew, ycnew, xenew, yenew = xc, yc, xerr, yerr

        dx = currentxy[0]-clickxy[0]
        dy = currentxy[1]-clickxy[1] 


        if event.button is 3 and self.isInitialized : # move
            xcnew, ycnew = xc + dx, yc + dy
            self.set_xy_arrays_for_center_sign(xcnew,ycnew,abs(xenew),abs(yenew))


        elif event.button is 1 and self.isInitialized : # for left mouse button

            if   vertindex == 1 :
                xenew = xerr + dx

            elif vertindex == 2 :
                xenew = xerr - dx

            elif vertindex == 3 :
                yenew = yerr + dy

            elif vertindex == 4 :
                yenew = yerr - dy
            else : # vertindex == 0 -is center, move 
                xcnew, ycnew = xc + dx, yc + dy

            # protection
            min_err_size = 5
            if xenew < min_err_size : xenew = min_err_size
            if yenew < min_err_size : yenew = min_err_size

            self.set_xy_arrays_for_center_sign(xcnew,ycnew,abs(xenew),abs(yenew))

        elif event.button is 1 and not self.isInitialized : # Initialization
            self.set_xy_arrays_for_center_sign(clickxy[0],clickxy[1],abs(dx),abs(dy))

        self.on_motion_graphic_manipulations()


    def on_release(self, event):
        'on release we reset the press data'
        self.xc, self.yc = self.figure.my_xyc = self.get_xy_center()
        #print 'Set new center coordinates xc, yc=' + str(self.xc) + ', ' +  str(self.yc) 
        #self.emit( QtCore.SIGNAL('new_xy_center(int,int)'), self.xc, self.yc)        
        self.on_release_graphic_manipulations()
        #if self.press is not None : self.print_pars()
        self.press = None

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

    nobj = 5
    x = xmin+(xmax-xmin)*np.random.rand(nobj)
    y = ymax+(ymin-ymax)*np.random.rand(nobj)
    xe = (xmax-xmin)/3*np.random.rand(nobj)
    ye = (ymin-ymax)/3*np.random.rand(nobj)
    #print ' x=',x, ' y=',y, ' w=',w, ' h=',h

    obj_list = []

    # Add objects with initialization through the parameters
    for indobj in range(nobj) :
        obj = DragCenter(x[indobj], y[indobj], xe[indobj], ye[indobj], color='g')
        obj_list.append(obj)

    return obj_list

#-----------------------------

def main_full_test():
    """Full test of the class DragCenter, using the class DragObjectSet
       1. make a 2-d plot
       2. make a list of random objects and add them to the plot
       3. use the class DragObjectSetr to switch between modes for full test of the class DragCenter
    """
    fig, axes, imsh = generate_test_image()
    list_of_objs = generate_list_of_objects(imsh.get_extent())

    t = DragObjectSet(fig, axes, DragCenter, useKeyboard=True, is_single_obj=True)
    t .set_list_of_objs(list_of_objs)

    plt.get_current_fig_manager().window.geometry('+50+10') # move(50, 10)
    plt.show()

#-----------------------------

def main_simple_test():
    """Simple test of the class DragCenter.
       1. make a 2-d plot
       2. make a list of random objects and add them to the plot
       3. add one more object with initialization at 1st click-and-drag of mouse-left button
    """
    fig, axes, imsh = generate_test_image()
    list_of_objs = generate_list_of_objects(imsh.get_extent())

    #Add one more object
    obj = DragCenter() # call W/O parameters => object will be initialized at first mouse click
    add_obj_to_axes(obj, axes, list_of_objs)

    plt.get_current_fig_manager().window.geometry('+50+10') # move(50, 10)
    plt.show()

#-----------------------------

if __name__ == "__main__" :

    #main_simple_test()
    main_full_test()
    sys.exit ('End of test')

#-----------------------------
