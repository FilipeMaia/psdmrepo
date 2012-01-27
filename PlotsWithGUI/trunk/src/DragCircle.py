#!/usr/bin/env python
#----------------------------------

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

            if event.button is 2 : # for middle mouse button
                self.remove_from_axes()

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
        self.press = None
        self.on_release_graphic_manipulations()


#-----------------------------
# Test
#-----------------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=13500)
    #arr = np.arange(2400)
    arr.shape = (90,150)
    return arr

#-----------------------------

def main():

    fig = plt.figure()
    fig.my_mode = None
    ax = fig.add_subplot(111)

    axesImage = ax.imshow(get_array2d_for_test(), origin='upper', interpolation='nearest', aspect='auto')#, extent=self.range
    #axisImage.set_clim(zmin,zmax)
    mycolbar = fig.colorbar(axesImage, pad=0.1, fraction=0.15, shrink=1.0, aspect=15, orientation=1)#, ticks=coltickslocs) #orientation=1,
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
        #ax.add_artist(obj) 
        obj.add_to_axes(ax)
        obj.connect()
        obj_list.append(obj)

    # Add one more object with mouse initialization
    obj = DragCircle() # W/O parameters !
    obj.add_to_axes(ax)
    obj.connect()
    obj_list.append(obj)

    plt.get_current_fig_manager().window.move(50, 10)    
    plt.show()

#-----------------------------
if __name__ == "__main__" :
    main()
#-----------------------------
