#!/usr/bin/env python
#----------------------------------

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


    #def set_dragged_obj_properties(self):
    #    self.set_color    ('k')
    #    self.set_linewidth(1)
    #    self.set_linestyle('--') #'--', ':'


    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.axes: return

        clickxy = event.xdata, event.ydata
        #print 'clickxy =',clickxy 

        if self.isInitialized :
            contains, attrd = self.contains(event)
            if not contains: return

            xy0 = self.get_xydata()
            #print 'event contains: xy0[0], xy0[1]', xy0[0], xy0[1]
            if self.distance(clickxy,xy0[0]) < self.distance(clickxy,xy0[1]) :
                vertindex = 0
            else :
                vertindex = 1
            self.press = xy0, clickxy, vertindex

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
        self.press = None
        self.on_release_graphic_manipulations()


#-----------------------------
# Test
#-----------------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=2400)
    #arr = np.arange(2400)
    arr.shape = (40,60)
    return arr

#-----------------------------

def main():

    fig = plt.figure()
    ax = fig.add_subplot(111)

    axisImage = ax.imshow(get_array2d_for_test(), origin='upper', interpolation='nearest', aspect='auto')#, extent=self.range
    #axisImage.set_clim(zmin,zmax)
    mycolbar = fig.colorbar(axisImage, pad=0.1, fraction=0.15, shrink=1.0, aspect=15, orientation=1)#, ticks=coltickslocs) #orientation=1,

    xmin,xmax,ymin,ymax =  axisImage.get_extent() 
    #print 'xmin,xmax,ymin,ymax = ', xmin,xmax,ymin,ymax

    nlines = 10
    x = (xmin,xmax)
    y = ymin+(ymax-ymin)*np.random.rand(nlines,2)

    dls = []
    # Add lines with initialization through the parameters
    for indline in range(nlines) :
        dl = DragLine(x, y[indline], linewidth=2, color='g', picker=5, linestyle='-')
        #ax.add_line(dl) 
        #ax.add_artist(dl) 
        dl.add_to_axes(ax)
        dl.connect()
        dls.append(dl)

    # Add line with mouse initialization

    dl = DragLine() # W/O parameters !
    dl.add_to_axes(ax)
    dl.connect()
    dls.append(dl)
        
    plt.show()

#-----------------------------
if __name__ == "__main__" :
    main()
#-----------------------------
