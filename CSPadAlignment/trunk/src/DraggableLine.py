#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module DraggableLine...
#
#------------------------------------------------------------------------

"""Module DraggableLine for CSPadAlignment package

CSPadAlignment package is intended to check quality of the CSPad alignment
using image of wires illuminated by flat field.
Shadow of wires are compared with a set of straight lines, which can be
interactively adjusted.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""
#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$
#------------------------------
#!/usr/bin/env python
#----------------------------------

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.lines  as lines
import math # cos(x), sin(x), radians(x), degrees()
#import copy
import ImageParameters   as imp
import Draw              as drw


class DraggableLine:
    def __init__(self, line, canvas=None):
        self.line = line
        self.press = None
        self.line.set_pickradius(5)
        self.is_picked = False

        if canvas == None : self.canvas = self.line.figure.canvas
        else              : self.canvas = canvas


    def connect(self):
        'connect to all the events we need'
        self.cidpick    = self.canvas.mpl_connect('pick_event',           self.on_pick)
        self.cidpress   = self.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cidrelease = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion  = self.canvas.mpl_connect('motion_notify_event',  self.on_motion)


    def disconnect(self):
        'disconnect all the stored connection ids'
        self.canvas.mpl_disconnect(self.cidpress)
        self.canvas.mpl_disconnect(self.cidrelease)
        self.canvas.mpl_disconnect(self.cidmotion)


    def distance(self,xy1,xy2) :
        dx = xy2[0] - xy1[0]
        dy = xy2[1] - xy1[1]
        return math.sqrt(dx*dx+dy*dy)


    def on_pick(self, event):
        'on button press we will see if the mouse is over us and store some data'

        picked_line = event.artist

        if picked_line != self.line :
            self.is_picked = False
            return
        else :
            self.is_picked = True
        
        imp.impars.selected_line_index = self.line_index
        print 'on_pick : imp.impars.selected_line_index =', imp.impars.selected_line_index


    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'

        if not self.is_picked : return

        if self.line_index != imp.impars.selected_line_index : return
        #if event.inaxes != self.line.axes: return

        contains, attrd = self.line.contains(event)
        if not contains: return

        xy0 = self.line.get_xydata()
        #print 'event contains: xy0[0], xy0[1]', xy0[0], xy0[1]

        clickxy = event.xdata, event.ydata
        #print 'clickxy =',clickxy 

        if self.distance(clickxy,xy0[0]) < self.distance(clickxy,xy0[1]) :
            vertindex = 0
        else :
            vertindex = 1

        #print 'vertindex = ', vertindex

        self.press = xy0, clickxy, vertindex
        self.line_original_color = self.line.get_color()
        self.line.set_color('r')
        self.canvas.draw()


    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'

        if not self.is_picked : return
        if self.press is None: return
        if self.line_index != imp.impars.selected_line_index : return
        #if event.inaxes != self.line.axes: return

        #print 'event on_moution', self.line.get_xydata()
        currentxy = event.xdata, event.ydata
        #print 'self.onmoutionxy =',currentxy 

        xy0, clickxy, vertindex = self.press
        dx = currentxy[0] - clickxy[0]
        dy = currentxy[1] - clickxy[1]

        xy = xy0.copy() # copy

        #print 'xy0=', xy0
        #print 'xy =', xy 

        xy[vertindex][0] += dx
        xy[vertindex][1] += dy

        self.line.set_data([[xy[0][0], xy[1][0]],[xy[0][1], xy[1][1]]])
        self.canvas.draw()


    def on_release(self, event):
        'on release we reset the press data'
        if not self.is_picked : return
        if self.line_index != imp.impars.selected_line_index : return
        #if event.inaxes != self.line.axes: return
        imp.impars.selected_line_index = None
        self.press = None
        self.line.set_color(self.line_original_color)
        self.update_line_coordinates()
        self.canvas.draw()
        drw.draw.plotForMovedLine(self.line_index)


    def update_line_coordinates(self):
        lind = self.line_index
        x12, y12 = self.line.get_data()
        orient = imp.impars.line_coord[lind][4]
        imp.impars.line_coord[lind] = [int(x12[0]), int(x12[1]), int(y12[0]), int(y12[1]), orient]
        print 'line', lind, imp.impars.line_coord[lind]
        
#-----------------------------

def main():

    fig = plt.figure(num=1, figsize=(12,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    ax = fig.add_subplot(111)

    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=2400)
    arr.shape = (40,60)

    axisImage = ax.imshow(arr, origin='upper', interpolation='nearest', aspect='auto')#, extent=self.range
    #axisImage.set_clim(zmin,zmax)
    mycolbar = fig.colorbar(axisImage, pad=0.1, fraction=0.15, shrink=1.0, aspect=15, orientation=1)#, ticks=coltickslocs) #orientation=1,

    xmin,xmax,ymin,ymax =  axisImage.get_extent() 
    #print 'xmin,xmax,ymin,ymax = ', xmin,xmax,ymin,ymax

    nlines = 10
    x = (xmin,xmax)
    y = ymin+(ymax-ymin)*np.random.rand(nlines,2)
    #print ' y=',y

    dls = []

    for indline in range(nlines) :
        line = lines.Line2D(x, y[indline], linewidth=1, color='w', picker=5)
        ax.add_line(line) 
        dl = DraggableLine(line)
        dl.connect()
        dls.append(dl)

    plt.show()

#-----------------------------
if __name__ == "__main__" :
    main()
#-----------------------------
