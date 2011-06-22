#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GraphicalWindow...
#
#------------------------------------------------------------------------

"""Module GraphicalWindowfor CSPadAlignment package

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

#--------------------------------
#  Imports of standard modules --
#--------------------------------
#import sys
#import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from   matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from   matplotlib.figure import Figure
import matplotlib.lines  as lines

import ImageParameters   as imp
#import GlobalMethods     as gm
import DraggableLine     as dragline
#---------------------
#  Class definition --
#---------------------

class GraphicalWindow :
    """Example of the plot for CSpad alignment project."""

    def __init__(self, fig, arr=None):
        self.arr = arr        
        self.fig = fig
        self.fig.myZmin = None
        self.fig.myZmax = None
        self.fig.myZmin = 1000
        self.fig.myZmax = 4000
        self.canvas = fig.canvas

        print 'self.arr:\n',self.arr
        #self.on_draw()
        if self.arr != None : self.set_image_array(self.arr)


    def set_image_array(self,arr):
        self.arr = arr
        self.on_draw()


    def on_draw(self):
        self.draw_CSPad_image_and_colorbar()
        self.draw_draggable_lines()
        #self.canvas.draw() # already done in self.draw_draggable_lines()
        #self.canvas.savefig('plot-image-' + imp.impars.plot_fname_suffix + '.png')


    def draw_CSPad_image_and_colorbar(self):
        self.fig.clear()
        self.fig.myaxesI = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.07, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
        Range = (imp.impars.ymin, imp.impars.ymax, imp.impars.xmax, imp.impars.xmin)
        #Range = None
        self.fig.myaxesImage = self.fig.myaxesI.imshow(self.arr, origin='upper', interpolation='nearest', aspect='auto',extent=Range)
        self.fig.mycolbar = self.fig.colorbar(self.fig.myaxesImage, pad=0.03, fraction=0.04, shrink=1.0, aspect=40, orientation=1)
        self.fig.myaxesImage.set_clim(self.fig.myZmin,self.fig.myZmax)

        #self.connect()


    def connect(self):
        'connect image and colorbar axes to mouse buttons'
        self.cidpress  = self.canvas.mpl_connect('button_press_event',   self.processMouseButtonPress)
        #self.cidrelese = self.canvas.mpl_connect('button_press_event',   self.processMouseButtonRelease)


    def disconnect(self):
        'disconnect image and colorbar axes from mouse buttons'
        self.canvas.mpl_disconnect(self.cidpress)
        #self.canvas.mpl_disconnect(self.cidrelese)


    def processMouseButtonPress(self, event) :
        print 'MouseButtonPress'
        self.fig = event.canvas.figure
        if event.inaxes == self.fig.mycolbar.ax : self.mousePressOnColorBar (event)
        #if event.inaxes == self.fig.myaxesI     : self.mousePressOnImage    (event)


    def mousePressOnImage(self, event) :
        print 'Image'


    def mousePressOnColorBar(self, event) :
        print 'Color bar'
        lims = self.fig.myaxesImage.get_clim()
        colmin = lims[0]
        colmax = lims[1]
        range = colmax - colmin
        value = colmin + event.xdata * range
        self.setColorLimits(event, colmin, colmax, value)


    def setColorLimits(self, event, colmin, colmax, value) :
        print colmin, colmax, value

        # left button
        if event.button is 1 :
            if value > colmin and value < colmax :
                colmin = value
                print "New mininum: ", colmin

        # middle button
        elif event.button is 2 :
            #colmin, colmax = self.getImageAmpLimitsFromWindowParameters()
            print 'Reset fig' # ,fig.number #, fig.nwin 
            colmin = None
            colmax = None

        # right button
        elif event.button is 3 :
            if value > colmin and value < colmax :
                colmax = value
                print "New maximum: ", colmax

        self.fig.myZmin = colmin
        self.fig.myZmax = colmax

        self.disconnect()
        self.disconnect_all_lines()
        self.on_draw()



    def draw_draggable_lines(self):
        self.dls = []
        for lind in range(len(imp.impars.line_coord)) :
            x = imp.impars.line_coord[lind][0:2]
            y = imp.impars.line_coord[lind][2:4] 
            if  imp.impars.line_coord[lind][4] == 'p' :
                self.oneline = lines.Line2D(x, y, linewidth=3, color='c', linestyle='--', picker=5)
            else :
                self.oneline = lines.Line2D(x, y, linewidth=1, color='w', picker=5)

            self.fig.myaxesI.add_line(self.oneline) 
            dl = dragline.DraggableLine(self.oneline,self.canvas)
            dl.line_index = lind
            dl.connect()
            self.dls.append(dl)


    def disconnect_all_lines(self):
        for dl in self.dls :
            dl.disconnect()


    def update_line_coordinates(self):
        lind=-1
        for dl in self.dls :
            lind+=1
            x12, y12 = dl.line.get_data()
            imp.impars.line_coord[lind] = int(x12[0]), int(x12[1]), int(y12[0]), int(y12[1])
            print 'line', lind, imp.impars.line_coord[lind]

#-----------------------------

def main():

    #plt.ion()

    mu, sigma = 2000, 200
    arr = mu + sigma*np.random.standard_normal(size=2400)
    arr.shape = (40,60)

    # Figure - does not have canvas !!!!
    # figure - already has a canvas

    #fig = Figure((5.0, 10.0), dpi=100, facecolor='w',edgecolor='w',frameon=True)
    #fig.canvas = FigureCanvas(fig)
    fig = plt.figure(num=1, figsize=(12,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)

    gw = GraphicalWindow(fig, arr)
    
    #plt.ioff()
    #fig.canvas.print_figure('test.png')

    fig.canvas.show()
    plt.show()

#-----------------------------
if __name__ == "__main__" :
    main()
#-----------------------------
