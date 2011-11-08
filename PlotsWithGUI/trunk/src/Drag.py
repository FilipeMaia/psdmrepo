#!/usr/bin/env python
#----------------------------------

#import numpy as np
#import copy
#import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import math # cos(x), sin(x), radians(x), degrees()
#import copy

class Drag :

    def __init__(self, linewidth=2, color='k', linestyle='solid') :
        """
        Actual initialization is done by the method add_to_axes(...)
        """        
        self.isInitialized = False

        self.myColor  = color
        self.myWidth  = linewidth
        self.myStyle  = linestyle

    def add_to_axes(self, axes) :
        axes.add_artist(self)
        #self.axes = axes # is defined in add_artist(self)


    def set_isInitialized(self, state) :
        self.isInitialized = state


    def get_isInitialized(self) :
        return self.isInitialized


    def remove_from_axes(self) :
        #self.axes.patches.remove(self)
        #self.remove()
        #self.figure.canvas.draw()
        pass


    def connect(self):
        'connect to all the events we need'
        self.cidpress   = self.figure.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion  = self.figure.canvas.mpl_connect('motion_notify_event',  self.on_motion)


    def disconnect(self):
        'disconnect all the stored connection ids'
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figure.canvas.mpl_disconnect(self.cidmotion)


    def distance(self,xy1,xy2) :
        dx = xy2[0] - xy1[0]
        dy = xy2[1] - xy1[1]
        return math.sqrt(dx*dx+dy*dy)


    def max_deviation(self,xy1,xy2) :
        dx = abs(xy2[0] - xy1[0])
        dy = abs(xy2[1] - xy1[1])
        return max(dx,dy)


    def save_obj_properties(self):
        #print 'save_line_properties',
        #self.myColor = self.get_color()
        #self.myWidth = self.get_linewidth()
        self.myStyle = self.get_linestyle()
        #print 'Save Color, Width, Style =', self.myColor, self.myWidth, self.myStyle 
        pass


    def restore_obj_properties(self):
        #print 'restore_line_properties'
        #print 'Save Color, Width, Style =', self.myColor, self.myWidth, self.myStyle 
        self.set_color    (self.myColor)
        self.set_linewidth(self.myWidth)
        self.set_linestyle(self.myStyle)


    def set_dragged_obj_properties(self):
        self.set_color    ('k')
        self.set_linewidth(1)
        self.set_linestyle('dashed')


    def on_press_graphic_manipulations(self) :
        self.save_obj_properties()              # Save original line properties
        self.set_linewidth(0)                   # Makes the line invisible
        #self.set_linestyle('')                 # Draw nothing - makes the line invisible
        self.figure.canvas.draw()               # Draw canvas and save the background 
        self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox) # Added for blit

        self.set_dragged_obj_properties()      # Set the line properties during dragging 
        self.axes.draw_artist(self)             # Re-draw the line before any motion
        self.figure.canvas.blit(self.axes.bbox) # blit canvas before any motion


    def on_motion_graphic_manipulations(self) :
        #self.figure.canvas.draw()                         # Regular case
        self.figure.canvas.restore_region(self.background) # Added for blit
        self.axes.draw_artist(self)                        # Added for blit
        self.figure.canvas.blit(self.axes.bbox)            # Added for blit


    def on_release_graphic_manipulations(self) :
        'on release we reset the press data'
        self.restore_obj_properties()
        self.axes.draw_artist(self)                        # Added for blit
        self.figure.canvas.blit(self.axes.bbox)            # Added for blit
        self.isInitialized = True

        #self.figure.canvas.draw()

#-----------------------------
if __name__ == "__main__" :
    sys.exit ('This module does not have main()')
#-----------------------------
