#!/usr/bin/env python
#----------------------------------

#import numpy as np
#import copy
import matplotlib.patches as patches
import math # cos(x), sin(x), radians(x), degrees()
#import copy

import ImgFigureManager    as imgfm

class Drag :

    def __init__(self, linewidth=2, color='k', linestyle='solid') :
        """
        Actual initialization is done by the method add_to_axes(...)
        """        
        self.isInitialized = False

        self.myColor     = color
        self.myWidth     = linewidth
        self.myStyle     = linestyle
        self.isSelected  = False
        self.isRemoved   = False
        self.fig_outside = None
        self.set_needs_in_redraw(False)         

        self.modeRemove  = 'Remove' # Should be the same as icp.modeRemove
        self.modeSelect  = 'Select'

        self.myType      = None #?????????????


    def add_to_axes(self, axes) :
        axes.add_artist(self)
        #self.axes = axes # is defined in add_artist(self)


    def set_fig_outside(self, fig) :
        self.fig_outside = fig


    def get_fig_outside(self) :
        return self.fig_outside


    def set_isInitialized(self, state) :
        self.isInitialized = state


    def get_isInitialized(self) :
        return self.isInitialized


    #def remove_from_axes(self) :
        ##self.axes.patches.remove(self)
        ##self.remove()
        ##self.figure.canvas.draw()
        #pass
        #self.remove_object_from_img()


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


    def on_press_remove(self) :
        if self.get_mode() == self.modeRemove :
            self.isRemoved = True
        else :
            self.isRemoved = False
        #print 'Drag: on_press_remove(), self.get_mode() =', self.get_mode(), ' isRemoved =', self.isRemoved


    def on_release_remove(self) :
        if self.isRemoved :
            #print 'Drag: on_release_remove()' # ,' fig_number =', self.fig_outside.number
            #
            # NEEDS IN CALBACK SIGNAL FOR REMOVE ACTION IN ORDER TO :
            # 1. REMOVE OBJECT FROM EXTERNAL LIST
            # 2. REMOVE ASSOCIATED OBJECTS 
            # imgfm.ifm.close_fig(self.fig_outside.number)

            self.disconnect()


    def save_obj_properties(self) :
        #print 'save_line_properties',
        #self.myColor = self.get_color()
        #self.myWidth = self.get_linewidth()
        self.myStyle = self.get_linestyle()
        #print 'Save Color, Width, Style =', self.myColor, self.myWidth, self.myStyle 
        pass


    def restore_obj_properties(self) :
        #print 'restore_line_properties'
        #print 'Save Color, Width, Style =', self.myColor, self.myWidth, self.myStyle 
        self.set_color    (self.myColor)
        self.set_linewidth(self.myWidth)
        self.set_linestyle(self.myStyle)
        if self.isRemoved : self.set_linewidth(0)


    def set_dragged_obj_properties(self) :
        self.set_color    ('k')
        self.set_linewidth(1)
        self.set_linestyle('dashed')
        if self.isRemoved : self.set_linewidth(0)


    def redraw_artist(self) :
        self.axes.draw_artist(self)


    def redraw_blit(self):
        self.figure.canvas.blit(self.axes.bbox) # blit canvas before any motion

        
    def redraw_obj(self) :
        self.axes.draw_artist(self)        
        self.figure.canvas.blit(self.axes.bbox) # blit canvas before any motion


    def save_current_background(self) :
        """Saves the background in a single place for this figure"""
        self.figure.my_background = self.figure.canvas.copy_from_bbox(self.axes.bbox) # Added for blit


    def restore_background(self) :
        self.figure.canvas.restore_region(self.figure.my_background) # Added for blit


    def get_mode(self) :
        return self.figure.my_mode


    def select_deselect_obj(self) :

        #print 'select_deselect_obj() : self.isRemoved = ', self.isRemoved

        if self.press == None     : return
        if not self.isInitialized : return

        self.set_needs_in_redraw(True)  

        if self.get_mode() != self.modeSelect   : return

        if  self.isSelected == False :
            self.isSelected =  True
            print 'object IS SELECTED'
        else:
            self.isSelected =  False
            print 'object IS DESELECTED'


    def needs_in_redraw(self) :
        if self.needsInRedraw : return True
        else                  : return False


    def set_needs_in_redraw(self, status=True) :
        self.needsInRedraw = status


    def set_select_deselect_color(self) :
        #if self.get_mode() != self.modeSelect : return
        if self.isSelected :
            self.set_color('w')
        else :
            self.set_color(self.myColor)


    def on_press_graphic_manipulations(self) :
        self.save_obj_properties()              # Save original line properties
        self.set_linewidth(0)                   # Makes the line invisible
        #self.set_linestyle('')                 # Draw nothing - makes the line invisible
        self.figure.canvas.draw()               # Draw canvas and save the background 
        self.save_current_background()          # Added for blit
        self.on_press_remove()                  # In case of remove mode
        self.set_dragged_obj_properties()       # Set the line properties during dragging 
        self.redraw_obj()
        self.select_deselect_obj()


    def on_motion_graphic_manipulations(self) :
        #self.figure.canvas.draw()                         # Regular case
        if self.isRemoved : return
        self.restore_background()
        self.redraw_obj()


    def on_release_graphic_manipulations(self) :
        """on release we reset the press data"""
        self.restore_obj_properties()
        self.set_select_deselect_color()
        self.redraw_obj()
        self.isInitialized = True
        self.on_release_remove()
        #self.figure.canvas.draw()

#-----------------------------
# is called by ImgFigureManager -> ImgDrawOnTop.py -> in order to remove object from main image

    def remove_object_from_img(self) :
        #print 'Drag : remove_object_from_img() : ', self.print_pars()
        self.set_linewidth(0)                   # Makes the line invisible
        #self.set_linestyle('')                 # Draw nothing - makes the line invisible
        self.figure.canvas.draw()               # Draw canvas with all current objects on it
        self.disconnect()
        self.isRemoved = True

#-----------------------------
#-----------------------------
# Global methods
#-----------------------------
#-----------------------------

def add_obj_to_axes(obj, axes, list_of_objs) :
    """Add object to axes and append the list of objects
    """
    obj.add_to_axes(axes)                        # 1. Add it to figure axes
    obj.connect()                                # 2. Connect object with signals
    list_of_objs.append(obj)                     # 3. Append the list of objects 

#-----------------------------
# Global methods for test
#-----------------------------

import numpy as np

def get_array2d_for_test() :
    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=13500)
    #arr = np.arange(2400)
    arr.shape = (90,150)
    return arr


import matplotlib.pyplot as plt

def generate_test_image() :
    """Produce the figure with 2-d image for the test purpose.
       returns the necessary for test parameters: 
    """
    fig = plt.figure()
    fig.my_mode = None # This is used to transmit signals
    axes = fig.add_subplot(111)

    axesImage = axes.imshow(get_array2d_for_test(), origin='upper', interpolation='nearest', aspect='auto')#, extent=self.range
    #axesImage.set_clim(zmin,zmax)
    mycolbar = fig.colorbar(axesImage, pad=0.1, fraction=0.15, shrink=1.0, aspect=15, orientation=1)#, ticks=coltickslocs) #orientation=1,

    return fig, axes, axesImage

#-----------------------------

def main_test_global():
    """Test of the global methods only
    """
    fig, axes, axesImage = generate_test_image()

    plt.get_current_fig_manager().window.move(50, 10)
    plt.show()

#-----------------------------
import sys

if __name__ == "__main__" :

    main_test_global()
    sys.exit ('End of test')

#-----------------------------
