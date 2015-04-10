#!/usr/bin/env python
#----------------------------------
import numpy as np
#import numpy.ma as ma
#import copy
import matplotlib.patches as patches
# from   matplotlib.nxutils import points_inside_poly # <--- depricated in 1.3.x 
from matplotlib.path import Path
import math # cos(x), sin(x), radians(x), degrees()


#from PyQt4 import QtGui, QtCore # need it in order to use QtCore.QObject for signal exchange...

class Drag () : # ( QtCore.QObject )

    dicBool = {'false':False, 'true':True}

    def __init__(self, linewidth=2, color='k', linestyle='solid', my_type=None) :
        """
        Actual initialization is done by the method add_to_axes(...)
        """        
        #QtCore.QObject.__init__(self, None) # need it for signal exchange...

        self.isInitialized = False

        self.myType          = str(my_type) # to print/save/read the string of parameters
        self.myWidth         = linewidth
        self.myStyle         = linestyle
        self.isSelected      = False
        self.isRemoved       = False
        self.fig_outside     = None
        self.isChanged       = False
        self.modeRemove      = 'Remove'  # Should be the same as icp.modeRemove
        self.modeSelect      = 'Select'
        self.maskIsAvailable = False
        self.myColor         = color
        self.myCurrentColor  = color

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


    def obj_contains_cursor(self, event):
        if event.inaxes != self.axes: return False

        if self.isInitialized : 
            contains, attrd = self.contains(event)
            if contains : return True
        return False


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
        self.set_linestyle('solid') #self.myStyle)


    def set_dragged_obj_properties(self) :
        self.set_color    ('k')
        self.set_linewidth(1)
        self.set_linestyle('dashed')


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

        if not self.isInitialized             : return
        if self.press is None                 : return
        self.isChanged = True

        if self.get_mode() != self.modeSelect : return
        self.swap_select_deselect_status()

        
    def swap_select_deselect_status(self) :

        if  self.isSelected == False :
            self.isSelected =  True
            #print 'object IS SELECTED'
        else:
            self.isSelected =  False
            #print 'object IS DESELECTED'


    def set_select_deselect_color(self, color='w') :

        if self.isSelected : self.myCurrentColor = color
        else               : self.myCurrentColor = self.myColor

        self.set_color(self.myCurrentColor)


    def on_press_graphic_manipulations(self) :
        """on press we do:
        1. check if object needs to be removed and remove it if the mode is Remove
        2. make the object invisible and re-draw canwas (w/o this object)
        3. save this as a canva background
        4. set the attributes for draged object
        5. redraw object with dragged attributes
        6. check if the object needs to be selected
        """

        if self.is_on_press_remove() : return   # In case of remove mode
        self.save_obj_properties()              # Save original line properties
        self.set_linewidth(0)                   # Makes the line invisible
        #self.set_linestyle('')                 # Draw nothing - makes the line invisible (Does not work for Circlr)
        self.figure.canvas.draw()               # Draw canvas and save the background 
        self.save_current_background()          # Added for blit
        self.set_dragged_obj_properties()       # Set the line properties during dragging 
        self.redraw_obj()
        self.select_deselect_obj()              # Set/reset the isSelected flag 


    def on_motion_graphic_manipulations(self) :
        """on motion we redraw the background and object in its current position
        """
        self.restore_background()
        self.redraw_obj()


    def on_release_graphic_manipulations(self) :
        """on release we reset the press data
        """
        self.restore_obj_properties()
        self.set_select_deselect_color()
        self.redraw_obj()
        self.isInitialized = True

#-----------------------------

    def is_on_press_remove(self) :
        """Remove the object for remove mode
        """
        if self.get_mode() == self.modeRemove :
            self.remove_object_from_img()
            return True
        else :
            return False

#-----------------------------
# Is called by
# 1. on_press_graphic_manipulations()
# 2. ImgFigureManager -> ImgDrawOnTop.py -> in order to remove object from main image

    def remove_object_from_img(self) :
        """Remove object from figure canvas, axes, disconnect from signals, mark for removal 
        """
        #print 'Drag : remove_object_from_img() : ', self.print_pars()
        #self.set_linestyle('')                 # Draw nothing - makes the line invisible
        self.remove()                           # Removes artist from axes
        self.figure.canvas.draw()               # Draw canvas with all current objects on it
        self.disconnect()                       # Disconnect object from canvas signals
        self.isRemoved = True                   # Set flag that the object is removed

#-----------------------------

    def select_deselect_object_by_call(self, color='w') :
        """Select/deselect object and change its color on canvas
        """
        self.swap_select_deselect_status()      # Swap the isSelected between True and False
        self.set_select_deselect_color(color)   # Set the object color depending on isSelected
        self.figure.canvas.draw()               # Draw canvas with all current objects on it

#-----------------------------

    def get_obj_mask(self, shape):
        if not self.maskIsAvailable :
            self.mask = get_mask(shape, self.get_poly_verts())
            self.maskIsAvailable = True
        if self.isSelected : return ~self.mask # inversed mask
        else               : return  self.mask # mask

#-----------------------------
#-----------------------------
# Global methods
#-----------------------------
#-----------------------------

def get_mask(shape, poly_verts) :
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    ij   = zip(x.flatten(), y.flatten()) # or np.vstack((x,y)).T
    #mask = np.array(points_inside_poly(ij, poly_verts))
    mask = np.array(Path(poly_verts).contains_points(ij))
    mask.shape = shape
    return mask


def add_obj_to_axes(obj, axes, list_of_objs) :
    """Add object to axes and append the list of objects
    """
    obj.add_to_axes(axes)                        # Add it to figure axes
    obj.connect()                                # Connect object with signals
    list_of_objs.append(obj)                     # Append the list of objects 


def remove_object_from_img_and_list(obj, list):
    #print 'Draw : remove_object'
    if obj in list :
        obj.remove_object_from_img()
        list.remove(obj) 


def set_list_need_in_redraw(list_of_obj):
    for obj in list_of_obj :
        obj.isChanged = True


def redraw_obj_update_list(obj, axes, list_of_objs):
    obj.disconnect() # disconnect object from canvas signals
    #obj.remove()     # remove object from axes
    if obj in list_of_objs : list_of_objs.remove(obj)
    add_obj_to_axes(obj, axes, list_of_objs)
    obj.myIndex=list_of_objs.index(obj) # index in the list
    #print 'Drag : redraw_obj_update_list(...),  obj.myIndex=', obj.myIndex, '(last added to the list)'


def redraw_objs_from_list(axes, list_of_objs) :   
    initial_list_of_objs = list(list_of_objs)
    for obj in initial_list_of_objs :
        redraw_obj_update_list(obj, axes, list_of_objs)
        #drag.redraw_obj_update_list(obj, axes, list_of_objs)


#-----------------------------
# Global methods for test
#-----------------------------

import numpy as np

def get_array2d_for_test() :
    mu, sigma = 200, 25
    rows, cols = 1300, 1340
    arr = mu + sigma*np.random.standard_normal(size=rows*cols)
    #arr = np.arange(2400)
    arr.shape = (rows,cols)
    return arr


import matplotlib.pyplot as plt

def generate_test_image() :
    """Produce the figure with 2-d image for the test purpose.
       returns the necessary for test parameters: 
    """
    fig = plt.figure(figsize=(7,6), dpi=100, facecolor='w', edgecolor='w', frameon=True)
    fig.my_mode = None # This is used by the Drag* objects
    #axes = fig.add_subplot(111)
    axes = fig.add_axes([0.08,  0.05, 0.89, 0.92])
 
    imsh = axes.imshow(get_array2d_for_test(), origin='upper', interpolation='nearest', aspect='auto')#, extent=self.range
    mycolbar = fig.colorbar(imsh, pad=0.01, fraction=0.08, shrink=1.0, aspect=30, orientation='vertical')#, ticks=coltickslocs)

    return fig, axes, imsh

#-----------------------------

def main_test_global():
    """Test of global methods only
    """
    fig, axes, imsh = generate_test_image()

    plt.get_current_fig_manager().window.geometry('+50+10') #move(50, 10)
    plt.show()

#-----------------------------

def main_test_global_mask():
    """Test mask
    """
    shape = (10,10)
    poly_verts = [(1,1), (5,1), (5,9), (3,2), (1,1)] 
    mask = get_mask(shape, poly_verts)

    print 'mask:\n', mask

#-----------------------------
import sys

if __name__ == "__main__" :

    #main_test_global()
    main_test_global_mask()
    sys.exit ('End of test')

#-----------------------------
