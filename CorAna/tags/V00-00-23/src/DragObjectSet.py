#!/usr/bin/env python
#-----------------------------
from Drag import * # for access to global methods like add_obj_to_axes(...)

from PyQt4 import QtGui, QtCore # For ability to override cursor...

class DragObjectSet : 

    def __init__(self, fig, axes=None, ObjectType=None, useKeyboard=False, is_single_obj=False, use_xyc=False, lw=1, col='b', picker=8) :

        self.ObjectType    = ObjectType
        self.axes          = axes
        self.fig           = fig
        self.is_single_obj = is_single_obj
        self.fig.my_mode   = None  # Mode for interaction between fig and obj
        self.use_xyc       = use_xyc
        self.lw            = lw
        self.col           = col
        self.picker        = picker
        self.new_obj       = None

        #print 'lw,col,picker=', lw, col, picker

        self.list_of_objs  = []

        self.connect_objs()

        if useKeyboard :
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)      # for test only
            self.print_mode_keys()


    def set_xy_center(self, xyc) :
        self.fig.my_xyc = xyc


    def get_list_of_objs(self) :
        return self.list_of_objs 


    def connect_objs(self) :
        self.cid_press  =self.fig.canvas.mpl_connect('button_press_event',   self.on_mouse_press)    # for Add mode
        self.cid_release=self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)  # for Remove
        self.cid_motion =self.fig.canvas.mpl_connect('motion_notify_event',  self.on_mouse_motion)   # For Remove (at the end)
        for obj in self.list_of_objs :
            obj.connect()


    def disconnect_objs(self):
        self.fig.canvas.mpl_disconnect(self.cid_press)
        self.fig.canvas.mpl_disconnect(self.cid_release)
        self.fig.canvas.mpl_disconnect(self.cid_motion)

        for obj in self.list_of_objs :
            obj.disconnect()



    def add_obj_for_str_of_pars(self, str_of_pars) :
        """Add object when load the forms from file"""
        obj_type = str_of_pars.split(' ',1)[0]
        #print 'Add object with pars:', str_of_pars
        if self.is_single_obj and len(self.list_of_objs) > 0 :
            print 'WARNING ! This is a singleton. One object is already in the list. Request to add more object(s) is ignored.'
            return
        # Creates the draggable object with 1st vertex in xy
        # else Creates the draggable object with 1st vertex in xy
        if self.use_xyc : obj = self.ObjectType(str_of_pars=str_of_pars, linewidth=self.lw, color=self.col, picker=self.picker, xy=self.fig.my_xyc)
        else            : obj = self.ObjectType(str_of_pars=str_of_pars, linewidth=self.lw, color=self.col, picker=self.picker)
        add_obj_to_axes(obj, self.axes, self.list_of_objs)        



    def set_list_of_objs(self, list) :
        for obj in list :
            if self.is_single_obj and len(self.list_of_objs) > 0 :
                print 'WARNING ! This is a singleton. One object is already in the list. Request to add more object(s) is ignored.'
                return
                add_obj_to_axes(obj, self.axes, self.list_of_objs)



    def print_mode_keys(self) :
        """Prints the hint for mode selection using keyboard
        """
        print '\n\nUse keyboard to select the mode: \nA=ADD, \nM=MARK, \nD=REMOVE SELECTED, \nR=REMOVE, \nN=NONE, \nW=PRINT list'



    def override_cursor(self, contains):
        if contains :
            QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        else :
            QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        #QtCore.Qt.PointingHandCursor,    
        #QtCore.Qt.SizeAllCursor



    def on_key_press(self, event) :
        """Responds on keyboard signals and switch the fig.my_mode
           It also prints the list of objects for key W.
        """
        if   event.key == 'a': self.fig.my_mode  = 'Add'
        elif event.key == 'n': self.fig.my_mode  = 'None'
        elif event.key == 'r': self.fig.my_mode  = 'Remove'
        elif event.key == 'm': self.fig.my_mode  = 'Select'
        elif event.key == 'w': self.print_list_of_objs()
        elif event.key == 'd': self.remove_selected_objs_from_img_by_call()
        else                 : self.print_mode_keys()
        print '\nCurrent mode:', self.fig.my_mode


    def on_mouse_motion(self, event) :
        #===========================
        #return # In order to get rid of dependence on PyQt for self test
        #===========================

        # Change cursor for fun...
        for obj in self.list_of_objs :
            if obj.obj_contains_cursor(event) :
                self.override_cursor(True)
                return
        self.override_cursor(False)


    def on_mouse_release(self, event) :
        #print 'on_mouse_release - test of order'
        #=========================
        self.update_list_of_objs()
        #=========================


    # THE ONLY OBJECT-SPECIFIC METHOD: Depends on  self.ObjectType and self.axes
    def on_mouse_press(self, event) :
        """Responds on mouse signals and do the object initialization for the mode Add
        """
        #x,y = (event.xdata, event.xdata+1)

        if self.fig.my_mode == 'Add' :

            # This check is required for multi-click objects like polygon
            # It should not create new object until previous is completely initializad
            if self.new_obj is not None : 
                if not self.new_obj.isInitialized : return

            if event.button != 1 : return # if other than Left mouse button
            #print 'mode=', self.fig.my_mode
            if self.is_single_obj and len(self.list_of_objs) > 0 :
                print 'WARNING ! This is a singleton. One object is already in the list. Request to add more object(s) is ignored.'
                return
            if self.use_xyc : self.new_obj = self.ObjectType(linewidth=self.lw, color=self.col, picker=self.picker, xy=self.fig.my_xyc) # Creates the draggable object with 1st vertex in xy
            else            : self.new_obj = self.ObjectType(linewidth=self.lw, color=self.col, picker=self.picker) # Creates the draggable object with default pars, which will be set dynamically

            add_obj_to_axes(self.new_obj, self.axes, self.list_of_objs)
            self.new_obj.on_press(event)                        # <<<==========================


    def send_signal_and_remove_object_from_list(self, obj, list_of_objs, remove_type) :
        #print 'Object is removing from the list. ACT HERE !!!, remove_type=', remove_type
        #obj.print_pars()
        #====================== REMOVE OBJECT =================================================
        # THIS IS A PLACE TO REMOVE EVERYTHING ASSOCIATED WITH OBJECT AFTER CLICK OR PROGRAM CALL
        # SIGNAL ABOUT REMOVAL SHOULD BE SENT FROM HERE
        if remove_type == 'Call' : obj.remove_object_from_img() # <<<========= PROGRAM CALL TO REMOVE THE OBJECT FROM IMG
        list_of_objs.remove(obj)                                # <<<========= REMOVE OBJECT FROM THE LIST
        #======================================================================================

    
    def update_list_of_objs(self) :
        #print 'update_list_of_objs()'
        initial_list_of_objs = list(self.list_of_objs) # the list() is used in order to COPY object in new list
        for obj in initial_list_of_objs :
            if obj.isRemoved :
                #print 'Object ', initial_list_of_objs.index(obj), 'is removing from the list. ACT HERE !!!'
                #====================== REMOVE OBJECT BY CLICK ON MOUSE ===============================
                self.send_signal_and_remove_object_from_list(obj, self.list_of_objs, 'Click')
                #======================================================================================


    def remove_selected_objs_from_img_by_call(self) :
        """Loop over list of objects and remove selected from the image USING CALL from code.
        """
        #print 'remove_selected_objs_from_img_by_call()'

        initial_list_of_objs = list(self.list_of_objs)
        for obj in initial_list_of_objs :
            if obj.isSelected :
                #print 'Object', initial_list_of_objs.index(obj), 'is selected and will be removed in this test'
                #====================== REMOVE OBJECT BY PROGRAM CALL =================================
                self.send_signal_and_remove_object_from_list(obj, self.list_of_objs, 'Call')
                #======================================================================================

    def remove_all_objs_from_img_by_call(self) :
        """Loop over list of objects and remove them from the image USING CALL from code.
        """
        initial_list_of_objs = list(self.list_of_objs)
        for obj in initial_list_of_objs :
            #====================== REMOVE OBJECT BY PROGRAM CALL =================================
            self.send_signal_and_remove_object_from_list(obj, self.list_of_objs, 'Call')
            #======================================================================================


    def print_list_of_objs(self) :
        """Prints the list of objects with its parameters.
        """
        print 'Print list of', len(self.list_of_objs), 'objects'
        for obj in self.list_of_objs :
            print 'ind=', self.list_of_objs.index(obj),':',
            obj.print_pars()

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
