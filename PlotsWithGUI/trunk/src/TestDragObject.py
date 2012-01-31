#!/usr/bin/env python
#-----------------------------

class TestDragObject : 

    def __init__(self, fig, axes) :

        self.fig          = fig
        self.axes         = axes
        self.fig.my_mode  = None  # Mode for interaction between fig and obj
        self.list_of_objs = []

        self.fig.canvas.mpl_connect('key_press_event',      self.on_key_press)      # for test only
        self.fig.canvas.mpl_connect('button_press_event',   self.on_mouse_press)    # for Add mode
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)  # for Remove
        #self.fig.canvas.mpl_connect('motion_notify_event',  self.on_mouse_motion)   # For Remove (at the end)

        self.print_mode_keys()


    def set_list_of_objs(self, list_of_objs) :
        self.list_of_objs = list_of_objs

        
    def print_mode_keys(self) :
        """Prints the hint for mode selection using keyboard
        """
        print '\n\nUse keyboard to select the mode: \nA=ADD, \nM=MARK, \nD=REMOVE SELECTED, \nR=REMOVE, \nN=NONE, \nW=PRINT list'


    def on_key_press(self, event) :
        """Responds on keyboard signals and switch the fig.my_mode
           It also prints the list of objects for key W.
        """
        if   event.key == 'a': self.fig.my_mode  = 'Add'
        elif event.key == 'n': self.fig.my_mode  = 'None'
        elif event.key == 'r': self.fig.my_mode  = 'Remove'
        elif event.key == 'm': self.fig.my_mode  = 'Select'
        elif event.key == 'w': self.print_list_of_objs()
        elif event.key == 'd': self.test_remove_selected_objs_from_img_by_call()
        else                 : self.print_mode_keys()
        print '\nCurrent mode:', self.fig.my_mode


    def on_mouse_release(self, event) :
        #print 'on_mouse_release - test of order'
        self.update_list_of_objs()


#    def on_mouse_motion(self, event) :
#        """HAVE TO USE IT, because I need to update the list after the button release loop over all objects...
#        """
#        #print 'mouse is moved.. do something...', event.xdata, event.ydata
#        if self.needInUpdate :
#            self.update_list_of_objs()
#            self.needInUpdate = False


    def send_signal_and_remove_object_from_list(self, obj, remove_type) :
        print 'Object is removing from the list. ACT HERE !!!'
        obj.print_pars()
        #====================== REMOVE OBJECT =================================================
        # THIS IS A PLACE TO REMOVE EVERYTHING ASSOCIATED WITH OBJECT AFTER CLICK OR PROGRAM CALL
        # SIGNAL ABOUT REMOVAL SHOULD BE SENT FROM HERE
        if remove_type == 'Call' : obj.remove_object_from_img()  # <<<========= PROGRAM CALL TO REMOVE THE OBJECT FROM IMG
        self.list_of_objs.remove(obj)                            # <<<========= REMOVE OBJECT FROM THE LIST
        #======================================================================================

    
    def update_list_of_objs(self) :
        print 'update_list_of_objs()'
        initial_list_of_objs = list(self.list_of_objs) # the list() is used in order to COPY object in new list
        for obj in initial_list_of_objs :
            if obj.isRemoved :
                #print 'Object ', initial_list_of_objs.index(obj), 'is removing from the list. ACT HERE !!!'
                #====================== REMOVE OBJECT BY CLICK ON MOUSE ===============================
                self.send_signal_and_remove_object_from_list(obj, 'Click')
                #======================================================================================


    def test_remove_selected_objs_from_img_by_call(self) :
        """Loop over list of objects and remove selected from the image USING CALL from code.
        """
        print 'test_remove_selected_objs_from_img_by_call()'

        initial_list_of_objs = list(self.list_of_objs)
        for obj in initial_list_of_objs :
            if obj.isSelected :
                #print 'Object', initial_list_of_objs.index(obj), 'is selected and will be removed in this test'
                #====================== REMOVE OBJECT BY PROGRAM CALL =================================
                self.send_signal_and_remove_object_from_list(obj, 'Call')
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
