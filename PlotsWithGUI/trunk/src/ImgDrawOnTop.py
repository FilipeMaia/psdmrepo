#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgDrawOnTop...
#
#------------------------------------------------------------------------

"""Additional graphics on top of 2-d image

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule
@version $Id: 
@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

import DragCircle          as dragc
import DragRectangle       as dragr
#import ImgConfigParameters as icp

#---------------------
#  Class definition --
#---------------------

class ImgDrawOnTop :
    """Additional graphics on top of 2-d image"""


    def __init__(self, icp=None):
        self.icp               = icp
        self.icp.idrawontop    = self

        self.rectsFromInputAreCreated = False

        #self.icp.list_of_rects = [] # moved to icp
        print 'ImgDrawOnTop : init' 


    def get_control(self) :
        return self.icp.control


    def get_config_pars(self) :
        return self.icp


    def get_wimg(self) :
        return self.icp.wimg


    def get_idrawout( self ) :
        return self.icp.idrawout


    def add_rect(self, type=None, selected=False, xy=None, width=1, height=1):
        self.update_list_of_rects()
        obj = dragr.DragRectangle(xy=xy, width=width, height=height, color='r') 
        obj.add_to_axes(self.get_wimg().fig.myaxesSIm)
        obj.connect()
        obj.myType     = type
        obj.isSelected = selected
        obj.set_select_deselect_color()
        self.icp.list_of_rects.append(obj)
        obj.myIndex=self.icp.list_of_rects.index(obj) # index in the list
        print 'obj.myIndex=', obj.myIndex
        #self.get_idrawout().draw_spectrum_for_rect(obj)


    def redraw_rect(self, obj):
        obj.disconnect() # presumably from old canvas
        obj.add_to_axes(self.get_wimg().fig.myaxesSIm)
        obj.connect()
        obj.set_select_deselect_color()
        obj.myIndex=self.icp.list_of_rects.index(obj) # index in the list
        print 'obj.myIndex=', obj.myIndex


    def none_rect(self):
        pass # self.update_list_of_rects()


    def overlay_rect(self):
        pass


    def select_rect(self):
        pass


    def remove_rect(self):
        #rect_for_remove = self.find_rect_for_remove()
        #if rect_for_remove == None : return
        #self.get_idrawout().close_figure_for_rect(rect_for_remove)        
        self.update_list_of_rects()
        # Rect for removal IS NOT SELECTED YET...


    def find_rect_for_remove(self):
        for self.obj in self.icp.list_of_rects :
            if self.obj.isRemoved :
                return self.obj
        return None


    def update_list_of_rects(self):
        for obj in self.icp.list_of_rects :
            #print 'update_list_of_rects: obj.isRemoved=', obj.isRemoved
            if obj.isRemoved :
                self.icp.list_of_rects.remove(obj)
        #print 'len(self.icp.list_of_rects)=', len(self.icp.list_of_rects)


    def draw_rects(self):

        if self.rectsFromInputAreCreated :
            for obj in self.icp.list_of_rects :
                self.redraw_rect(obj)
        else:
            for rectPars in self.icp.listOfRectInputParameters :
                #print rectPars
                t,s,x,y,w,h = rectPars
                if t == self.icp.typeCurrent :
                    self.add_rect(type=t, selected=s, xy=(x,y), width=w, height=h)

        self.rectsFromInputAreCreated = True


    def draw_on_top(self):
        print 'draw_on_top()'
        self.draw_rects()

#-----------------------------
# is called from ImgFigureManager.

    def remove_object(self, obj):
        print 'ImgDrawOnTop : remove_object'
        if obj in self.icp.list_of_rects :
            obj.remove_object_from_img()
            self.icp.list_of_rects.remove(obj) 
            self.icp.control.set_signal_info(mode=self.icp.modeNone)

#-----------------------------
# Test
#

def main():
    w = ImgDrawOnTop(None)

#-----------------------------

if __name__ == "__main__" :
    main()

#-----------------------------
