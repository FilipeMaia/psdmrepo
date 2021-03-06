#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgDrawSpectrum...
#
#------------------------------------------------------------------------

"""Additional graphics outside of 2-d image

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
#import os

import matplotlib.pyplot as plt
import ImgFigureManager  as imgfm

#---------------------
#  Class definition --
#---------------------

class ImgDrawSpectrum :
    """Additional graphics outside of the 2-d image for spectrum"""

    def __init__(self, icp=None):
        self.icp           = icp
        self.icp.idrawspec = self
        self.arr           = None

        #self.icp.list_of_rects = [] # moved to icp

    def get_control(self) :
        return self.icp.control


    def draw_spectrum_for_rect(self,obj) :
        x,y,w,h,lw,col,s,t,r = obj.get_list_of_rect_pars() 
        print 'draw_spectrum_for_rect : x,y,w,h,lw,col,s,t,r =', x,y,w,h,lw,col,s,t,r

        plt.ion()

        fig_outside = obj.get_fig_outside()                    # Get figure from the object

        if fig_outside == None : # if figure is not open yet
        
            fig_outside = imgfm.ifm.get_figure(figsize=(5,4), type='type1', icp=self.icp) # type='maxspace'
            obj.set_fig_outside(fig_outside)                   # Preserve figure in the object
            fig_outside.obj_index = obj.myIndex 
            fig_outside.my_object = obj
            fig_outside.canvas.set_window_title('Spectrum for rect %d' % obj.myIndex  )
            print 'fig_outside.number=',fig_outside.number

        else :

            if not obj.isChanged : return
            fig_outside.clear()
            fig_outside.canvas.set_window_title('Spectrum for rect %d' % fig_outside.obj_index )


        if s :   # Re-draw the figure window on top

            fig_outside.canvas.manager.window.activateWindow() # Makes window active
            fig_outside.canvas.manager.window.raise_()         # Move window on top

            #number = rect.get_fig_outside().number               # Preserve the figure number
            #pos = imgfm.ifm.get_fig_window_position(fig_outside) # Preserve the figure position
            #self.get_control().signal_and_close_fig(number)      # Close figure
            #fig_outside = imgfm.ifm.get_figure(num=number, figsize=(5,4), type='type1', icp=self.icp)
            #imgfm.ifm.set_fig_window_position(fig_outside, pos)  # Restore the figure position
            #rect.set_fig_outside(fig_outside)
            #fig_outside.rect_index = rect.myIndex 
            #fig_outside.my_object  = rect
            #fig_outside.canvas.set_window_title('Spectrum for rect %d' % fig_outside.rect_index )

        arrwin =  self.arr[y:y+h,x:x+w]

        axsb = fig_outside.add_subplot(111)
        axsb.hist(arrwin.flatten(), bins=100)#, range=range)

        fig_outside.canvas.draw()

        obj.isChanged = False
        #plt.ioff()

#-----------------------------

    def draw_outside_plots_for_list_of_objs(self, arr) :
        print 'ImgDrawSpectrum : draw_outside_plots_for_list_of_objs(...)'

        self.arr = arr

        for obj in self.icp.list_of_rects :                      # <====== DEPENDS ON FORM
            if obj.myType == self.icp.typeSpectrum :             # <====== DEPENDS ON FORM
                self.draw_spectrum_for_rect(obj)                 # <====== DEPENDS ON FORM

#-----------------------------
# Test
#-----------------------------

def main():
    w = ImgDrawSpectrum()

#-----------------------------

if __name__ == "__main__" :
    #main()
    sys.exit ('Module is not supposed to be run for test...')

#-----------------------------
