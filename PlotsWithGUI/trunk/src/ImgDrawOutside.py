#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgDrawOutside...
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
import os

import matplotlib.pyplot as plt

import ImgConfigParameters as icp
import ImgFigureManager    as imgfm
#---------------------
#  Class definition --
#---------------------

class ImgDrawOutside :
    """Additional graphics outside of the 2-d image"""

    def __init__(self, icp=None):
        self.icp          = icp
        self.icp.idrawout = self

        #self.icp.list_of_rects = [] # moved to icp
        print 'ImgDrawOutside : init' 


    def get_control(self) :
        return self.icp.control


    def get_config_pars(self) :
        return self.icp


    def get_wimg(self) :
        return self.icp.wimg


    def draw_spectrum_for_rect(self,rect) :
        x,y,w,h,s,t = rect.get_list_of_rect_pars() 
        print 'draw_spectrum_for_rect : x,y,w,h,s,t =', x,y,w,h,s,t

        plt.ion()

        if rect.get_fig_outside() == None : # if figure is not open yet
        
            fig_outside = imgfm.ifm.get_figure(figsize=(5,4), type='type1', icp=self.icp) # type='maxspace'
            rect.set_fig_outside(fig_outside)
            fig_outside.rect_index = rect.myIndex 
            fig_outside.my_object  = rect
            fig_outside.canvas.set_window_title('Spectrum for rect %d' % fig_outside.rect_index )
            print 'fig_outside.number=',fig_outside.number

        else :

            if not rect.needsInRedraw : return

            fig_outside = rect.get_fig_outside()
            #fig_outside = plt.figure(num=fig_outside.number,figsize=(4,3))
            fig_outside.clear()
            fig_outside.canvas.set_window_title('Spectrum for rect %d' % fig_outside.rect_index )


        if s : # rect is selected -> redraw its outside figure
            number = rect.get_fig_outside().number
            self.get_control().signal_and_close_fig(number)
            fig_outside = imgfm.ifm.get_figure(num=number, figsize=(5,4), type='type1', icp=self.icp)
            rect.set_fig_outside(fig_outside)
            fig_outside.rect_index = rect.myIndex 
            fig_outside.my_object  = rect
            fig_outside.canvas.set_window_title('Spectrum for rect %d' % fig_outside.rect_index )

        arrwin =  self.arr[y:y+h,x:x+w]

        axsb = fig_outside.add_subplot(111)
        axsb.hist(arrwin.flatten(), bins=100)#, range=range)

        fig_outside.canvas.draw()

        rect.needsInRedraw = False
        #plt.ioff()


    def draw_spectra(self) :
        print 'draw_spectra(...)'
        for rect in self.icp.list_of_rects :
            if rect.myType == self.icp.typeCurrent : self.draw_spectrum_for_rect(rect)


    def draw_outside(self) :
        self.draw_spectra()

#-----------------------------
# New

    def remove_spectra(self, obj) :
        number = obj.get_fig_outside().number
        self.get_control().signal_and_close_fig(number)

#-----------------------------
# Test
#

def main():
    w = ImgDrawOutside()

#-----------------------------

if __name__ == "__main__" :
    main()

#-----------------------------
