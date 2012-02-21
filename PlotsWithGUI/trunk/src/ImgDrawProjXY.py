#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgDrawProjXY...
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
import numpy as np

import matplotlib.gridspec     as gridspec
import matplotlib.pyplot       as plt
import matplotlib.ticker       as mtick
import ImgFigureManager        as imgfm
import FastArrayTransformation as fat

#---------------------
#  Class definition --
#---------------------

class ImgDrawProjXY :
    """Additional graphics outside of the 2-d image for zoomed region with projections"""

    def __init__(self, icp=None):
        self.icp           = icp
        self.icp.idrawprxy = self
        self.arr           = None

        #self.icp.list_of_rects = [] # moved to icp

    def get_control(self) :
        return self.icp.control


    def draw_prxy_for_rect(self,obj) :
        x,y,w,h,lw,col,s,t,r = obj.get_list_of_rect_pars() 
        print 'draw_zoom_for_rect : x,y,w,h,lw,col,s,t,r =', x,y,w,h,lw,col,s,t,r

        plt.ion()

        fig_outside = obj.get_fig_outside()                    # Get figure from the object

        if fig_outside == None : # if figure is not open yet
        
            fig_outside = imgfm.ifm.get_figure(figsize=(7,7), type='type1', icp=self.icp) # type='maxspace','type1'
            obj.set_fig_outside(fig_outside)                   # Preserve figure in the object
            fig_outside.obj_index = obj.myIndex 
            fig_outside.my_object = obj
            fig_outside.canvas.set_window_title('Projections X and Y for rect %d' % obj.myIndex  )
            print 'fig_outside.number=',fig_outside.number

        else :

            if not obj.isChanged : return
            fig_outside.clear()
            fig_outside.canvas.set_window_title('Projections X and Y for rect %d' % fig_outside.obj_index )

        if s :   # Re-draw the figure window on top

            fig_outside.canvas.manager.window.activateWindow() # Makes window active
            fig_outside.canvas.manager.window.raise_()         # Move window on top

        self.drawImgAndProjsInRect(fig_outside, self.arr, obj)

        fig_outside.canvas.draw()

        obj.isChanged = False
        #plt.ioff()


#-----------------------------


    def drawImgAndProjsInRect(self, fig, arr, obj) :

        x,y,w,h,lw,col,s,t,r = obj.get_list_of_rect_pars() 

        nx_slices, ny_slices = obj.get_number_of_slices_for_rect()

        arrwin  =  arr[y:y+h,x:x+w]
        xyrange = [x, x+w, y+h, y]
      
        #axsb = fig.add_subplot(111)
        gs   = gridspec.GridSpec(20, 20)

    #                            [  Y   ,   X ]
        axsa = fig.add_subplot(gs[ 1:14,  0:12])
        axsb = fig.add_subplot(gs[ 1:14, 12:19])
        axsc = fig.add_subplot(gs[14:  ,  0:12])
        axsd = fig.add_subplot(gs[15:  , 13:19])
        axCB = fig.add_subplot(gs[    0,  0:12])

        #------------------
        # Panel A
        axim = axsa.imshow(arrwin, interpolation='nearest', aspect='auto', extent=xyrange) #, origin='bottom'
        axsa.xaxis.set_major_formatter(mtick.NullFormatter()) 

        #------------------
        # Panel CB
        colb = fig.colorbar(axim, cax=axCB, orientation='horizontal') # pad=0.004, fraction=0.1, aspect=40) 
        axCB.xaxis.set_ticks_position('top')

        #------------------
        # Panel B
        xrange  = (x,x+w,ny_slices)
        yrange  = (y,y+h,h)
        arr2dy  = fat.rebinArray(arr, xrange, yrange) 
        #axprojy = axsb.imshow(arr2dy, interpolation='nearest', origin='bottom', aspect='auto', extent=xyrange)

        for slice in range(ny_slices) :
            arr1slice = arr2dy[...,slice]
            yarr = np.linspace(y, y+h, num=h, endpoint=True)
            if arr1slice.sum() == 0 :
                print 'Empty histogram for slice =', slice,'is ignored'
                continue

            axsb.hist(yarr, bins=h, weights=arr1slice, histtype='step', orientation='horizontal')

        axsb.set_ylim(y+h, y)
        axsb.xaxis.set_ticks_position('top')
        axsb.yaxis.set_ticks_position('right')

        self.rotate_lables_for_xaxis(axsb, angle=50, alignment='left')

        #------------------
        # Panel C
        xrange  = (x,x+w,w)
        yrange  = (y,y+h,nx_slices)
        arr2dx  = fat.rebinArray(arr, xrange, yrange) 
        #axprojx = axsc.imshow(arr2dx, interpolation='nearest', origin='bottom', aspect='auto', extent=xyrange)

        for slice in range(nx_slices) :
            arr1slice = arr2dx[slice,...]
            xarr = np.linspace(x, x+w, num=w, endpoint=True)
            if arr1slice.sum() == 0 :
                print 'Empty histogram for slice =', slice,'is ignored'
                continue

            axsc.hist(xarr, bins=w, weights=arr1slice, histtype='step')

        axsc.set_xlim(x, x+w)

        #------------------
        # Panel D
        axsd.hist(arrwin.flatten(), bins=100)#, range=range)
        axsd.yaxis.set_ticks_position('right')
        self.rotate_lables_for_xaxis(axsd, angle=50, alignment='right')

#-----------------------------

    def rotate_lables_for_xaxis(self, axes, angle=50, alignment='center') : # 'right'
        """Rotate axis labels by anble in degree
        """
        for label in axes.get_xticklabels() :
            label.set_rotation(angle)
            label.set_horizontalalignment(alignment)

#-----------------------------

    def draw_outside_plots_for_list_of_objs(self, arr) :
        print 'ImgDrawProjXY : draw_outside_plots_for_list_of_objs(...)'

        self.arr = arr

        for obj in self.icp.list_of_rects :                  # <====== DEPENDS ON FORM
            if obj.myType == self.icp.typeProjXY :           # <====== DEPENDS ON FORM
                self.draw_prxy_for_rect(obj)                 # <====== DEPENDS ON FORM

#-----------------------------
# Test
#-----------------------------

def main():
    w = ImgDrawProjXY()

#-----------------------------

if __name__ == "__main__" :
    #main()
    sys.exit ('Module is not supposed to be run for test...')

#-----------------------------
