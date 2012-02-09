#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgDrawProfile...
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


import matplotlib.pyplot as plt
import ImgFigureManager  as imgfm

#---------------------
#  Class definition --
#---------------------

class ImgDrawProfile :
    """Additional graphics outside of the 2-d image for profile"""

    def __init__(self, icp=None):
        self.icp           = icp
        self.icp.idrawprof = self
        self.arr           = None

        #self.icp.list_of_rects = [] # moved to icp

    def get_control(self) :
        return self.icp.control


    def draw_profile_for_line(self,obj) :                                            # <====== DEPENDS ON FORM INSIDE
        #x,y,w,h,lw,col,s,t,r = obj.get_list_of_rect_pars() 
        x1,x2,y1,y2,lw,col,s,t,r = obj.get_list_of_line_pars()

        print 'draw_profile_for_line : x1,x2,y1,y2,lw,col,s,t,r = ', x1,x2,y1,y2,lw,col,s,t,r

        plt.ion()

        fig_outside = obj.get_fig_outside()                    # Get figure from the object

        if fig_outside == None : # if figure is not open yet
        
            fig_outside = imgfm.ifm.get_figure(figsize=(5,4), type='type1', icp=self.icp) # type='maxspace'
            obj.set_fig_outside(fig_outside)                   # Preserve figure in the object
            fig_outside.obj_index = obj.myIndex 
            fig_outside.my_object = obj
            fig_outside.canvas.set_window_title('Profile for line %d' % obj.myIndex  )
            print 'fig_outside.number=',fig_outside.number

        else :

            if not obj.isChanged : return
            fig_outside.clear()
            fig_outside.canvas.set_window_title('Profile for line %d' % fig_outside.obj_index )

        if s :   # Re-draw the figure window on top

            fig_outside.canvas.manager.window.activateWindow() # Makes window active
            fig_outside.canvas.manager.window.raise_()         # Move window on top

        #arrwin =  self.arr[y:y+h,x:x+w]
        #axsb = fig_outside.add_subplot(111)
        #axsb.hist(arrwin.flatten(), bins=100)#, range=range)

        self.drawProfileAlongLine(fig_outside, self.arr, obj)

        fig_outside.canvas.draw()

        obj.isChanged = False

#-----------------------------

    def drawProfileAlongLine(self, fig, arr, obj) :

        arr2d = np.array(arr)
        x1,x2,y1,y2,lw,col,s,t,r = obj.get_list_of_line_pars()

        xmin = min(x1,x2)
        xmax = max(x1,x2)
        ymin = min(y1,y2)
        ymax = max(y1,y2)
        print 'xmin, xmax, ymin, ymax = ', xmin, xmax, ymin, ymax

        profile = []
        tit_add = ''
        if xmax-xmin > ymax-ymin :
            print 'Plot profile for X bins'
            tit_add = 'X-binning'
            if x1 == xmin: ysta, yend = y1, y2
            else         : ysta, yend = y2, y1
            k = float(yend-ysta)/float(xmax-xmin)
            arrX = np.arange(xmin,xmax,dtype=np.int16)

            for x in arrX :
                y = int( k*(x-xmin) + ysta )
                profile.append(arr2d[y,x])
                #print 'x,y=',x,y,'   profile=',arr2d[y,x]
 
        else :
            print 'Plot profile for Y bins'
            tit_add = 'Y-binning'
            if y1 == ymin: xsta, xend = x1, x2
            else         : xsta, xend = x2, x1
            k = float(xend-xsta)/float(ymax-ymin)
            arrX = np.arange(ymin,ymax,dtype=np.int16)

            for y in arrX :
                x = int( k*(y-ymin) + xsta )
                profile.append(arr2d[y,x])
                #print 'x,y=',x,y,'   profile=',arr2d[y,x]

        arrY = np.array(profile)

        axsb = fig.add_subplot(111)
        axsb.hist(arrX, bins=arrX.shape[0], weights=arrY, histtype='step')
        #plt.ylim(0,4000)
        axsb.grid()
        fig.canvas.set_window_title('Profile along the line '+str(obj.myIndex) + ' : ' + tit_add )
        #plt.savefig('plot-profile-along-line-' + str(obj.myIndex) + '-' + imp.impars.plot_fname_suffix + '.png')


#-----------------------------

    def draw_outside_plots_for_list_of_objs(self, arr) :
        print 'ImgDrawProfile : draw_outside_plots_for_list_of_objs(...)'

        self.arr = arr

        for obj in self.icp.list_of_lines :                     # <====== DEPENDS ON FORM
            if obj.myType == self.icp.typeProfile :             # <====== DEPENDS ON FORM
                self.draw_profile_for_line(obj)                 # <====== DEPENDS ON FORM

#-----------------------------
# Test
#-----------------------------

def main():
    w = ImgDrawProfile()

#-----------------------------

if __name__ == "__main__" :
    #main()
    sys.exit ('Module is not supposed to be run for test...')

#-----------------------------
