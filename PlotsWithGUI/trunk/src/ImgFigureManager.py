#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgFigureManager...
#
#------------------------------------------------------------------------

"""Uniform place for manipulation with figures

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
import numpy as np # for test only

import matplotlib
#matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
import matplotlib.pyplot as plt

#---------------------
#  Class definition --
#---------------------

class ImgFigureManager :
    """Uniform place for manipulation with figures"""

    def __init__(self) :
        self.list_of_open_figs = []


    def get_figure(self, num=None, figsize=(5,10), type=None) :
        """Get fig, make a new figure object if it is not created yet."""

        fig = plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w',edgecolor='w',frameon=True)

        if fig.number not in self.list_of_open_figs :
            self.list_of_open_figs.append(fig.number)

        if type == None :
            return fig

        if type == 'type1' :
            fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        elif type == 'type2' :
            fig.subplots_adjust(left=0.08, bottom=0.02, right=0.98, top=0.98, wspace=0.2, hspace=0.1)
        return fig


    def close_fig( self, num=None ) :
        """Close fig and its window."""
        if num==None :
            plt.close('all') # closes all the figure windows
        else :
            plt.close(num)
            if num in self.list_of_open_figs : self.list_of_open_figs.remove(num)


    def print_list_of_open_figs( self ) :
        print 'ImgFigureManager: List of open figs:',
        for num in self.list_of_open_figs : 
            print num,
        print ''


#    def get_figure_number(self) :
#        return self.fig.number

ifm = ImgFigureManager()

#-----------------------------
# Test
#-----------------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=2400)
    #arr = np.arange(2400)
    arr.shape = (40,60)
    return arr


def set_colormap() :
    pass
    plt.autumn() # yellow-red-orange
    #plt.winter()
    #plt.spring() #grinish
    #plt.summer()
    #plt.bone() # grey
    #plt.cool() # blue
    #plt.copper() # 
    #plt.gray() # 
    #plt.hot() # yellow-red-orange
    #plt.hsv() # default?
    #plt.jet() # default? 
    #plt.pink() # 
    #plt.flag() # 
    #plt.prism() # 
    #plt.spectral() # 


def main() :

    fig1  = ifm.get_figure(figsize=(7,5))
    plt.get_current_fig_manager().window.move(10,10)
    axes1 = fig1.add_subplot(111)
    axes1.imshow( get_array2d_for_test(), interpolation='nearest', origin='bottom', aspect='auto') #, cmap=matplotlib.cm.gray, extent=self.range
    print 'figure number =', fig1.number

    set_colormap()

    fig2  = ifm.get_figure(figsize=(7,5))
    plt.get_current_fig_manager().window.move(100,100)
    axes2 = fig2.add_subplot(111)
    axes2.imshow( get_array2d_for_test(), interpolation='nearest', origin='bottom', aspect='auto') #, cmap=matplotlib.cm.gray, extent=self.range
    print 'figure number =', fig2.number

    ifm.print_list_of_open_figs()

    plt.show()

#-----------------------------
# Test
#
if __name__ == "__main__" :
    main()
    sys.exit ('End of test')
#-----------------------------
