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


from PyQt4 import QtGui, QtCore

#---------------------
#  Class definition --
#---------------------

class ImgFigureManager :
    """Uniform place for manipulation with figures"""

    def __init__(self) :
        self.list_of_open_figs  = []


    def get_figure(self, num=None, figsize=(5,10), type=None, icp=None) :
        """Get fig, make a new figure object if it is not created yet."""

        fig = plt.figure(num=num, figsize=figsize, dpi=100, facecolor='w',edgecolor='w',frameon=True)
        fig.my_icp              = icp
        fig.my_object           = None
        fig.my_close_mode       = None

        if fig not in self.list_of_open_figs :
            self.list_of_open_figs.append(fig)
            print 'Add figure number =',  fig.number, 'in the list' 

        if   type == None :
            pass

        elif type == 'type1' :
            fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)

        elif type == 'maxspace' :
            fig.subplots_adjust(left=0.06, bottom=0.03, right=0.99, top=0.99, wspace=0, hspace=0)

        elif type == 'test' :
            return fig

        fig.canvas.mpl_connect('close_event',        self.onCloseEvent)       # In order to close window by click on X
        fig.canvas.mpl_connect('button_press_event', self.onButtonPressEvent) # In order to "actiwate" the window and assoc. obj.

        #qtwidget = fig.canvas.manager.window
        #qtwidget = QtGui.QWidget(plt.get_current_fig_manager().window)
        #qtwidget.connect(qtwidget, QtCore.SIGNAL('mousePressEvent()'), self.onButtonPressEvent)
        #qtwidget.connect(qtwidget, QtCore.SIGNAL('clicked()'), self.onButtonPressEvent)
        #fig.canvas.connect(qtwidget, QtCore.SIGNAL('clicked()'), self.onWindowPressEvent)
        #qtwidget.clicked.connect(self.onWindowPressEvent)

        self.set_fig_window_default_position(fig)
        return fig


    def set_fig_window_default_position(self,fig) : 
        wmain_pos  = fig.my_icp.control.getPosition()
        #fig_window = plt.get_current_fig_manager().window
        fig_window = fig.canvas.manager.window
        #wmain_size = fig.my_icp.control.getSize()
        #fig_window.move(wmain_pos[0]+wmain_size[0]+50*(fig.number-2), wmain_pos[1]+20*(fig.number-2))
        fig_window.move(900+50*(fig.number-2), 10+20*(fig.number-2))


    def get_fig_window_position(self, fig) : 
        #return plt.get_current_fig_manager().window.pos()
        return fig.canvas.manager.window.pos()


    def set_fig_window_position(self, fig, pos) : 
        #plt.get_current_fig_manager().window.move(pos.x(), pos.y())
        fig.canvas.manager.window.move(pos.x(), pos.y())


    def print_list_of_open_figs( self ) :
        print 'ImgFigureManager: List of open figs:',
        for fig in self.list_of_open_figs : 
            print fig.number,
        print ''


    def get_figure_by_number( self, num ) :
        for fig in self.list_of_open_figs : 
            if fig.number == num : return fig
        return None


    # This is a colse figure request from program
    def close_fig( self, num=None ) :
        """Close fig and its window."""
        print 'ImgFigureManager : close_fig() program call to close fig number =', num

        if num==None :
            plt.close('all') # closes all the figure windows
        else :
            fig = self.get_figure_by_number(num)
            fig.my_close_mode = 'Call'
            plt.close( num ) # sends signal 'close_event' will auto-call the onCloseEvent(...)


    # This is always automatically envoked when figure is closing
    def onCloseEvent( self, event ):
        """Figure will be closed automatically, but it is necesary to remove its number from the list..."""
        print 'ImgFigureManager : onCloseEvent() close event, fig number =', event.canvas.figure.number

        fig = event.canvas.figure # plt.gcf() does not work, because closed canva may be non active

        if fig in self.list_of_open_figs : 

            if  fig.my_object != None : # if the object, associated with figure is not yet removed.

                #fig.my_icp.control.signal_outside_fig_is_closing_by_call (fig)
                if fig.my_close_mode == 'Call' : fig.my_icp.control.signal_outside_fig_is_closing_by_call (fig)
                else                           : fig.my_icp.control.signal_outside_fig_is_closing_by_click(fig)

            self.list_of_open_figs.remove(fig)


    def onButtonPressEvent( self, event ):
        """Figure is picked"""
        #print 'ImgFigureManager : onButtonPressEvent(), fig number =', event.canvas.figure.number
        fig    = event.canvas.figure
        fig.my_icp.control.signal_figure_is_selected(fig)



    def onWindowPressEvent( self, event ):
        """Figure is picked"""
        print 'ImgFigureManager : onWindowPressEvent(), fig number =', event.canvas.figure.number

#-----------------------------
#-----------------------------
ifm = ImgFigureManager()
#-----------------------------
#-----------------------------

#-----------------------------
# Test
#-----------------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25
    arr = mu + sigma*np.random.standard_normal(size=2400)
    #arr = np.arange(2400)
    arr.shape = (40,60)
    return arr


def set_colormap(name = None) :
    if   name== None :
        return
    elif name=='autumn' :
        plt.    autumn()    # yellow-red-orange
    elif name=='winter' :
        plt.    winter()    # blue-green
    elif name=='spring' :
        plt.    spring()    # yellow-purple
    elif name=='summer' :
        plt.    summer()    # grinish
    elif name=='bone' :
        plt.    bone()      # grey
    elif name=='cool' :
        plt.    cool()      # blueish
    elif name=='copper' :
        plt.    copper()    # monotone - red
    elif name=='gray' :
        plt.    gray()      # monotone
    elif name=='hot' :
        plt.    hot()       # flame colors yellow-red
    elif name=='hsv' :
        plt.    hsv()       # violet-blue-green
    elif name=='jet' :
        plt.    jet()       # brighter than default?
    elif name=='pink' :
        plt.    pink()      # monotone - pinkish
    elif name=='flag' :
        plt.    flag()      # very contrast
    elif name=='prism' :
        plt.    prism()     # bright blue-yellow-red-green
    elif name=='spectral' :
        plt.    spectral()  # green-blue-red-yellow
    

def main() :

    fig1  = ifm.get_figure(figsize=(7,5), type='test')
    plt.get_current_fig_manager().window.move(10,10)
    axes1 = fig1.add_subplot(111)
    axes1.imshow( get_array2d_for_test(), interpolation='nearest', origin='bottom', aspect='auto')
                  #, cmap=matplotlib.cm.gray, extent=self.range
    print 'figure number =', fig1.number

    set_colormap('spectral')

    fig2  = ifm.get_figure(figsize=(7,5), type='test')
    plt.get_current_fig_manager().window.move(100,100)
    axes2 = fig2.add_subplot(111)
    axes2.imshow( get_array2d_for_test(), interpolation='nearest', origin='bottom', aspect='auto')
                  #, cmap=matplotlib.cm.gray, extent=self.range
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
