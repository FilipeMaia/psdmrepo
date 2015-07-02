#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotGraphWidget...
#
#------------------------------------------------------------------------

"""Plot array as a graphic and as a histogram.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: 

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import random
import numpy as np
from math import log10

from ViewResults import valueToIndexProtected

# For self-run debugging:
#if __name__ == "__main__" :
#    import matplotlib
#    matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)


#from   matplotlib.figure import Figure
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
#import matplotlib.ticker   as mtick
from   matplotlib.ticker   import MaxNLocator, NullFormatter
#from   matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from PyQt4 import QtGui, QtCore

#---------------------
#  Class definition --
#---------------------

class PlotGraphWidget (QtGui.QWidget) :
    """Plot array as a graphic and as a histogram"""

    def __init__(self, parent=None, arrays=None, figsize=(10,10), title='', axlabs=('','')):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('Matplotlib image embadded in Qt widget')
        self.setGeometry(10, 25, 1000, 700)
 
        # Expected shape arrsy.shape = (Ntau, Nq)
        self.arrays  = arrays
        self.title   = title
        self.parent  = parent
        self.figsize = figsize
        self.nbins   = 10

        self.xlab, self.ylab = axlabs # (r'$g_{2}$', r'$\tau$ (in number of frames)')

        self.fig = plt.figure(figsize=figsize, dpi=100, facecolor='w', edgecolor='w', frameon=True)

        #-----------------------------------
        self.canvas = self.fig.canvas
        self.vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        self.vbox.addWidget(self.canvas)        # <=== Add figure 
        #self.vbox.addStretch(1)
        self.setLayout(self.vbox)
        #-----------------------------------

        self.canvas.mpl_connect('button_press_event',   self.processMouseButtonPress) 
        self.canvas.mpl_connect('button_release_event', self.processMouseButtonRelease) 
        self.canvas.mpl_connect('motion_notify_event',  self.processMouseMotion)
        self.canvas.mpl_connect('axes_leave_event',     self.processAxesLeaveEvent)
        self.canvas.mpl_connect('axes_enter_event',     self.processAxesEnterEvent)
        self.canvas.mpl_connect('figure_leave_event',   self.processFigureLeaveEvent)

        self.setFrame()
        self.initParameters()
        self.on_draw()


    def setFrame(self) :
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def getCanvas(self) :
        return self.canvas


    def resizeEvent(self, e) :
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def closeEvent(self, event) : # is called for self.close() or when click on "x"
        #print 'PlotGraphWidget: closeEvent'
        pass


    def set_arrays_to_plot(self) :

        if self.arr_x is None : self.arrx = np.arange(self.arrsy.shape[1])
        else :                  self.arrx = self.arr_x

        bin_range   = (self.arr_n[0], self.arr_n[-1], self.nbins)
        bin_numbers = valueToIndexProtected(self.arr_n, bin_range)

        # Find min/max indexes for all bins:
        length = len(self.arr_n)
        # Protection if the requested number of bins too large
        if length < self.nbins : self.nbins = length

        self.bin_ind_min     = length*np.ones(self.nbins, dtype=np.uint)
        self.bin_ind_max     = np.zeros(self.nbins, dtype=np.uint)
        self.bin_ind_counter = np.zeros(self.nbins, dtype=np.uint)

        for i in range(length):
            bin = bin_numbers[i]
            if i<self.bin_ind_min[bin] : self.bin_ind_min[bin]=i
            if i>self.bin_ind_max[bin] : self.bin_ind_max[bin]=i
            self.bin_ind_counter[bin] += 1 

        #print 'bin_numbers:', bin_numbers
        #print 'bin_ind_min:', bin_ind_min
        #print 'bin_ind_max:', bin_ind_max
        #print 'bin_ind_counter:', bin_ind_counter

        self.list_of_arr_y = []

        for bin in range(self.nbins):
             imin, imax, ninds = self.bin_ind_min[bin], self.bin_ind_max[bin], self.bin_ind_counter[bin]
             #print 'bin:%2d, imin:%3d, imax:%3d, ninds:%3d' % ( bin, imin, imax, ninds )
             arrsy_for_bin = self.arrsy[imin:imax,:]
             arr_y = np.sum(arrsy_for_bin, axis=0)
             if ninds>1 : arr_y /= ninds
             #print 'arr_y = ', arr_y
             self.list_of_arr_y.append(arr_y)

    def initParameters(self) :
        self.gr_xmin  = None
        self.gr_xmax  = None
        self.gr_ymin  = None
        self.gr_ymax  = None
        self.gridIsOn = False
        self.logIsOn  = False


    def on_draw_in_limits(self) :
        self.on_draw(self.gr_xmin, self.gr_xmax, self.gr_ymin, self.gr_ymax)


    def processDraw(self) :
        """For backward compatability with PlotArrayButtons"""
        self.on_draw()


    def on_draw(self, gr_xmin=None, gr_xmax=None, gr_ymin=None, gr_ymax=None):
        """Redraws the figure"""

        self.fig.clear()

        self.axgr = self.fig.add_axes([0.08, 0.08, 0.9, 0.8]) # [x0, y0, width, height]

        self.arrsy, self.arr_x, self.arr_n = self.arrays

        if self.arr_n is not None :
            #self.axgr = self.fig.add_axes([0.08, 0.08, 0.6, 0.8]) # [x0, y0, width, height]
            self.on_draw_multigraphs()
            self.set_xyaxes_limits(self.arrx, np.array(self.list_of_arr_y).flatten(), gr_xmin, gr_xmax, gr_ymin, gr_ymax)

        else :
            self.on_draw_onegraph()
            self.set_xyaxes_limits(self.arr_x, self.arr_y, gr_xmin, gr_xmax, gr_ymin, gr_ymax)

        if self.logIsOn :
            self.axgr.set_xscale('log')
        else :
            self.axgr.xaxis.set_major_locator(MaxNLocator(5))

        self.axgr.set_title(self.title, fontsize=14, color='k')
        self.axgr.tick_params(axis='both', which='major', labelsize=8)
        self.axgr.yaxis.set_major_locator(MaxNLocator(5))
        self.axgr.grid(self.gridIsOn)

        self.axgr.set_xlabel(self.xlab, fontsize=14)
        self.axgr.set_ylabel(self.ylab, fontsize=14)
        #self.axgr.xaxis.set_major_formatter(NullFormatter())

        self.list_of_axgr = []
        self.list_of_axgr.append(self.axgr)

        self.canvas.draw()


    def set_xyaxes_limits(self, arrx, arry, gr_xmin=None, gr_xmax=None, gr_ymin=None, gr_ymax=None):

        if gr_xmin is None : xmin = arrx[0]
        else             : xmin = gr_xmin

        if gr_xmax is None : xmax = arrx[-1] # Last element
        else             : xmax = gr_xmax

        if xmin==xmax : xmax=xmin+1 # protection against equal limits

        if gr_ymin is None : ymin = 0 # min(yarr)
        else             : ymin = gr_ymin

        if gr_ymax is None : ymax = 1.05*max(arry)
        else             : ymax = gr_ymax

        self.axgr.set_xlim(xmin,xmax) 
        self.axgr.set_ylim(ymin,ymax) 



    def on_draw_multigraphs(self):
        #------------------------
        self.set_arrays_to_plot()
        #------------------------

        self.list_of_gr = []
        list_of_markers = ('o','v','^','s','*','h','H','D','<','>','p') #,'1','2','3','4','+','x'
        list_of_colors  = ('b','g','r','c','m','y','k') #,'w'

        for bin in range(self.nbins) : 
            xarr = self.arrx
            yarr = self.list_of_arr_y[bin]
            imin, imax = self.bin_ind_min[bin], self.bin_ind_max[bin]
            s='t:%6.1f-%6.1f' % (self.arr_n[imin], self.arr_n[imax])
            opt = '-' + list_of_colors[bin%len(list_of_colors)] + list_of_markers[bin%len(list_of_markers)] # for example: '-bo'
            gr, = self.axgr.plot(xarr, yarr, opt, label=s)# '-ro'
            self.list_of_gr.append(gr) 

        handles, labels = self.axgr.get_legend_handles_labels()
        self.axgr.legend(handles, labels, bbox_to_anchor=(0.65, 0.99), loc=2, borderaxespad=0., title='', numpoints=1, labelspacing=0)
        #self.axgr.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., title='', numpoints=1, labelspacing=0)


    def on_draw_onegraph(self):
        self.arr_y, self.arr_x, self.arr_n = self.arrays
        if self.arr_n is not None : return
        self.axgr.plot(self.arr_x, self.arr_y, '-bo')


    def processAxesEnterEvent(self, event) :
        #print 'AxesEnterEvent'
        if self.event_is_in_axgr(event) :
            #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
            QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.SizeHorCursor))
            #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.SizeAllCursor))


    def processAxesLeaveEvent(self, event) :
        #print 'AxesLeaveEvent'
        try : self.curstext.remove()
        except : pass
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))


    def processFigureLeaveEvent(self, event) :
        #print 'FigureLeaveEvent'
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))


    def event_is_in_axgr(self, event) :
        for axgr in self.list_of_axgr :
            if event.inaxes ==  axgr : return True
        return False


    def processMouseMotion(self, event) :
        if self.event_is_in_axgr(event) :
            self.drawXCoordinateOfCoursor(event)
            self.drawVerticalLineThroughCoursor(event)
        else :
            pass


    def drawXCoordinateOfCoursor(self, event) :
        axes = event.inaxes
        #xmin, xmax = axes.get_xlim()
        #ymin, ymax = axes.get_ylim()
        x, y = event.xdata, event.ydata
        s = '%6.3f' % (event.xdata)
        #s = '%6.1f,%6.1f' % (event.xdata, event.ydata)
        try : self.curstext.remove()
        except : pass
        self.curstext = axes.text(x, y, s) #, ha='center')
        self.canvas.draw()


    def drawVerticalLineThroughCoursor(self, event) :
        axes   = event.inaxes
        fb     = self.canvas.figure.bbox
        bb     = axes.bbox
        #print bb

        bbx0, bby0, bbh, bbw = bb.x0, bb.y0, bb.height, bb.width
        fbx0, fby0, fbh, fbw = fb.x0, fb.y0, fb.height, fb.width

        xd = event.xdata
        yd = event.ydata
        x = event.x
        y = event.y

        x0 = bbx0-1 
        y0 = fbh - bby0 - bbh
        w  = x - x0

        rect = [x0, y0, w, bbh]
        self.fig.canvas.drawRectangle( rect )            
        #self.fig.canvas.draw()


    def processMouseButtonPress(self, event) :
        #print 'MouseButtonPress'
        if self.event_is_in_axgr(event) : self.mousePressOnGraph(event)


    def mousePressOnGraph(self, event) :
        #print 'PressOnGraph'
        #print 'event.xdata, ydata, x, y =', event.xdata, event.ydata, event.x, event.y

        if   event.button == 1 :
            self.gr_xmin = float(event.xdata)
        elif event.button == 3 :
            self.gr_xmax = float(event.xdata)
        else :
            self.gr_xmin = None
            self.gr_xmax = None

        self.on_draw_in_limits()


    def processMouseButtonRelease(self, event) :
        #print 'MouseButtonRelease'

        if event.button == 1 :
            pass

        elif event.button == 2 : # middle or right button
            if self.event_is_in_axgr(event) : 
                self.gr_xmin = None
                self.gr_xmax = None

            #elif event.inaxes == self.axhi :
            #    self.gr_ymin = None
            #    self.gr_ymax = None

            self.on_draw_in_limits()
            #self.on_draw()

        elif event.button == 3 :
            pass


    def saveFigure(self, fname='fig.png') :
        self.fig.savefig(fname)

#-----------------------------
# Test
#-----------------------------

def get_arrays_for_test() :
    rows, cols = 31, 20 # for q and tau
    mu, sigma = 1., 0.2
    arrsy = mu + sigma*np.random.standard_normal( size = rows*cols )
    arrsy.shape = (rows,cols)
    arr_x = np.arange(cols)
    arr_n   = np.arange(rows)
    #return arrsy, arr_x, arr_n
    return arrsy[0,:], arr_x, None

def print_array(arr, msg='') :
    print '\n' + msg + ':\n', arr
    print 'shape:', arr.shape

#-----------------------------

def main() :

    app = QtGui.QApplication(sys.argv)
    w = PlotGraphWidget(None, get_arrays_for_test(), axlabs=(r'$g_{2}$', r'$\tau$ '))    
    w.move(QtCore.QPoint(50,50))
    w.show()    
    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
