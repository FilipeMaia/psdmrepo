#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotG2Widget...
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

# For self-run debugging:
if __name__ == "__main__" :
    import matplotlib
    matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)

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

class PlotG2Widget (QtGui.QWidget) :
    """Plot array as a graphic and as a histogram"""

    def __init__(self, parent=None, arrays=None, figsize=(10,10), title=''):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('Matplotlib image embadded in Qt widget')
        self.setGeometry(10, 25, 1000, 700)
 
        self.arr_g2, self.arr_tau, self.arr_q = arrays
        # Expected shape arr_g2.shape = (Ntau, Nq)

        self.set_xarray(np.array(self.arr_tau))
        self.title   = title
        self.parent  = parent
        self.figsize = figsize
        self.nwin_max = 9

        self.fig = plt.figure(figsize=figsize, dpi=100, facecolor='w',edgecolor='w',frameon=True)

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


    def setFrame(self):
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameStyle( QtGui.QFrame.Box | QtGui.QFrame.Sunken )
        self.frame.setLineWidth(0)
        self.frame.setMidLineWidth(1)
        self.frame.setGeometry(self.rect())
        #self.frame.setVisible(False)


    def getCanvas(self):
        return self.canvas


    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        #print 'PlotG2Widget: closeEvent'
        pass


    #def set_array(self, arr, title=''):
    #    self.arry  = arr
    #    self.title = title
    #    self.on_draw()
    #    #self.on_draw_in_limits()


    def set_xarray(self,arr):
        if arr is None :
            self.arrx = np.arange(self.arr_g2.shape[0])
        else :
            self.arrx = arr


    def initParameters(self) :
        self.gr_xmin  = None
        self.gr_xmax  = None
        self.gr_ymin  = None
        self.gr_ymax  = None
        self.gridIsOn = False
        self.logIsOn  = True
        self.iq_begin = 0


    def get_iq_list(self, iq_begin=0) :
        iq_min   = 0
        iq_max   = self.arr_q.shape[0]
        #return [0,1,2,3,4,5,6,7,8]

        if   iq_max <= self.nwin_max          : return range(iq_max)
        elif iq_max-iq_begin <= self.nwin_max : return range(iq_max-self.nwin_max, iq_max)
        else                                  : return range(iq_begin, iq_begin+self.nwin_max)



    def on_draw_in_limits(self) :
        self.on_draw(self.gr_xmin, self.gr_xmax, self.gr_ymin, self.gr_ymax, self.iq_begin)



    def on_draw(self, gr_xmin=None, gr_xmax=None, gr_ymin=None, gr_ymax=None, iq_begin=0):
        """Redraws the figure"""

        self.fig.clear()

        if gr_xmin is None : xmin = self.arrx[0]
        else               : xmin = gr_xmin

        if gr_xmax is None : xmax = self.arrx[-1] # Last element
        else               : xmax = gr_xmax

        if xmin==xmax : xmax=xmin+1 # protection against equal limits

        wwidth  = 0.26 
        wheight = 0.24 

        self.list_of_axgr = []

        #iq_begin = 5
        iq_list  = self.get_iq_list(iq_begin)
        #print 'iq_list:', iq_list, ' at self.iq_max =',self.arr_q.shape[0]

        for iwin, iq in enumerate(iq_list) :

            iwin_row = int(iwin/3)
            iwin_col = int(iwin%3)
            wx0      = 0.08 + iwin_col*0.32
            wy0      = 0.70 - iwin_row*0.3

            xarr =  self.arrx
            yarr  = self.arr_g2[:,iq]
            q_ave = self.arr_q[iq]
            q_str = 'q(%d)=%8.4f' % (iq, q_ave) 

            if gr_ymin is None : ymin = min(yarr)
            else               : ymin = gr_ymin

            if gr_ymax is None : ymax = max(yarr)
            else               : ymax = gr_ymax

            axgr = self.fig.add_axes([wx0, wy0, wwidth, wheight])
            if self.logIsOn :
                axgr.set_xscale('log')
            else :
                axgr.xaxis.set_major_locator(MaxNLocator(5))

            axgr.plot(xarr, yarr, '-bo')# '-ro'

            axgr.set_xlim(xmin,xmax) 
            axgr.set_ylim(ymin,ymax) 
            axgr.set_title(q_str, fontsize=10, color='b')
            axgr.tick_params(axis='both', which='major', labelsize=8)
            axgr.yaxis.set_major_locator(MaxNLocator(5))
            axgr.grid(self.gridIsOn)


            if iwin_col == 0 :
                axgr.set_ylabel(r'$g_{2}$', fontsize=14)
            if iwin_row == 2 :
                axgr.set_xlabel(r'$\tau$ (in number of frames)', fontsize=12)
            else :
                axgr.xaxis.set_major_formatter(NullFormatter())

            self.list_of_axgr.append(axgr)

            self.canvas.draw()


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
        s = '%6.1f' % (event.xdata)
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

        x0 = bbx0 
        y0 = fbh  - bby0 - bbh # -1
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


    def saveFigure(self, fname='fig.png'):
        self.fig.savefig(fname)

#-----------------------------
# Test
#-----------------------------

def get_arrays_for_test() :
    rows, cols = 31, 20 # for q and tau
    mu, sigma = 1., 0.2
    arr_g2 = mu + sigma*np.random.standard_normal( size = rows*cols )
    arr_g2.shape = (rows,cols)
    arr_tau = np.arange(rows)
    arr_q   = np.arange(cols)
    return arr_g2, arr_tau, arr_q

def print_array(arr, msg='') :
    print '\n' + msg + ':\n', arr
    print 'shape:', arr.shape

#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)
    w = PlotG2Widget(arrays=get_arrays_for_test())
    w.move(QtCore.QPoint(50,50))
    w.show()    
    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
