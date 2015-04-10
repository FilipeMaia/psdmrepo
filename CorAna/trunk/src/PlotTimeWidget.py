#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotTimeWidget...
#
#------------------------------------------------------------------------

"""Plots image and spectrum for 2d numpy array.

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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from   matplotlib.ticker import MaxNLocator
#from   matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from   EventTimeRecords import *

from PyQt4 import QtGui, QtCore

#---------------------
#  Class definition --
#---------------------

class PlotTimeWidget (QtGui.QWidget) :
    """Plots image and spectrum for 2d numpy array."""

    def __init__(self, parent=None, ifname=None, figsize=(10,5)):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('Matplotlib image embadded in Qt widget')

        #print 'figsize=', figsize

        self.figsize= figsize
        self.ifname = ifname

        self.get_array_from_file()

        self.fig = plt.figure(figsize=figsize, dpi=100, facecolor='w',edgecolor='w',frameon=True)
        #self.fig = Figure(    figsize=(5,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)
        #print 'fig.number =', self.fig.number

        #-----------------------------------
        #self.canvas = FigureCanvas(self.fig)
        self.canvas = self.fig.canvas
        self.vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        self.vbox.addWidget(self.canvas)        # <=== Add figure 
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


    def get_array_from_file(self):
        etr = EventTimeRecords (self.ifname)
        #etr.print_arr_for_plot()
        self.arr = etr.get_arr_for_plot()


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
        #print 'PlotTimeWidget: closeEvent'
        pass
        plt.close(self.fig.number)


    def set_image_array(self,arr):
        self.arr = arr
        self.processDraw()


    def initParameters(self) :
        self.gr_xmin  = None
        self.gr_xmax  = None
        self.gr_ymin  = None
        self.gr_ymax  = None
        self.gridIsOn = False
        self.logIsOn  = False
        self.nbins    = 100


    def processDraw(self) :
        self.on_draw(self.gr_xmin, self.gr_xmax, self.gr_ymin, self.gr_ymax, self.nbins)


    def on_draw(self, gr_xmin=None, gr_xmax=None, gr_ymin=None, gr_ymax=None, nbins=100):
        """Redraws the figure"""

        self.fig.clear()

        arr = self.arr
        xarr = arr[:,3]
        yarr = arr[:,2]
        iarr = arr[:,0]

        if gr_xmin is None : xmin = xarr[1]
        else               : xmin = gr_xmin

        if gr_xmax is None : xmax = xarr[-1] # Last element
        else               : xmax = gr_xmax

        yarrwin = yarr[int(xmin):int(xmax)]
        xarrwin = xarr[int(xmin):int(xmax)]
        iarrwin = iarr[int(xmin):int(xmax)]

        if gr_ymin is None : ymin = min(yarrwin)
        else               : ymin = gr_ymin

        if gr_ymax is None : ymax = max(yarrwin)
        else               : ymax = gr_ymax

        hi_range = (ymin, ymax)

        self.axgr = self.fig.add_axes([0.1,  0.64, 0.80, 0.35])
        self.axhi = self.fig.add_axes([0.1,  0.14, 0.35, 0.35])
        self.axti = self.fig.add_axes([0.55, 0.14, 0.35, 0.35])

        self.axhi.xaxis.set_major_locator(MaxNLocator(4))
        self.axti.xaxis.set_major_locator(MaxNLocator(4))
        self.axhi.yaxis.set_major_locator(MaxNLocator(4))
        self.axti.yaxis.set_major_locator(MaxNLocator(4))
        self.axgr.yaxis.set_major_locator(MaxNLocator(4))

        self.axti.plot(xarrwin, iarrwin, '-g')
        self.axti.set_xlabel('Event index')
        self.axti.set_ylabel('Time index')
        self.axti.set_xlim(xmin,xmax) 

        #ax1.plot(tau, cor1, '-r', tau, cor2, '-g')
        #self.axgr.plot(xarr,    yarr,    '-r')
        self.axgr.plot(xarrwin, yarrwin, '-r')
        self.axgr.set_xlim(xmin,xmax) 
        self.axgr.set_ylim(ymin,ymax) 

        self.axgr.set_xlabel('Time index')
        self.axgr.set_ylabel('dt(sec)')

        self.axhi.hist(yarrwin, bins=nbins, range=hi_range, log=self.logIsOn)
        self.axhi.set_xlabel('dt(sec)')
        self.axhi.set_ylabel('Frames')

        self.axgr.grid(self.gridIsOn)
        self.axhi.grid(self.gridIsOn)
        self.axti.grid(self.gridIsOn)

        self.canvas.draw()
        #print 'End of on_draw'


    def processAxesEnterEvent(self, event) :
        #print 'AxesEnterEvent'
        if event.inaxes == self.axhi or event.inaxes == self.axgr or event.inaxes == self.axti :
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


    def processMouseMotion(self, event) :
        if event.inaxes == self.axhi or event.inaxes == self.axgr or event.inaxes == self.axti :
            self.drawXCoordinateOfCoursor(event)
            self.drawVerticalLineThroughCoursor(event)


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
        y0 = fbh  - bby0 - bbh -1
        w  = x - x0

        rect = [x0, y0, w, bbh]
        self.fig.canvas.drawRectangle( rect )            
        #self.fig.canvas.draw()


    def processMouseButtonPress(self, event) :
        #print 'MouseButtonPress'
        if event.inaxes == self.axgr     : self.mousePressOnGraph(event)
        if event.inaxes == self.axti     : self.mousePressOnGraph(event)
        if event.inaxes == self.axhi     : self.mousePressOnHisto(event)


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

        self.processDraw()


    def mousePressOnHisto(self, event) :
        #print 'PressOnHistogram'
        #lims = self.axhi.get_xlim()
        #print 'event.xdata, ydata, x, y =', event.xdata, event.ydata, event.x, event.y

        if   event.button == 1 :
            self.gr_ymin = float(event.xdata)
        elif event.button == 3 :
            self.gr_ymax = float(event.xdata)
        else :
            self.gr_ymin = None
            self.gr_ymax = None

        self.processDraw()


    def processMouseButtonRelease(self, event) :
        #print 'MouseButtonRelease'

        if event.button == 1 :
            pass

        elif event.button == 2 : # middle or right button
            if event.inaxes == self.axgr or event.inaxes == self.axti : 
                self.gr_xmin = None
                self.gr_xmax = None

            elif event.inaxes == self.axhi :
                self.gr_ymin = None
                self.gr_ymax = None

            self.processDraw()
            #self.on_draw()

        elif event.button == 3 :
            pass


#    def stringOrNone(self,value):
#        if value is None : return 'None'
#        else             : return str(value)


#    def intOrNone(self,value):
#        if value is None : return None
#        else             : return int(value)


    def saveFigure(self, fname='fig.png'):
        self.fig.savefig(fname)

#-----------------------------
# Test
#-----------------------------

def get_array2d_for_test() :
    mu, sigma = 200, 25
    #arr = mu + sigma*np.random.standard_normal(size=2400)
    arr = 100*np.random.standard_exponential(size=2400)
    #arr = np.arange(2400)
    arr.shape = (40,60)
    return arr

#-----------------------------

def main():

    app = QtGui.QApplication(sys.argv)
    #fname = 'work/cora-xcsi0112-r0015-data-scan-tstamp-list.txt'
    fname = '/reg/neh/home1/dubrovin/LCLS/PSANA-V01/work-1/t1-xcsi0112-r0015-data-scan-tstamp-list.txt'
    w = PlotTimeWidget(None, fname, figsize=(10,5))
    w.move(QtCore.QPoint(50,50))
    w.show()    
    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
