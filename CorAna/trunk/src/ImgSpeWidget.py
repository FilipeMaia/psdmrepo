#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ImgSpeWidget...
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
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import random
import numpy as np
from math import log10

try : 
    import matplotlib
   #matplotlib.use('GTKAgg') # forse Agg rendering to a GTK canvas (backend)
    matplotlib.use('Qt4Agg') # forse Agg rendering to a Qt4 canvas (backend)
except :
    pass

#from   matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from   matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from PyQt4 import QtGui, QtCore

#---------------------
#  Class definition --
#---------------------

class ImgSpeWidget (QtGui.QWidget) :
    """Plots image and spectrum for 2d numpy array."""

    def __init__(self, parent=None, arr=None):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('Matplotlib image embadded in Qt widget')
        self.arr = arr        
        self.fig = plt.figure(num=1, figsize=(5,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)
       #self.fig = Figure(          figsize=(5,10), dpi=100, facecolor='w',edgecolor='w',frameon=True)

        #-----------------------------------
        #self.canvas = FigureCanvas(self.fig)
        self.canvas = self.fig.canvas
        vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        vbox.addWidget(self.canvas)        # <=== Add figure 
        self.setLayout(vbox)
        #-----------------------------------

        self.canvas.mpl_connect('button_press_event',   self.processMouseButtonPress) 
        self.canvas.mpl_connect('button_release_event', self.processMouseButtonRelease) 
        self.canvas.mpl_connect('motion_notify_event',  self.processMouseMotion)

        self.initParameters()
        self.setFrame()
        if self.arr != None : self.on_draw()


    def initParameters(self):
        self.fig.myXmin      = None
        self.fig.myXmax      = None
        self.fig.myYmin      = None
        self.fig.myYmax      = None
        self.fig.myZmin      = None
        self.fig.myZmax      = None
        self.fig.myNBins     = 100
        self.fig.myGridIsOn  = False
        self.fig.myLogIsOn   = False
        self.fig.myZoomIsOn  = False
        self.fig.ntbZoomIsOn = False

        self.xpress    = 0
        self.ypress    = 0
        self.xpressabs = 0
        self.ypressabs = 0


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
        #print 'ImgSpeWidget: closeEvent'
        pass


    def set_image_array(self,arr):
        self.arr = arr
        self.processDraw()


    def processDraw(self) :
        #fig = event.canvas.figure
        f = self.fig
        self.on_draw(f.myXmin, f.myXmax, f.myYmin, f.myYmax, f.myZmin, f.myZmax, f.myNBins)


    def on_draw(self, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, nbins=100):
        """Redraws the figure"""

        if xmin == None or xmax == None or ymin == None or ymax == None :
            self.arrwin  = self.arr
            self.range   = None # original image range in pixels
        else :
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)

            #print 'xmin, xmax, ymin, ymax =', xmin, xmax, ymin, ymax
            self.arrwin =  self.arr[ymin:ymax,xmin:xmax]
            self.range  = [xmin, xmax, ymax, ymin]

        zmin = self.intOrNone(zmin)
        zmax = self.intOrNone(zmax)

        if zmin==None and zmax==None : self.range_his = None
        else                         : self.range_his = (zmin,zmax)

        #print 'self.range_his = ', self.range_his
        #print 'self.arrwin = ', self.arrwin

        self.fig.clear()        
        self.nbins = nbins

        if self.fig.myLogIsOn : self.plots_in_log10_scale()
        else :                  self.plots_in_linear_scale()

        self.axhis.grid(self.fig.myGridIsOn)
        self.aximg.grid(self.fig.myGridIsOn)

        self.canvas.draw()
        #print 'End of on_draw'

    def plots_in_log10_scale(self) :
        self.arr2d = np.log10(self.arrwin)
        #self.arr2d = self.arrwin

        if self.range_his == None : 
            vmin, vmax = np.min(self.arrwin), np.max(self.arrwin)
        else :
            vmin, vmax = self.range_his

        self.fig.myZmin, self.fig.myZmax = vmin, vmax 

        if vmin<0.1 : vmin=0.1
        if vmax<10  : vmax=10
        log_vmin, log_vmax = int(log10(vmin)), int(log10(vmax))+1
        #print 'vmin, vmax, log_vmin, log_vmax = ', vmin, vmax, log_vmin, log_vmax
                        
        self.axhis = self.fig.add_axes([0.15, 0.75, 0.78, 0.23])
        self.axhis.set_xscale('log')
        #logbins=10**np.linspace(log_vmin, log_vmax, self.nbins)
        logbins=np.logspace(log_vmin, log_vmax, self.nbins)
        #print 'logbins =', logbins        
        #self.axhis.hist(self.arr2d.flatten(), bins=self.nbins, range=self.range_his)
        self.axhis.hist(self.arrwin.flatten(), bins=logbins )
        self.set_hist_yticks()

        self.aximg = self.fig.add_axes([0.10, 0.04, 0.85, 0.65])
        self.aximshow = self.aximg.imshow(self.arr2d, origin='upper', \
                                          interpolation='nearest', \
                                          extent=self.range, aspect='auto')
        self.aximshow.set_clim(log_vmin,log_vmax)

        #self.axcolbar = self.fig.add_axes([0.15, 0.95, 0.78, 0.05])
        #self.colbar = self.fig.colorbar(self.aximshow, cax=self.axcolbar, \
        #                                orientation='horizontal')#, ticks=xticks)
        self.colbar = self.fig.colorbar(self.aximshow, orientation='vertical', \
                                        fraction=0.1, pad=0.01, shrink=1.0, aspect=20)
        # fraction - of the 2d plot occupied by the color bar
        # pad      - is a space between 2d image and color bar
        # shrink   - factor for the length of the color bar
        # aspect   - ratio length/width of the color bar


    def plots_in_linear_scale(self) :
        self.arr2d = self.arrwin
        #print 'self.arr2d = ', self.arr2d

        gs = gridspec.GridSpec(20, 20)
        self.axhis = self.fig.add_subplot(gs[15:19,:]) # self.fig.add_subplot(212)
        self.axhis.hist(self.arrwin.flatten(), bins=self.nbins, range=self.range_his)
        self.set_hist_yticks()

        xticks = self.axhis.get_xticks()
        #print 'xticks =', xticks 
        self.axhis.set_xticklabels('')
        cmin, cmax = self.axhis.get_xlim() 
        self.fig.myZmin, self.fig.myZmax = cmin, cmax 
        #print 'cmin, cmax, self.range_his =', cmin, cmax, self.range_his

        self.aximg = self.fig.add_subplot(gs[0:14,:])
        self.aximshow = self.aximg.imshow(self.arr2d, origin='upper', \
                                          interpolation='nearest', \
                                          extent=self.range, aspect='auto')
        self.aximshow.set_clim(cmin,cmax)
        self.axcolbar = self.fig.add_subplot(gs[19,:])
        self.colbar = self.fig.colorbar(self.aximshow, cax=self.axcolbar, \
                                        orientation=1, ticks=xticks)
                             #fraction=0.15, pad=0.1, shrink=1.0, aspect=15
        #self.colbar.set_clim(zmin,zmax)


    def set_hist_yticks(self) :
        Nmin, Nmax = self.axhis.get_ylim() 
        #print 'Nmin, Nmax =', Nmin, Nmax
        if (Nmax-Nmin)<4 : yticks = np.arange(Nmin,Nmin+4)
        else             : yticks = np.arange(Nmin, Nmax, int((Nmax-Nmin)/4))
        self.axhis.set_yticks( yticks )
        self.axhis.set_ylabel('N pixels')


    def processMouseMotion(self, event) :

        if self.fig.ntbZoomIsOn : return

        if event.inaxes == self.aximg and self.fig.myZoomIsOn :
            #print 'processMouseMotion',
            ##print 'event.xdata, event.ydata =', event.xdata, event.ydata
            #print 'event.x, event.y =', event.x, event.y
            height = self.canvas.figure.bbox.height
            self.xmotion    = event.xdata
            self.ymotion    = event.ydata
            x0 = self.xpressabs
            x1 = event.x
            y0 = height - self.ypressabs
            y1 = height - event.y
            w  = x1 - x0
            h  = y1 - y0
            rect = [x0, y0, w, h]
            self.fig.canvas.drawRectangle( rect )            

    def processMouseButtonPress(self, event) :
        #print 'MouseButtonPress'
        #self.fig = event.canvas.figure

        if event.inaxes == self.colbar.ax : self.mousePressOnColorBar (event)
        if event.inaxes == self.aximg     : self.mousePressOnImage    (event)
        if event.inaxes == self.axhis     : self.mousePressOnHistogram(event)


    def mousePressOnImage(self, event) :
        if event.inaxes == self.aximg :
           #print 'PressOnImage'
           #print 'event.xdata, event.ydata =', event.xdata, event.ydata
            self.xpress    = event.xdata
            self.ypress    = event.ydata
            self.xpressabs = event.x
            self.ypressabs = event.y
            self.fig.myZoomIsOn = True

    def mousePressOnHistogram(self, event) :
        #print 'PressOnHistogram'
        lims = self.axhis.get_xlim()
        self.setColorLimits(event, lims[0], lims[1], event.xdata)


    def mousePressOnColorBar(self, event) :
        #print 'PressOnColorBar'

        if self.fig.myLogIsOn : return

        lims = self.aximshow.get_clim()
        colmin = lims[0]
        colmax = lims[1]
        range = colmax - colmin
        value = colmin + event.xdata * range
        self.setColorLimits(event, colmin, colmax, value)


    def setColorLimits(self, event, colmin, colmax, value) :

        #print colmin, colmax, value

        # left button
        if event.button is 1 :
            if value > colmin and value < colmax :
                colmin = value
                #print "New mininum: ", colmin

        # middle button
        elif event.button is 2 :
            #colmin, colmax = self.getImageAmpLimitsFromWindowParameters()
            #print 'Reset fig' # ,fig.number #, fig.nwin 
            colmin = None
            colmax = None

        # right button
        elif event.button is 3 :
            if value > colmin and value < colmax :
                colmax = value
                #print "New maximum: ", colmax

        self.fig.myZmin = colmin
        self.fig.myZmax = colmax

        self.processDraw()


    def processMouseButtonRelease(self, event) :
        #print 'MouseButtonRelease'

        fig = event.canvas.figure # or plt.gcf()
        #self.fig = fig
        #figNum  = fig.number 
        #axes     = event.inaxes # fig.gca() 
                
        if event.inaxes == self.aximg and event.button == 1 and fig.myZoomIsOn :

            self.xrelease = event.xdata
            self.yrelease = event.ydata
            if   abs(self.xrelease-self.xpress)<2 : pass
            elif abs(self.yrelease-self.ypress)<2 : pass
            else :
                fig.myXmin = int(min(self.xpress, self.xrelease))
                fig.myXmax = int(max(self.xpress, self.xrelease))  
                fig.myYmin = int(min(self.ypress, self.yrelease))
                fig.myYmax = int(max(self.ypress, self.yrelease))

                #print ' Xmin, Xmax, Ymin, Ymax =', fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax
                #self.on_draw(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax, fig.myZmin, fig.myZmax)
                self.processDraw()

        elif event.button == 2 : # middle or right button
            if event.inaxes == self.aximg : 
                fig.myXmin = None
                fig.myXmax = None
                fig.myYmin = None
                fig.myYmax = None
                self.processDraw()
                #self.on_draw()

            elif event.inaxes == self.axhis or event.inaxes == self.colbar.ax :
                fig.myZmin = None
                fig.myZmax = None
                self.processDraw()
                #self.on_draw()

            #plt.draw() # redraw the current figure
        fig.myZoomIsOn = False

        #self.setEditFieldValues()


    def stringOrNone(self,value):
        if value == None : return 'None'
        else             : return str(value)


    def intOrNone(self,value):
        if value == None : return None
        else             : return int(value)


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

    w = ImgSpeWidget(None, get_array2d_for_test())
    #w = ImgSpeWidget(None)
    #w.set_image_array( get_array2d_for_test() )
    w.move(QtCore.QPoint(50,50))
    w.show()    

    app.exec_()
        
#-----------------------------
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    main()

#-----------------------------
