#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module PlotImgSpeWidget...
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
from   matplotlib.ticker import MaxNLocator, NullFormatter
#from   matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from PyQt4 import QtGui, QtCore

#---------------------

def arr_rot_n90(arr, rot_ang_n90=0) :
    if   rot_ang_n90==  0 : return arr
    elif rot_ang_n90== 90 : return np.flipud(arr.T)
    elif rot_ang_n90==180 : return np.flipud(np.fliplr(arr))
    elif rot_ang_n90==270 : return np.fliplr(arr.T)
    else                  : return arr

#---------------------

class PlotImgSpeWidget (QtGui.QWidget) :
    """Plots image and spectrum for 2d numpy array."""

    def __init__(self, parent=None, arr=None, rot_ang_n90=0, y_is_flip=False):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('Matplotlib image embadded in Qt widget')
        self.y_is_flip = y_is_flip
        self.rot_ang_n90 = int(rot_ang_n90)
        self.arr = arr_rot_n90(arr, self.rot_ang_n90)       
        self.fig = plt.figure(figsize=(5,10), dpi=100, facecolor='w', edgecolor='w', frameon=True)
        #self.fig = Figure(    figsize=(5,10), dpi=100, facecolor='w', edgecolor='w', frameon=True)
        #print 'fig.number =', self.fig.number

        #-----------------------------------
        #self.canvas = FigureCanvas(self.fig)
        self.canvas = self.fig.canvas
        self.vbox = QtGui.QVBoxLayout()         # <=== Begin to combine layout 
        self.vbox.addWidget(self.canvas)        # <=== Add figure 
        self.setLayout(self.vbox)
        #-----------------------------------

        self.canvas.mpl_connect('axes_leave_event',     self.processAxesLeaveEvent)
        self.canvas.mpl_connect('axes_enter_event',     self.processAxesEnterEvent)
        self.canvas.mpl_connect('figure_leave_event',   self.processFigureLeaveEvent)

        self.connectZoomMode()
        self.cid_digi_motion  = self.canvas.mpl_connect('motion_notify_event',  self.onMouseMotion)

        self.initParameters()
        self.setFrame()

        self.fig.clear()        

        self.axhi = self.fig.add_axes([0.15, 0.06, 0.78, 0.21])
        self.axim = self.fig.add_axes([0.15, 0.32, 0.78, 0.67])
        self.axcb = self.fig.add_axes([0.15, 0.03, 0.78, 0.028])
  
        if self.arr is not None : self.on_draw()


    def connectZoomMode(self):
        self.cid_press   = self.canvas.mpl_connect('button_press_event',   self.processMouseButtonPress) 
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.processMouseButtonRelease) 
        self.cid_motion  = self.canvas.mpl_connect('motion_notify_event',  self.processMouseMotion)


    def disconnectZoomMode(self):
        self.canvas.mpl_disconnect(self.cid_press)
        self.canvas.mpl_disconnect(self.cid_release)
        self.canvas.mpl_disconnect(self.cid_motion)

        
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
        self.fig.my_xyc      = None

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

    def get_axim(self):
        return self.axim

    def get_imsh(self):
        return self.imsh

    def get_xy_img_center(self):
        xmin,xmax,ymin,ymax = self.imsh.get_extent()
        return abs(ymin-ymax)/2, abs(xmax-xmin)/2  # return in terms of row, column ????

    def get_img_shape(self):
        return self.arr.shape

    def resizeEvent(self, e):
        #print 'resizeEvent' 
        self.frame.setGeometry(self.rect())


    def closeEvent(self, event): # is called for self.close() or when click on "x"
        #print 'PlotImgSpeWidget: closeEvent'
        pass
        plt.close(self.fig.number)


    def set_image_array(self,arr):
        self.arr = arr_rot_n90(arr, self.rot_ang_n90)
        self.processDraw()


    def set_image_array_new(self, arr, rot_ang_n90=0, y_is_flip=False):
        self.y_is_flip = y_is_flip
        self.rot_ang_n90 = int(rot_ang_n90)
        self.arr = arr_rot_n90(arr, self.rot_ang_n90)
        self.on_draw()


    def processDraw(self) :
        #fig = event.canvas.figure
        f = self.fig
        self.on_draw(f.myXmin, f.myXmax, f.myYmin, f.myYmax, f.myZmin, f.myZmax, f.myNBins)


    def on_draw(self, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, nbins=100):
        """Redraws the figure"""

        rows,cols = self.arr.shape
        if xmin is None or xmax is None or ymin is None or ymax is None :
            self.arrwin  = self.arr
            if self.y_is_flip : self.range = [0,cols,0,rows] # original image range in pixels
            else              : self.range = None # original image range in pixels
        else :
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)

            #print 'xmin, xmax, ymin, ymax =', xmin, xmax, ymin, ymax

            if self.y_is_flip :
                self.range = [xmin, xmax, ymin, ymax]
                self.arrwin =  self.arr[rows-ymax:rows-ymin,xmin:xmax]
                #self.arrwin =  self.arr[ymin:ymax,xmin:xmax]
            else :
                self.range = [xmin, xmax, ymax, ymin]
                self.arrwin =  self.arr[ymin:ymax,xmin:xmax]

        zmin = self.floatOrNone(zmin)
        zmax = self.floatOrNone(zmax)

        if zmin is None and zmax is None : self.range_his = None
        else                         : self.range_his = (zmin,zmax)

        #print 'self.range_his = ', self.range_his
        #print 'self.arrwin = ', self.arrwin

        #self.fig.clear() # <<<<==== ????????
        self.nbins = nbins

        self.axhi.clear()
        #self.axim.clear()
        #self.axcb.clear()

        #if self.fig.myLogIsOn : self.plots_in_log10_scale_for_img_and_xhist()
        if self.fig.myLogIsOn : self.plots_in_log10_scale_for_img_and_yhist()
        else :                  self.plots_in_linear_scale()

        self.axhi.grid(self.fig.myGridIsOn)
        self.axim.grid(self.fig.myGridIsOn)

        self.canvas.draw()
        #print 'End of on_draw'


    def plots_in_log10_scale_for_img_and_yhist(self) :
        self.arr2d = np.log10(self.arrwin)
        #self.arr2d = self.arrwin

        if self.range_his is None : 
            vmin, vmax = np.min(self.arrwin), np.max(self.arrwin)
        else :
            vmin, vmax = self.range_his

        self.fig.myZmin, self.fig.myZmax = vmin, vmax 

        if vmin<0.1 : vmin=0.1
        if vmax<10  : vmax=10
        log_vmin, log_vmax = log10(vmin), log10(vmax)
        if log_vmax == log_vmin : log_vmax = log_vmin + 0.1


        #print 'vmin, vmax, log_vmin, log_vmax = ', vmin, vmax, log_vmin, log_vmax
                        
        #self.axhi = self.fig.add_axes([0.15, 0.75, 0.78, 0.23])
        #self.axim = self.fig.add_axes([0.10, 0.04, 0.85, 0.65])

        self.axhi.xaxis.set_major_locator(MaxNLocator(5))
        self.axhi.yaxis.set_major_locator(MaxNLocator(4))
        self.axhi.xaxis.set_major_formatter(NullFormatter())

        self.axhi.hist(self.arrwin.flatten(), bins=self.nbins, range=self.range_his, log=True)

        self.imsh = self.axim.imshow(self.arr2d, origin='upper', \
                                          interpolation='nearest', \
                                          extent=self.range, aspect='auto')
        self.imsh.set_clim(log_vmin,log_vmax)

#        self.colb = self.fig.colorbar(self.imsh, orientation='vertical')#, cax=self.axim, \
#                                      fraction=0.1, pad=0.01, shrink=1.0, aspect=20) #cax=self.axim, 

        #self.axcb = self.fig.add_axes([0.15, 0.95, 0.78, 0.05])
        #self.colb = self.fig.colorbar(self.imsh, cax=self.axcb, \
        #                                orientation='horizontal')#, ticks=xticks)

        #self.axim = self.fig.add_axes([0.15, 0.32, 0.78, 0.67])
        #self.axcb = self.fig.add_axes([0.90, 0.32, 0.10, 0.67])
        #self.colb = self.fig.colorbar(self.imsh, orientation='horizontal') #, \
#                                      fraction=0.1, pad=0.01, shrink=1.0, aspect=20)

        # fraction - of the 2d plot occupied by the color bar
        # pad      - is a space between 2d image and color bar
        # shrink   - factor for the length of the color bar
        # aspect   - ratio length/width of the color bar

        try    : del self.colb
        except : pass
        self.axcb = self.fig.add_axes([0.15, 0.03, 0.78, 0.028])
        self.colb = self.fig.colorbar(self.imsh, cax=self.axcb, \
                                      orientation='horizontal')#, ticks=xticks)


    def plots_in_log10_scale_for_img_and_xhist(self) :
        self.arr2d = np.log10(self.arrwin)
        #self.arr2d = self.arrwin

        if self.range_his is None : 
            vmin, vmax = np.min(self.arrwin), np.max(self.arrwin)
        else :
            vmin, vmax = self.range_his

        self.fig.myZmin, self.fig.myZmax = vmin, vmax 

        if vmin<0.1 : vmin=0.1
        if vmax<10  : vmax=10
        log_vmin, log_vmax = int(log10(vmin)), int(log10(vmax))+1
        if log_vmax == log_vmin : log_vmax = log_vmin + 1
        #print 'vmin, vmax, log_vmin, log_vmax = ', vmin, vmax, log_vmin, log_vmax
                        
        self.axhi = self.fig.add_axes([0.15, 0.75, 0.78, 0.23])
        self.axim = self.fig.add_axes([0.10, 0.04, 0.85, 0.65])

        self.axhi.xaxis.set_major_locator(MaxNLocator(5))
        self.axhi.yaxis.set_major_locator(MaxNLocator(4))

        self.axhi.set_xscale('log')
        #logbins=10**np.linspace(log_vmin, log_vmax, self.nbins)
        logbins=np.logspace(log_vmin, log_vmax, self.nbins)
        #print 'logbins =', logbins        
        #self.axhi.hist(self.arr2d.flatten(), bins=self.nbins, range=self.range_his)

        self.axhi.hist(self.arrwin.flatten(), bins=logbins )
        self.set_hist_yticks()

        self.imsh = self.axim.imshow(self.arr2d, origin='upper', \
                                          interpolation='nearest', \
                                          extent=self.range, aspect='auto')
        self.imsh.set_clim(log_vmin,log_vmax)

        #self.axcb = self.fig.add_axes([0.15, 0.95, 0.78, 0.05])
        #self.colb = self.fig.colorbar(self.imsh, cax=self.axcb, \
        #                                orientation='horizontal')#, ticks=xticks)
        self.colb = self.fig.colorbar(self.imsh, orientation='vertical', \
                                        fraction=0.1, pad=0.01, shrink=1.0, aspect=20)
        # fraction - of the 2d plot occupied by the color bar
        # pad      - is a space between 2d image and color bar
        # shrink   - factor for the length of the color bar
        # aspect   - ratio length/width of the color bar


    def plots_in_linear_scale(self) :
        self.arr2d = self.arrwin
        #print 'self.arr2d = ', self.arr2d

        #self.axhi = self.fig.add_axes([0.15, 0.06, 0.78, 0.21])
        #self.axim = self.fig.add_axes([0.15, 0.32, 0.78, 0.67])
        #self.axcb = self.fig.add_axes([0.15, 0.03, 0.78, 0.028])

        self.axhi.hist(self.arrwin.flatten(), bins=self.nbins, range=self.range_his)
        self.set_hist_yticks()
        self.axhi.xaxis.set_major_formatter(NullFormatter())

        xticks = self.axhi.get_xticks()
        #print 'xticks =', xticks 
        self.axhi.set_xticklabels('')
        cmin, cmax = self.axhi.get_xlim() 
        self.fig.myZmin, self.fig.myZmax = cmin, cmax 
        #print 'cmin, cmax, self.range_his =', cmin, cmax, self.range_his

        self.imsh = self.axim.imshow(self.arr2d, origin='upper', \
                                          interpolation='nearest', \
                                          extent=self.range, aspect='auto')
        self.imsh.set_clim(cmin,cmax)
        try    : del self.colb
        except : pass
        self.axcb = self.fig.add_axes([0.15, 0.03, 0.78, 0.028])
        self.colb = self.fig.colorbar(self.imsh, cax=self.axcb, \
                                      orientation='horizontal', ticks=xticks)
                             #fraction=0.15, pad=0.1, shrink=1.0, aspect=15
        #self.colb.set_clim(zmin,zmax)


    def set_hist_yticks(self) :
        Nmin, Nmax = self.axhi.get_ylim() 
        #print 'Nmin, Nmax =', Nmin, Nmax
        if (Nmax-Nmin)<4 : yticks = np.arange(Nmin,Nmin+4)
        else             : yticks = np.arange(Nmin, Nmax, int((Nmax-Nmin)/4))
        self.axhi.set_yticks( yticks )
        self.axhi.set_ylabel('N pixels')


    def drawXCoordinateOfCoursor(self, event) :
        axes = event.inaxes
        #xmin, xmax = axes.get_xlim()
        #ymin, ymax = axes.get_ylim()
        x, y = event.xdata, event.ydata
        s = '%6.1f' % (event.xdata)
        try : self.curstext.remove()
        except : pass
        self.curstext = axes.text(x, y, s, fontsize=10) #, ha='center')
        self.canvas.draw()


    def drawXYCoordinateOfCoursor(self, event) :
        axes = event.inaxes
        #xmin, xmax = axes.get_xlim()
        #ymin, ymax = axes.get_ylim()
        x, y = event.xdata, event.ydata
        s = '%d, %d' % (event.xdata, event.ydata)
        try : self.curstext.remove()
        except : pass
        self.curstext = axes.text(x, y, s, fontsize=10) #, ha='center')
        self.canvas.draw()


    def drawVerticalLineThroughCoursor(self, event) :
        axes   = event.inaxes
        fb     = self.canvas.figure.bbox
        bb     = axes.bbox
        #print bb

        bbx0, bby0, bbh, bbw = bb.x0, bb.y0, bb.height, bb.width
        fbx0, fby0, fbh, fbw = fb.x0, fb.y0, fb.height, fb.width

        #xd = event.xdata
        #yd = event.ydata
        x = event.x
        y = event.y

        h  = bbh
        x0 = bbx0 
        y0 = fbh - bby0 - h
        w  = x - x0

        rect = [x0, y0, w, h]
        self.line_ver = self.fig.canvas.drawRectangle( rect )            
        #self.fig.canvas.draw()


    def drawHorizontalLineThroughCoursor(self, event) :
        axes   = event.inaxes
        fb     = self.canvas.figure.bbox
        bb     = axes.bbox
        #print bb

        bbx0, bby0, bbh, bbw = bb.x0, bb.y0, bb.height, bb.width
        fbx0, fby0, fbh, fbw = fb.x0, fb.y0, fb.height, fb.width

        #xd = event.xdata
        #yd = event.ydata
        x = event.x
        y = event.y

        x0 = bbx0
        y0 = fbh - bby0 - bbh
        h  = bbh - (y - bby0)

        rect = [x0, y0, bbw+1, h]
        self.line_hor = self.fig.canvas.drawRectangle( rect )            
        #self.fig.canvas.draw()


    def processAxesEnterEvent(self, event) :
        #print 'AxesEnterEvent'
        if event.inaxes == self.axhi :
            #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
            QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.SizeHorCursor))
            #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.SizeAllCursor))

        elif event.inaxes == self.axim :
            #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.SizeAllCursor))
            QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))



    def processAxesLeaveEvent(self, event) :
        #print 'AxesLeaveEvent'
        try : self.curstext.remove()
        except : pass
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))


    def processFigureLeaveEvent(self, event) :
        #print 'FigureLeaveEvent'
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))


    def onMouseMotion(self, event) :

        if event.inaxes == self.axhi :
            self.drawXCoordinateOfCoursor(event)
            self.drawVerticalLineThroughCoursor(event)
            #self.drawHorizontalLineThroughCoursor(event)

        if event.inaxes == self.axim :
            self.drawXYCoordinateOfCoursor(event)


    def processMouseMotion(self, event) :

        if self.fig.ntbZoomIsOn : return

        #if event.inaxes == self.axhi :
        #    self.drawXCoordinateOfCoursor(event)
        #    self.drawVerticalLineThroughCoursor(event)
        #    #self.drawHorizontalLineThroughCoursor(event)

        #if event.inaxes == self.axim :
        #    self.drawXYCoordinateOfCoursor(event)

        if event.inaxes == self.axim and self.fig.myZoomIsOn :
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

        if event.inaxes == self.colb.ax : self.mousePressOnColorBar (event)
        if event.inaxes == self.axim    : self.mousePressOnImage    (event)
        if event.inaxes == self.axhi    : self.mousePressOnHistogram(event)


    def mousePressOnImage(self, event) :
        if event.inaxes == self.axim :
           #print 'PressOnImage'
           #print 'event.xdata, event.ydata =', event.xdata, event.ydata
            self.xpress    = event.xdata
            self.ypress    = event.ydata
            self.xpressabs = event.x
            self.ypressabs = event.y
            self.fig.myZoomIsOn = True
            QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.SizeAllCursor))

    def mousePressOnHistogram(self, event) :
        #print 'PressOnHistogram'
        lims = self.axhi.get_xlim()
        self.setColorLimits(event, lims[0], lims[1], event.xdata)


    def mousePressOnColorBar(self, event) :
        #print 'PressOnColorBar'

        if self.fig.myLogIsOn : return

        lims = self.imsh.get_clim()
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
                
        if event.inaxes == self.axim and event.button == 1 and fig.myZoomIsOn :

            self.xrelease = event.xdata
            self.yrelease = event.ydata
            if   abs(self.xrelease-self.xpress)<2 : pass
            elif abs(self.yrelease-self.ypress)<2 : pass
            else :
                fig.myXmin = int(min(self.xpress, self.xrelease))
                fig.myXmax = int(max(self.xpress, self.xrelease))  
                fig.myYmin = int(min(self.ypress, self.yrelease))
                fig.myYmax = int(max(self.ypress, self.yrelease))

                QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.SizeAllCursor))
                #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.BusyCursor))
                #print ' Xmin, Xmax, Ymin, Ymax =', fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax
                #self.on_draw(fig.myXmin, fig.myXmax, fig.myYmin, fig.myYmax, fig.myZmin, fig.myZmax)
                self.processDraw()

        elif event.button == 2 : # middle or right button
            if event.inaxes == self.axim : 
                fig.myXmin = None
                fig.myXmax = None
                fig.myYmin = None
                fig.myYmax = None
                self.processDraw()
                #self.on_draw()

            elif event.inaxes == self.axhi or event.inaxes == self.colb.ax :
                fig.myZmin = None
                fig.myZmax = None
                self.processDraw()
                #self.on_draw()

            #plt.draw() # redraw the current figure
        fig.myZoomIsOn = False

        #self.setEditFieldValues()


    def stringOrNone(self,value):
        if value is None : return 'None'
        else             : return str(value)


    def floatOrNone(self,value):
        if value is None : return None
        else             : return float(value) # return int(value)


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

    w = PlotImgSpeWidget(None, get_array2d_for_test())
    #w = PlotImgSpeWidget(None)
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
