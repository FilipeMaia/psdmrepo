import time
import numpy as np
import matplotlib.pyplot as plt
from PyQt4 import QtCore

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid

def draw_on(fignr):
    fig = plt.figure(num=200)
    axes = fig.add_subplot(111)
    axes.set_title("Hello MatPlotLib")
    
    plt.show()
    
    dark_image = np.load("pyana_cspad_average_image.npy")
    axim = plt.imshow( dark_image )#, origin='lower' )
    colb = plt.colorbar(axim,pad=0.01)
    
    plt.draw()
    
    print "Done drawing"
    
    axim = plt.imshow( dark_image[500:1000,1000:1500] )#, origin='lower' )
    
    return fig

def draw_on_simple(fignr):
    fig = plt.figure(num=200)
    axes = fig.add_subplot(111)
    axes.set_title("Hello MatPlotLib")
    
    plt.show()
    
    dark_image = np.load("pyana_cspad_average_image.npy")
    axim = plt.imshow( dark_image )#, origin='lower' )
    colb = plt.colorbar(axim,pad=0.01)
    
    plt.draw()
    
    print "Done drawing"
    
    axim = plt.imshow( dark_image[500:1000,1000:1500] )#, origin='lower' )
    
    return fig


class Threshold( object ) :
    def __init__( self,
                  area = None,
                  minvalue = None,
                  maxvalue = None,
                  ) :

        self.area = area        
        self.minvalue = minvalue
        self.maxvalue = maxvalue


class Plotter(object):
    def __init__(self):
        self.plot_vmin = None
        self.plot_vmax = None
        self.cid1 = None
        self.cid2 = None
        self.colb = None
        self.grid = None
        self.threshold = None
        self.shot_number = None
        self.display_mode = None

        # matplotlib backend is set to QtAgg, and this is needed to avoid
        # a bug in raw_input ("QCoreApplication::exec: The event loop is already running")
        QtCore.pyqtRemoveInputHook()

        
    def connect(self):

        if self.cid1 is None:
            self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        if self.cid2 is None:
            self.cid2 = self.fig.canvas.mpl_connect('pick_event', self.onpick)

    def suptitle(self,fignum,title):
        if self.fig is None:
            print "suptitle must be called after the figure has been created"
            return
        self.fig.suptitle( title )

    def gridplot(self, frameimage, title="", fignum=1, subplot=(1,1,1)):
        
        r,c,p = subplot
        w = c*6.0
        h = r*5.0

        self.fig = plt.figure(figsize=(w,h),num=fignum)
        self.connect()

        if self.grid is None: 
            self.grid = AxesGrid( self.fig, 111,
                                  nrows_ncols = (r,c),
                                  axes_pad = 0.5,
                                  label_mode = 1,
                                  share_all = False,
                                  cbar_mode = "each",
                                  cbar_location = "right",
                                  cbar_size = "7%",
                                  cbar_pad = "2%",
                                  )
        self.axesim = self.grid[p].imshow( frameimage, 
                                      vmin = self.plot_vmin,
                                      vmax = self.plot_vmax )
        self.grid.cbar_axes[p].colorbar(self.axesim)
        plt.draw()
        
    def addtofigure( self, frameimage, title="", fignum=1, subplot=(1,1,1)):

        r,c,p = subplot
        w = c*6.0
        h = r*5.0

        self.fig = plt.figure(figsize=(w,h),num=fignum)
        self.fig.clf()
        self.connect()

        self.axes = self.fig.add_subplot(r,c,p)

        self.fig.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.95)
        self.fig.subplots_adjust(wspace=0.1, hspace=0.1)

        self.axes.set_title(title)
        self.axesim = plt.imshow( frameimage,
                                  vmin = self.plot_vmin,
                                  vmax = self.plot_vmax )
        self.colb = plt.colorbar(self.axesim,ax=self.axes,pad=0.04,fraction=0.10,shrink=0.90)
        plt.draw()

    def addcolbar(self):
        self.colb = plt.colorbar(self.axesim,ax=self.axes,pad=0.01,fraction=0.10,shrink=0.90)

    def draw_figurelist(self, fignum, event_display_images ) :
        """
        @fignum                  figure number, i.e. fig = plt.figure(num=fignum)
        @event_display_images    a list of tuples (title,image)
        """
        axspos = 0

        nplots = len(event_display_images)
        ncol = 3
        if nplots<3 : ncol = nplots
        nrow = int( nplots/ncol)
        fig = plt.figure(fignum,(5.0*ncol,4*nrow))
        fig.clf()
        fig.suptitle("Event#%d"%self.shot_number)


        pos = 0
        self.caxes = [] # list of references to colorbar Axes
        self.axims = [] # list of references to image Axes
        for ad, im in sorted(event_display_images) :
            pos += 1
            
            # Axes
            ax = fig.add_subplot(nrow,ncol,pos)
            ax.set_title( "%s" % ad )

            # AxesImage
            axim = plt.imshow( im, origin='lower' )
            self.axims.append( axim )
        
            cbar = plt.colorbar(axim,pad=0.02,shrink=0.78) 
            self.caxes.append( cbar.ax )
            
            self.orglims = axim.get_clim()
            # min and max values in the axes are


            #grid[pos-1].set_title("%s"%(ad))
            #axesim = grid[pos-1].imshow( im )
            #grid.cbar_axes[pos-1].colorbar(axesim)


            #self.plotter.gridplot( image, title, fignum, (2,ncols,axspos) )
        #for cax in self.caxes :
        #    print cax
        #for axim in self.axims :
        #    print axim
        plt.draw()

    def drawframe( self, frameimage, title="", fignum=1):

        if self.display_mode == 2 :
            plt.ion()

        self.fig = plt.figure(figsize=(10,8),num=fignum)
        self.fig.clf()

        axes = self.fig.add_subplot(111)

        self.fig.subplots_adjust(left=0.10,
                                 bottom=0.05,
                                 right=0.90,
                                 top=0.90,
                                 wspace=0.1,
                                 hspace=0.1)

        self.axesim = plt.imshow( frameimage,
                                  vmin=self.plot_vmin, vmax=self.plot_vmax )

        #self.colb = plt.colorbar(self.axesim,ax=axes,pad=0.01)#,fraction=0.10,shrink=0.90)
        divider = make_axes_locatable(axes)

        x = np.sum(frameimage,0)
        y = np.sum(frameimage,1)

        #axes.set_aspect(1.0)
        axHistx = divider.append_axes("top", size="20%", pad=0.05,sharex=axes)
        axHisty = divider.append_axes("left", size="20%", pad=0.05,sharey=axes)

        # vertical and horizontal dimensions, axes, projections
        vdim,hdim = np.shape(frameimage)
        vbins = np.arange(0,vdim,1)
        hbins = np.arange(0,hdim,1)

        proj_vert = np.sum(frameimage,1) # sum along horizontal axis
        proj_horiz = np.sum(frameimage,0) # sum along vertical axis

        # these are the limits I want my histogram to use
        vmin = np.min(proj_vert) - 0.1 * np.min(proj_vert)
        vmax = np.max(proj_vert) + 0.1 * np.max(proj_vert)
        hmin = np.min(proj_horiz) - 0.1 * np.min(proj_horiz)
        hmax = np.max(proj_horiz) + 0.1 * np.max(proj_horiz) 

        axHistx.plot(hbins,proj_horiz)
        axHisty.plot(proj_vert,vbins)
        #axHistx.hist(hbins, bins=hdim, histtype='step', weights=proj_horiz)
        #axHisty.hist(vbins, bins=vdim, histtype='step', weights=proj_vert,orientation='horizontal')

        axHistx.set_xlim(0,hdim)
        axHistx.set_ylim(hmin, hmax )
        ticks = [ 10000 * np.around(hmin/10000) ,
                  10000 * np.around((hmax-hmin)/20000),
                  10000 * np.around(hmax/10000) ]
        axHistx.set_yticks( ticks )

        axHisty.set_ylim(0,vdim)
        axHisty.set_xlim(vmin, vmax )
        ticks = [ 10000 * np.around( vmin/10000) ,
                  10000 * np.around((vmax-vmin)/20000),
                  10000 * np.around(vmax/10000) ]
        axHisty.set_xticks( ticks )

        cax = divider.append_axes("right",size="5%", pad=0.05)
        self.colb = plt.colorbar(self.axesim,cax=cax)
        # colb is the colorbar object

        if self.plot_vmin is None: 
            self.orglims = self.axesim.get_clim()
            # min and max values in the axes are
            print "Original value limits: ", self.orglims
            self.plot_vmin, self.plot_vmax = self.orglims
                    
        # show the active region for thresholding
        if self.threshold and self.threshold.area is not None:
            xy = [self.threshold.area[0],self.threshold.area[2]]
            w = self.threshold.area[1] - self.threshold.area[0]
            h = self.threshold.area[3] - self.threshold.area[2]
            self.thr_rect = plt.Rectangle(xy,w,h, facecolor='none', edgecolor='red', picker=5)
            axes.add_patch(self.thr_rect)
            
        if self.display_mode == 1 :

            self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.cid2 = self.fig.canvas.mpl_connect('pick_event', self.onpick)

            print """
            To change the color scale, click on the color bar:
            - left-click sets the lower limit
            - right-click sets higher limit
            - middle-click resets to original
            """

        axes.set_title(title)
        plt.draw()
        
    def onpick(self, event):
        print "Current   threshold = ", self.threshold.minvalue
        print "          active area [xmin xmax ymin ymax] = ", self.threshold.area
        print "To change threshold value, middle-click..." 
        print "To change active area, right-click..." 

        if event.mouseevent.button == 3 :
            print "Enter new coordinates to change this area:"
            xxyy_string = raw_input("xmin xmax ymin ymax = ")
            xxyy_list = xxyy_string.split(" ")
            
            if len( xxyy_list ) != 4 :
                print "Invalid entry, ignoring"
                return
        
            for i in range (4):
                self.threshold.area[i] = float( xxyy_list[i] )
            
            x = self.threshold.area[0]
            y = self.threshold.area[2]
            w = self.threshold.area[1] - self.threshold.area[0]
            h = self.threshold.area[3] - self.threshold.area[2]
            
            self.thr_rect.set_bounds(x,y,w,h)
            plt.draw()
            
        if event.mouseevent.button == 2 :
            text = raw_input("Enter new threshold value (current = %.2f) " % self.threshold.minvalue)
            if text == "" :
                print "Invalid entry, ignoring"
            else :
                self.threshold.minvalue = float(text)
                print "Threshold value has been changed to ", self.threshold.minvalue
            plt.draw()

            
    # define what to do if we click on the plot
    def onclick(self, event) :
    
        # can we open a dialogue box here?
        if not event.inaxes and event.button == 3 :
            print "can we open a menu here?"
            

        # change display mode
        if not event.inaxes and event.button == 2 :
            new_mode = None
            new_mode_str = raw_input("Switch display mode? Enter new mode: ")
            if new_mode_str != "":
                if new_mode_str == "NoDisplay"   or new_mode_str == "0" : new_mode = 0
                if new_mode_str == "Interactive" or new_mode_str == "1" : new_mode = 1
                if new_mode_str == "SlideShow"   or new_mode_str == "2" : new_mode = 2
            if self.display_mode != new_mode:
                print "Display mode has been changed from (%d) to (%d)" , (self.display_mode,new_mode)
                self.display_mode = new_mode
                

        # change color scale
        if self.colb is not None and event.inaxes == self.colb.ax :

            print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y
            print ' xdata=',event.xdata,' ydata=', event.ydata
        
            lims = self.axesim.get_clim()
        
            self.plot_vmin = lims[0]
            self.plot_vmax = lims[1]
            range = self.plot_vmax - self.plot_vmin
            value = self.plot_vmin + event.ydata * range
            #print self.plot_vmin, self.plot_vmax, range, value
        
            # left button
            if event.button is 1 :
                self.plot_vmin = value
                print "mininum changed:   ( %.2f , %.2f ) " % (self.plot_vmin, self.plot_vmax )
                
                # middle button
            elif event.button is 2 :
                self.plot_vmin, self.plot_vmax = self.orglims
                print "reset"
                    
                # right button
            elif event.button is 3 :
                self.plot_vmax = value
                print "maximum changed:   ( %.2f , %.2f ) " % (self.plot_vmin, self.plot_vmax )
                
            
        
            plt.clim(self.plot_vmin,self.plot_vmax)
            plt.draw() # redraw the current figure




    
