import time
import numpy as np
import matplotlib.pyplot as plt

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



class Plotter(object):
    def __init__(self):
        self.plot_vmin = None
        self.plot_vmax = None
        self.cid1 = None
        self.cid2 = None
        self.colb = None
        self.grid = None
        
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

        
    def addtofigure( self, frameimage, title="", fignum=1, subplot=(1,1,1)):

        r,c,p = subplot
        w = c*6.0
        h = r*5.0

        self.fig = plt.figure(figsize=(w,h),num=fignum)
        self.connect()

        self.axes = self.fig.add_subplot(r,c,p)

        self.fig.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.95)
        self.fig.subplots_adjust(wspace=0.1, hspace=0.1)

        self.axes.set_title(title)
        self.axesim = plt.imshow( frameimage,
                                  vmin = self.plot_vmin,
                                  vmax = self.plot_vmax )
        self.colb = plt.colorbar(self.axesim,ax=self.axes,pad=0.04,fraction=0.10,shrink=0.90)

    def addcolbar(self):
        self.colb = plt.colorbar(self.axesim,ax=self.axes,pad=0.01,fraction=0.10,shrink=0.90)

    def drawframe( self, frameimage, title="", fignum=1):

        self.fig = plt.figure(figsize=(10,8),num=fignum)
        self.fig.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
        axes = self.fig.add_subplot(111)        
        axes.set_title(title)
        self.axesim = plt.imshow( frameimage,
                                  vmin=self.plot_vmin, vmax=self.plot_vmax )

        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect('pick_event', self.onpick)
    
    
        print "vmin vmax now: ", self.plot_vmin, self.plot_vmax
        if self.colb is None: 
            self.colb = plt.colorbar(self.axesim,ax=axes,pad=0.01,fraction=0.10,shrink=0.90)
        # colb is the colorbar object
    
    
        #        if self.plot_vmin is None: 
        #            self.orglims = self.axesim.get_clim()
        #            # min and max values in the axes are
        #            print "Original value limits: ", self.orglims
        #            self.plot_vmin, self.plot_vmax = self.orglims
        
        print """
        To change the color scale, click on the color bar:
        - left-click sets the lower limit
        - right-click sets higher limit
        - middle-click resets to original
        """
    
        ## show the active region for thresholding
        #        if self.thr_area is not None:
        #            xy = [self.thr_area[0],self.thr_area[2]]
        #            w = self.thr_area[1] - self.thr_area[0]
        #            h = self.thr_area[3] - self.thr_area[2]
        #            self.thr_rect = plt.Rectangle(xy,w,h, facecolor='none', edgecolor='red', picker=5)
        #            myplot.axes.add_patch(self.thr_rect)
        
        
        #cspad_image[self.thr_area[0]:self.thr_area[1], self.thr_area[2]:self.thr_area[3]])



        #plt.show() # starts the GUI main loop
        #           # you need to kill window to proceed... 
        #           # (this shouldn't be done for every event!)


    def onpick(self, event):
        print "Currently active area for threshold evaluation: [xmin xmax ymin ymax] = ", self.thr_area
        print "To change this area, right-click..." 
        print "To change threshold, middle-click..." 
        if event.mouseevent.button == 3 :
            print "Enter new coordinates to change this area:"
            xxyy_string = raw_input("xmin xmax ymin ymax = ")
            xxyy_list = xxyy_string.split(" ")
            
            if len( xxyy_list ) != 4 :
                print "Invalid entry, ignoring"
                return
        
            for i in range (4):
                self.thr_area[i] = float( xxyy_list[i] )
            
            x = self.thr_area[0]
            y = self.thr_area[2]
            w = self.thr_area[1] - self.thr_area[0]
            h = self.thr_area[3] - self.thr_area[2]
            
            self.thr_rect.set_bounds(x,y,w,h)
            #plt.draw()
            
        if event.mouseevent.button == 2 :
            text = raw_input("Enter new threshold value (current = %.2f) " % self.threshold)
            if text == "" :
                print "Invalid entry, ignoring"
                self.threshold = float(text)
                print "Threshold value has been changed to ", self.threshold
            
    # define what to do if we click on the plot
    def onclick(self, event) :
    
        # can we open a dialogue box here?
        if not event.inaxes and event.button == 3 :
            print "can we open a menu here?"
            
        # check that the click was on the color bar
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




    
