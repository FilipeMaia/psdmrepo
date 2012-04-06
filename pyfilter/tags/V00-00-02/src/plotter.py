#--------------------------------------------------------------------------
# File and Version Information:
# pyfilter.plotter
#------------------------------------------------------------------------
import logging
import time, math

import matplotlib.pylab as plt
import numpy as np

class plotter (object) :
    """Class whose instance will be used as a user analysis module. """

    def __init__ ( self,
                   source   = "CxiDs1-0|Cspad-0",
                   image_name   = "CxiDs1-0|Cspad-0",
                   plot_quantities = "image spectrum",
                   figsize = "10,8",
                   n_std_range = "1,3") :
        """
        @param source    Address of detector/device in xtc file.
        """
        self.source = source
        self.imname = image_name
        self.plot_quantities = plot_quantities.split(' ')
        self.nStd = map(float, n_std_range.split(','))
        self.figsize = tuple( map(int,figsize.split(',')))
        self.counter = 0
        plt.ion()

        self._make_plot = { "image"  : self.plot_image,
                            "spectrum" : self.plot_spectrum }
        
    # ---------------------------
    # pyana functions
    # ---------------------------

    def beginjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "pyfilter.plotter.beginjob() called" )

        self.figures = {}
        for plot_item in self.plot_quantities:
            fig = plt.figure(figsize=self.figsize)
            fig.add_subplot(111,
                            title=plot_item)
            self.figures[plot_item] = fig 
            fig.suptitle(self.source)
        plt.draw()

    def beginrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "pyfilter.plotter.beginrun() called" )

        # get a reference to the CSPad object, if it exists
        self.cspad = evt.get("CsPadAssembler:%s"%self.source)
                

    def begincalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "pyfilter.plotter.begincalibcycle() called" )

                
    def event( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        self.counter += 1

        # Get the image from the previous module
        image = evt.get(self.imname)
        if image is None :
            # draw the cspad image instead 
            image = self.cspad.assemble_image()
                
                
        shot_number = evt.get("shot_number")
        
        s =  evt.getTime().seconds()
        ns = evt.getTime().nanoseconds() 

        lt = time.localtime(s)
        self.title = "Shot#%d: %s and %d ns"%(shot_number,time.strftime("%a %d %b %Y %H:%M:%S",lt),ns)

        vrange = (None, None)
        try:
            activepix = image[ np.nonzero(image)]
            mu,std = activepix.mean(),activepix.std()
            vrange = (mu-self.nStd[0]*std), (mu+self.nStd[1]*std)
            #print "Pyfilter.Plotter vrange: ", vrange
        except:
            pass
        
        for plot_item in self.plot_quantities: 
            self._make_plot[plot_item]( plot_item, image, vrange )
            
        plt.draw()
                
    def endcalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "pyfilter.plotter.endcalibcycle() called" )

    def endrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "pyfilter.plotter.endrun() called" )

    def endjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "pyfilter.plotter.endjob() called" )

        plt.ioff()
        plt.show()

    # ---------------------------
    # other functions
    # ---------------------------

    def plot_image(self, name, image, vrange=(None,None)):
        #print "Plot image", image.shape
        ax = self.figures[name].gca()
        axims = ax.get_images()
        if len(axims)==0:
            #print "..............1"
            ax.clear()
            axim = ax.imshow(image,interpolation='nearest',vmin=vrange[0],vmax=vrange[1])
            #axim = ax.imshow(image,interpolation='nearest')
            cax = ax.figure.colorbar(axim,pad=0.02)
        else :
            #print "..............2"
            axim = axims[0]
            axim.set_data( image )
            if vrange[0] is not None:
                axim.set_clim(vrange[0],vrange[1])
            else:
                axim.set_clim(np.min(image),np.max(image))

        ax.set_title(self.title)
        ax.figure.canvas.draw()

    def plot_spectrum(self, name, image, vrange=(None,None)):
        #print "Plot spectrum", image.shape
        spectrum = None
        if vrange[0] is not None:
            mask = np.logical_and(image>vrange[0],image<vrange[1])
            mask = np.logical_and(mask,image!=0)
            spectrum = image[mask]
        else:
            spectrum = image[np.nonzero(image)]

        ax = self.figures[name].gca()
        lines = ax.get_lines()
        if len(lines)==0:
            #print "nytt plott"
            ax.clear()
            n,bins,patches= ax.hist( spectrum, 1000, histtype='stepfilled')
        else:
            #print "gammelt plott"            
            line = lines[0]
            line.set_ydata(spectrum) 
        ax.set_title(self.title)
        ax.figure.canvas.draw()
