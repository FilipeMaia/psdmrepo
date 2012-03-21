#--------------------------------------------------------------------------
# File and Version Information:
# pyfilter.plotter
#------------------------------------------------------------------------
import logging

import matplotlib.pylab as plt

class plotter (object) :
    """Class whose instance will be used as a user analysis module. """

    def __init__ ( self,
                   image_name   = "CxiDs1-0|Cspad-0",
                   plot_frequency = 1,
                   plot_quantities = "image spectrum") :
        """
        @param source    Address of detector/device in xtc file.
        """
        self.imname = image_name
        self.frequency = int(plot_frequency)
        self.plot_quantities = plot_quantities.split(' ')
        self.counter = 0
        plt.ion()

        self._make_plot = { "image" : self.plot_image,
                            "spectrum" : self.plot_spectrum }
        
    # ---------------------------
    # pyana functions
    # ---------------------------

    def beginjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter3.beginjob() called" )

        self.figures = {}
        for plot_item in self.plot_quantities:
            fig = plt.figure()
            fig.add_subplot(111,
                            title=plot_item)
            self.figures[plot_item] = fig 
        plt.draw()

    def beginrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter3.beginrun() called" )


    def begincalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter3.begincalibcycle() called" )

                
    def event( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        self.counter += 1
        if (self.counter%self.frequency)==0 :
            # Get the image from the previous module
            image = evt.get(self.imname)

            for plot_item in self.plot_quantities: 
                self._make_plot[plot_item]( plot_item, image )
           
            
    def endcalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_filter3.endcalibcycle() called" )

    def endrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_filter3.endrun() called" )

    def endjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_filter3.endjob() called" )


    # ---------------------------
    # other functions
    # ---------------------------

    def plot_image(self, name, image):
        ax = self.figures[name].gca()
        axims = ax.get_images()
        if len(axims)==0:
            ax.clear()
            axim = ax.imshow(image)
        else :
            axim = axims[0]
            axim.set_data( image )
        ax.set_title("%s shot # %d"%(self.imname,self.counter))
        ax.figure.canvas.draw()

    def plot_spectrum(self, name, image):
        ax = self.figures[name].gca()
        lines = ax.get_lines()
        if len(lines)==0:
            #print "nytt plott"
            ax.clear()
            #ax.set_ylim(1.0e-2,9.9e5)
            n,bins,patches= ax.hist( image.flatten(), 100, log=True, histtype='stepfilled')
            print n, bins, patches
        else:
            #print "gammelt plott"            
            line = lines[0]
            line.set_ydata(image.flatten()) 
        ax.set_title("%s shot#%d"%(name,self.counter))
        ax.figure.canvas.draw()
