#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#   Module pyana_cspad
#   pyana module with intensity threshold, plotting with matplotlib, allow rescale color plot
#
#   Example xtc file: /reg/d/psdm/sxr/sxrcom10/xtc/e29-r0603-s00-c00.xtc 
#
#   To run: pyana -m mypkg.pyana_cspad <filename>
#
"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or
part of it, please give an appropriate acknowledgment.
@author Ingrid Ofte
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import time

import numpy as np

import matplotlib 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import AxesGrid

from matplotlib.gridspec import GridSpec


#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import xtc
from cspad import CsPad

#---------------------
#  Class definition --
#---------------------

#class MyRectangle(plt.Rectangle):
#    def __call__(self, ax):
#        self.set_bounds(*ax.viewLim.bounds)
#        ax.figure.canvas.draw_idle()

class  pyana_cspad ( object ) :

    #--------------------
    #  Class variables --
    #--------------------
    
    #----------------
    #  Constructor --
    #----------------
            
    # initialize
    def __init__ ( self,
                   image_source=None,
                   draw_each_event = 0,
                   dark_img_file = None,
                   output_file = None,
                   plot_vrange = None,
                   threshold = None,
                   thr_area = None ):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param image_source     string, Address of Detector-Id|Device-ID
        @param draw_each_event  bool, Draw plot for each event? (Default=False). 
        @param dark_img_file    filename, Dark image file to be loaded, if any
        @param output_file      filename (If collecting: write to this file)
        @param plot_vrange      range=vmin-vmax of values for plotting (pixel intensity)
        @param threshold        lower threshold for image intensity in threshold area of the plot
        @param thr_area         range=xmin,xmax,ymin,ymax defining threshold area
        """

        # initializations from argument list
        self.img_addr = image_source
        print "Using image_source = ", self.img_addr

        self.draw_each_event = bool(draw_each_event)
        if self.draw_each_event and ( draw_each_event == "No" or
                                      draw_each_event == "0" or
                                      draw_each_event == "False" ) : self.draw_each_event = False
        print "Using draw_each_event = ", self.draw_each_event


        self.dark_img_file = dark_img_file
        if dark_img_file == "" or dark_img_file == "None" : self.dark_img_file = None
        print "Using dark image file: ", self.dark_img_file

        self.output_file = output_file
        if output_file == "" or output_file == "None" : self.output_file = None
        print "Using output_file: ", self.output_file

        self.plot_vmin = None
        self.plot_vmax = None
        if plot_vrange is not None and plot_vrange is not "" : 
            self.plot_vmin = float(plot_vrange.split("-")[0])
            self.plot_vmax = float(plot_vrange.split("-")[1])
            print "Using plot_vrange = %f-%f"%(self.plot_vmin,self.plot_vmax)

        self.threshold = None
        if threshold is not None :
            self.threshold = float(threshold)
            print "Using threshold value ", self.threshold

        # subset of image where threshold is applied
        self.thr_rect = None
        self.thr_area = None
        if thr_area is not None: 
            self.thr_area = np.array([0.,0.,0.,0.])
            for i in range (4) : self.thr_area[i] = float(thr_area.split(",")[i])
            print "Using threshold region ", self.thr_area

        # initializations of other class variables

        # sum of image data
        self.img_data = None

        # these will be plotted too
        self.lolimits = []
        self.hilimits = []

        # to keep track
        self.n_events = 0
        self.n_img = 0

        # load dark image
        self.dark_image = None
        if self.dark_img_file is None :
            print "No dark-image file provided. The images will not be background subtracted."
        else :
            print "Loading dark image from ", self.dark_img_file
            self.dark_image = np.load(self.dark_img_file)




    # start of job
    def beginjob ( self, evt, env ) : 

        config = env.getConfig(xtc.TypeId.Type.Id_CspadConfig, self.img_addr )
        if not config:
            print '*** cspad config object is missing ***'
            return
        
        quads = range(4)
        
        print "Cspad configuration"
        print "  N quadrants   : %d" % config.numQuads()
        print "  Quad mask     : %#x" % config.quadMask()
        print "  payloadSize   : %d" % config.payloadSize()
        print "  badAsicMask0  : %#x" % config.badAsicMask0()
        print "  badAsicMask1  : %#x" % config.badAsicMask1()
        print "  asicMask      : %#x" % config.asicMask()
        print "  numAsicsRead  : %d" % config.numAsicsRead()
        try:
            # older versions may not have all methods
            print "  roiMask       : [%s]" % ', '.join([hex(config.roiMask(q)) for q in quads])
            print "  numAsicsStored: %s" % str(map(config.numAsicsStored, quads))
        except:
            pass
        print "  sections      : %s" % str(map(config.sections, quads))
        
        self.cspad = CsPad(config)


    # process event/shot data
    def event ( self, evt, env ) :


        self.images = []
        self.ititle = []

        # this one counts every event
        self.n_events+=1

        # print a progress report
        if (self.n_events%1000)==0 :
            print "Event ", self.n_events
        
        quads = evt.getCsPadQuads(self.img_addr, env)
        if not quads :
            print '*** cspad information is missing ***'
            return
        
        # dump information about quadrants
        #print "Number of quadrants: %d" % len(quads)
        qimages = np.zeros((4, self.cspad.npix_quad, self.cspad.npix_quad ), dtype="uint16")

        for q in quads:
            
            #print "  Quadrant %d" % q.quad()
            #print "    virtual_channel: %s" % q.virtual_channel()
            #print "    lane: %s" % q.lane()
            #print "    tid: %s" % q.tid()
            #print "    acq_count: %s" % q.acq_count()
            #print "    op_code: %s" % q.op_code()
            #print "    seq_count: %s" % q.seq_count()
            #print "    ticks: %s" % q.ticks()
            #print "    fiducials: %s" % q.fiducials()
            #print "    frame_type: %s" % q.frame_type()
            #print "    sb_temp: %s" % map(q.sb_temp, range(4))
            
            # image data as 3-dimentional array
            data = q.data()
            #print "min and max of original array for quad#%d: %d, %d" %(q.quad(),np.min(data),np.max(data))
            
            qimage = self.cspad.CsPadElement(data, q.quad())
            qimages[q.quad()] = qimage

            #ax = fig2.add_subplot(2,2,q.quad())
            #ax.set_title("Q %d" % q.quad() )
            #axes = plt.imshow( qimage, origin='lower')


        # need to do this a better way:
        h1 = np.hstack( (qimages[0], qimages[1]) )
        h2 = np.hstack( (qimages[3], qimages[2]) )
        cspad_image = np.vstack( (h1, h2) )
        self.vmax = np.max(cspad_image)
        self.vmin = np.min(cspad_image)

        # collect min and max intensity of this image
        self.lolimits.append( self.vmin )
        self.hilimits.append( self.vmax )

        # subtract background if provided
        if self.dark_image is not None: 
            cspad_image = cspad_image - self.dark_image 

        # set threshold
        if self.threshold is not None:
            if self.thr_area is not None:
                subset = cspad_image[self.thr_area[0]:self.thr_area[1],   # x1:x2
                                     self.thr_area[2]:self.thr_area[3]]   # y1:y2
                if np.max(subset) < self.threshold :
                    print "skipping this event!  %f < %f " % (float(np.max(subset)), float(self.threshold))
                    return
            else :
                if np.max(cspad_image) < self.threshold :
                    print "skipping this event!  %f < %f " % (float(np.max(subset)), float(self.threshold))
                    return
                
                print "Threshold area min,max = %.2f, %.2f " % (np.min(subset),np.max(subset))

        # add this image to the sum
        self.n_img+=1
        if self.img_data is None :
            self.img_data = np.float_(cspad_image)
        else :
            self.img_data += cspad_image


        # Draw this event.
        title = "Event # %d" % self.n_events
        if self.dark_image is not None:
            title = title + " (background subtracted) "
            
        if self.draw_each_event :
            self.drawframe(cspad_image,title, fignum=200 )

        plt.show()



    # after last event has been processed. 
    def endjob( self, env ) :

        print "Done processing       ", self.n_events, " events"        
        
#        # plot the minimums and maximums
#        print len(self.lolimits)
#        xaxis = np.arange(self.n_events)
#        plt.clf()
#        plt.plot( xaxis, np.array(self.lolimits), "gv", xaxis, np.array(self.hilimits), "r^" )
#        plt.title("high (A) and low (V) limits")


        if self.img_data is None :
            print "No image data found from source ", self.img_addr
            return

        # plot the average image
        average_image = self.img_data/self.n_img 
        self.drawframe(average_image,"Average of %d events" % self.n_img, fignum=100 )
        plt.show()


        # save the average data image (numpy array)
        # binary file .npy format
        if self.output_file is not None :
            if ".npy" in self.output_file :
                print "saving to ",  self.output_file
                np.save(self.output_file, average_image)
            else :
                print "outputfile file does not have the required .npy ending..."
                svar = raw_input("Do you want to provide an alternative file name? ")
                if svar == "" :
                    print "Nothing saved"
                else :
                    if ".npy" not in svar:
                        print "I still don't like your file name, saving anyway..."
                    print "saving to ",  svar
                    np.save(svar, average_image)
                
        print "-------------------"
        print "Done running pyana."
        print "To run pyana again, edit config file if needed and hit \"Run pyana\" button again"
        print "Send any feedback on this program to ofte@slac.stanford.edu"
        print "Thank you!"
        print "-------------------"

    # -------------------------------------------------------------------
    # Additional functions

    def drawframemore( self, frameimage, title="", fignum=1):

        # plot image frame
        #if fig is None :

        self.fig = plt.figure(figsize=(8,10),num=fignum)
        #plt.suptitle("LCLS Event Display")
        cid1 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        ## To do: add insert showing the active region for thresholding
        #if self.thr_area is not None:
        #cspad_image[self.thr_area[0]:self.thr_area[1], self.thr_area[2]:self.thr_area[3]])

        gs = GridSpec(2,1, height_ratios=[6,1] )

        axes1 = plt.subplot(gs[0])
        axes1.set_title(title)
        # the main "Axes" object (on where the image is plotted)

        print "vmin vmax now: ", self.plot_vmin, self.plot_vmax
        self.axesim = plt.imshow( frameimage, vmin=self.plot_vmin, vmax=self.plot_vmax )#, origin='lower' )
        # axes image 
        
        self.colb = plt.colorbar(self.axesim,pad=0.01)
        # colb is the colorbar object


        if self.plot_vmin is None: 
            self.orglims = self.axesim.get_clim()
            # min and max values in the axes are
            print "Original value limits: ", self.orglims
            self.plot_vmin, self.plot_vmax = self.orglims

        axes2 = plt.subplot(gs[1])
        axes2.set_title("spectrum")
        self.axesim2 = plt.hist( frameimage.ravel(),
                                 bins=1000,
                                 histtype='stepfilled',
                                 range=(self.plot_vmin,self.plot_vmax))
        
        print """
        To change the color scale, click on the color bar:
          - left-click sets the lower limit
          - right-click sets higher limit
          - middle-click resets to original
        """
        #plt.show() # starts the GUI main loop
        #           # you need to kill window to proceed... 
        #           # (this shouldn't be done for every event!)


    def drawframe( self, frameimage, title="", fignum=1):

        # plot image frame
        #if fig is None :

        self.fig = plt.figure(num=fignum)
        cid1 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid2 = self.fig.canvas.mpl_connect('pick_event', self.onpick)
        axes = self.fig.add_subplot(111)
        
        axes.set_title(title)
        # the main "Axes" object (on where the image is plotted)

        self.axesim = plt.imshow( frameimage, vmin=self.plot_vmin, vmax=self.plot_vmax )#, origin='lower' )
        # axes image 
        
        print "vmin vmax now: ", self.plot_vmin, self.plot_vmax

        self.colb = plt.colorbar(self.axesim,pad=0.01)
        # colb is the colorbar object

        if self.plot_vmin is None: 
            self.orglims = self.axesim.get_clim()
            # min and max values in the axes are
            print "Original value limits: ", self.orglims
            self.plot_vmin, self.plot_vmax = self.orglims
        
        print """
        To change the color scale, click on the color bar:
          - left-click sets the lower limit
          - right-click sets higher limit
          - middle-click resets to original
        """

        ## show the active region for thresholding
        if self.thr_area is not None:
            xy = [self.thr_area[0],self.thr_area[2]]
            w = self.thr_area[1] - self.thr_area[0]
            h = self.thr_area[3] - self.thr_area[2]
            self.thr_rect = plt.Rectangle(xy,w,h, facecolor='none', edgecolor='red', picker=5)
            axes.add_patch(self.thr_rect)

    
        #cspad_image[self.thr_area[0]:self.thr_area[1], self.thr_area[2]:self.thr_area[3]])
        
        

        #plt.show() # starts the GUI main loop
        #           # you need to kill window to proceed... 
        #           # (this shouldn't be done for every event!)


    def onpick(self, event):
        print "Currently active area for threshold evaluation: [xmin xmax ymin ymax] = ", self.thr_area
        print "To change this area, right-click..." 
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
            plt.draw()
        
                               
    # define what to do if we click on the plot
    def onclick(self, event) :

        # can we open a dialogue box here?
        if not event.inaxes and event.button == 3 :
            print "can we open a menu here?"
            
        # check that the click was on the color bar
        if event.inaxes == self.colb.ax :

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
                #if value > self.plot_vmin and value < self.plot_vmax :
                #else :
                #    print "min has not been changed (click inside the color bar to change the range)"
                        
            # middle button
            elif event.button is 2 :
                self.plot_vmin, self.plot_vmax = self.orglims
                print "reset"
                    
            # right button
            elif event.button is 3 :
                self.plot_vmax = value
                print "maximum changed:   ( %.2f , %.2f ) " % (self.plot_vmin, self.plot_vmax )
                #if value > self.plot_vmin and value < self.plot_vmax :
                #else :
                #    print "max has not been changed (click inside the color bar to change the range)"

            plt.clim(self.plot_vmin,self.plot_vmax)
            plt.draw() # redraw the current figure




    # define what to do if a button is pressed
    def onpress(self, event) :

        if event.key not in ('t', 'l'): return
        if event.key=='t' : self.set_threshold()
        if event.key=='l' : self.add_savelist()
        

    def set_threshold(self) :
        print " open a dialog to change the threshold to a new value"
        pass


    def add_savelist(self) :
        print "Schedule this image array for saving to binary file"
        pass

    
