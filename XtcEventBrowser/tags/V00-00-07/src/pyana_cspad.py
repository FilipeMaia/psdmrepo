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


#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import xtc


#---------------------
#  Class definition --
#---------------------
class  pyana_cspad ( object ) :

    # initialize
    def __init__ ( self,
                   image_source=None,
                   draw_each_event = 0,
                   collect_darks = 0,
                   dark_img_file = None):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param image_source     address string of Detector-Id|Device-ID
        @param draw_each_event  0 (False) or 1 (True) Draw plot for each event
        @param collect_darks    0 (False) or 1 (True) Collect dark images (write to file)
        @param dark_img_file    filename (If collecting: write to this file)
        """

        self.img_addr = image_source
        print "Using image_source = ", self.img_addr

        self.draw_each_event = bool(int(draw_each_event))
        print "Using draw_each_event = ", draw_each_event

        self.collect_darks = bool(int(collect_darks))
        print "Collecting darks? " , self.collect_darks

        self.dark_img_file = dark_img_file
        print "Using dark image file: ", self.dark_img_file

        # sum up all image data (above threshold) and all dark data (below threshold)
        self.img_data = None

        self.colmin = 0
        self.colmax = 16360

        # these will be plotted too
        self.lolimits = []
        self.hilimits = []

        # to keep track
        self.n_events = 0
        self.n_img = 0

        # load dark image
        self.dark_image = None
        if not self.collect_darks :
            if self.dark_img_file is None :
                print "No dark-image file provided. The images will not be background subtracted."
            else :
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
        
        self.sections = map(config.sections, quads)
        

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
        
        #fig = plt.figure(200)
        
        # dump information about quadrants
        #print "Number of quadrants: %d" % len(quads)
        qimages = np.zeros((4,776,776), dtype="uint16")

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
            
            qimage = self.CsPadElement(data, q.quad())
            qimages[q.quad()] = qimage

            #ax = fig2.add_subplot(2,2,q.quad())
            #ax.set_title("Q %d" % q.quad() )
            #axes = plt.imshow( qimage, origin='lower')


        h1 = np.hstack( (qimages[0], qimages[1]) )
        h2 = np.hstack( (qimages[3], qimages[2]) )
        cspad_image = np.vstack( (h1, h2) )
        self.vmax = np.max(cspad_image)
        self.vmin = np.min(cspad_image)

        # collect min and max intensity of this image
        self.lolimits.append( self.vmin )
        self.hilimits.append( self.vmax )

        # add this image to the sum
        self.n_img+=1
        if self.img_data is None :
            self.img_data = np.float_(cspad_image)
        else :
            self.img_data += cspad_image


        # Draw this event. Background subtracted if possible.
        if self.draw_each_event :
            if self.dark_image is None: 
                self.drawframe(cspad_image,"Event # %d" % self.n_events )
            else :
                subtr_image = cspad_image - self.dark_image 
                title = "Event # %d, background subtracted" % self.n_events 
                self.drawframe(subtr_image, title )
                        



    # after last event has been processed. 
    def endjob( self, env ) :

        print "Done processing       ", self.n_events, " events"        
        
        # plot the minimums and maximums
        print len(self.lolimits)
        xaxis = np.arange(self.n_events)
        plt.plot( xaxis, np.array(self.lolimits), "gv", xaxis, np.array(self.hilimits), "r^" )
        plt.title("high (A) and low (V) limits")
        plt.show()
        print "Show?"

        #plt.plot( np.array(self.lolimits))
        #plt.plot( np.array(self.hilimits))

        if self.img_data is None :
            print "No image data found from source ", self.img_addr
            return

        # plot the average image
        average_image = self.img_data/self.n_img
        self.drawframe(average_image,"Average of %d events" % self.n_img )
        plt.show()


        # save the average data image (numpy array)
        # binary file .npy format
        if self.collect_darks :
            print "saving to ",  self.dark_img_file
            np.save(self.dark_img_file, average_image)
  
        


    # -------------------------------------------------------------------
    # Additional functions
        
    def CsPadElement( self, data3d, qn ):
        # Construct one image for each quadrant, each with 8 sections
        # from a data3d = 3 x 2*194 x 185 data array
        #   +---+---+-------+
        #   |   |   |   6   |
        #   + 5 | 4 +-------+
        #   |   |   |   7   |
        #   +---+---+---+---+
        #   |   2   |   |   |
        #   +-------+ 0 | 1 |
        #   |   3   |   |   |
        #   +-------+---+---+

        zeros = np.zeros((18,388),dtype=data3d.dtype)
        zeros9 = np.zeros((9,388),dtype=data3d.dtype)
        zeros6 = np.zeros((6,388),dtype=data3d.dtype)

        # if any sections are missing, insert zeros
        if len( data3d ) < 8 :
            zsec = np.zeros( (185,388), dtype=data3d.dtype)
            for i in range (8) :
                if i not in self.sections[qn] :
                    data3d = np.insert( data3d, i, zsec, axis=0 )
                #print "section ", i
                #print data3d[i]


        s01 = np.concatenate( (zeros6.T,
                               data3d[0][:,::-1].T,
                               zeros6.T,
                               data3d[1][:,::-1].T,
                               zeros6.T),
                              1)
        s23 = np.concatenate( (zeros6,
                               data3d[2], 
                               zeros6,
                               data3d[3],
                               zeros6 ),
                              0 )
        s45 = np.concatenate( (zeros6.T,
                               data3d[5][::-1,:].T,
                               zeros6.T,
                               data3d[4][::-1,:].T,
                               zeros6.T), 
                              1 )
        s67 = np.concatenate( (zeros6,
                               data3d[6], 
                               zeros6,
                               data3d[7],
                               zeros6 ),
                              0 )

        m1 = np.hstack( (s23, s01) )
        m2 = np.hstack( (s45, s67) )
        e0 = np.vstack( (m2, m1) )

        if qn>0 : e0 = np.rot90( e0, 4-qn)
        return e0


    def drawframe( self, frameimage, title="", fig = None ):

        # plot image frame
        #if fig is None :
        #    fig = plt.figure( 1 )

        self.fig = plt.figure(200)
        cid1 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas = self.fig.add_subplot(111)

        self.canvas.set_title(title)
        # canvas is the main "Axes" object

        self.axes = plt.imshow( frameimage )#, origin='lower' )
        # axes is the are where the image is plotted
        
        self.colb = plt.colorbar(self.axes,pad=0.01)
        # colb is the colorbar object

        self.orglims = self.axes.get_clim()
        # min and max values in the axes are
        print "Original value limits: ", self.orglims

        plt.clim(self.colmin,self.colmax)
        plt.draw() # redraw the current figure

        print """
        To change the color scale, click on the color bar:
          - left-click sets the lower limit
          - right-click sets higher limit
          - middle-click resets to original
        """

        #plt.draw() 
        plt.show() # starts the GUI main loop
                   # you need to kill window to proceed... 
                   # (this shouldn't be done for every event!)



                               
    # define what to do if we click on the plot
    def onclick(self, event) :

        # can we open a dialogue box here?
        print 'mouse click: button=', event.button,' x=',event.x, ' y=',event.y,
        print ' xdata=',event.xdata,' ydata=', event.ydata

        if event.inaxes :
            lims = self.axes.get_clim()
            
            self.colmin = lims[0]
            self.colmax = lims[1]
            range = self.colmax - self.colmin
            value = self.colmin + event.ydata * range
            #print colmin, colmax, range, value
            
            # left button
            if event.button is 1 :
                if value > self.colmin and value < self.colmax :
                    self.colmin = value
                    print "new mininum: ", self.colmin
                else :
                    print "min has not been changed (click inside the color bar to change the range)"
                        
            # middle button
            elif event.button is 2 :
                self.colmin, self.colmax = self.orglims
                print "reset"
                    
            # right button
            elif event.button is 3 :
                if value > self.colmin and value < self.colmax :
                    self.colmax = value
                    print "new maximum: ", self.colmax
                else :
                    print "max has not been changed (click inside the color bar to change the range)"

            plt.clim(self.colmin,self.colmax)
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

    
