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
                   good_range="0--999999",
                   dark_range="-999999--0",
                   draw_each_event = False):
        """Class constructor.
        Parameters are passed from pyana.cfg configuration file.
        All parameters are passed as strings

        @param image_source     address string of Detector-Id|Device-ID
        @param good_range       threshold values selecting images of interest
        @param dark_range       threshold values selecting dark images
        @param draw_each_event  bool
        """

        self.img_addr = image_source
        print "Using image_source = ", self.img_addr

        tli = str(good_range).split("--")[0]
        thi = str(good_range).split("--")[1]
        tld = str(dark_range).split("--")[0]
        thd = str(dark_range).split("--")[1]

        self.thr_low_image = float( tli )
        self.thr_high_image = float( thi )
        self.thr_low_dark = float( tld )
        self.thr_high_dark = float( thd )
        print "Using good_range = %s " % good_range
        print "  (thresholds =  %d (low) and %d (high) " % (self.thr_low_image, self.thr_high_image)
        print "Using dark_range = %s " % dark_range
        print "  (thresholds =  %d (low) and %d (high) " % (self.thr_low_dark, self.thr_high_dark)

        self.draw_each_event = draw_each_event
        print "Using draw_each_event = ", draw_each_event

        # sum up all image data (above threshold) and all dark data (below threshold)
        self.img_data = None
        self.dark_data = None

        # these will be plotted too
        self.lolimits = []
        self.hilimits = []

        # to keep track
        self.n_events = 0
        self.n_img = 0
        self.n_dark = 0

        self.fig = plt.figure()
        cid1 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas = self.fig.add_subplot(111)



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
        
        #fig2 = plt.figure(200)
        
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
        vmax = np.max(cspad_image)
        vmin = np.min(cspad_image)

        self.drawframe(cspad_image,"Event # %d" % self.n_events, fig=self.fig )

        # collect min and max intensity of this image
        self.lolimits.append( vmin )
        self.hilimits.append( vmax )

        # select good images
        isGood = False
        if ( cspad_image.max() > self.thr_low_image) and (cspad_image.max() < self.thr_high_image) :
            isGood = True
            
            # add this image to the sum
            self.n_img+=1
            if self.img_data is None :
                self.img_data = np.float_(cspad_image)
            else :
                self.img_data += cspad_image

        # select dark image
        isDark = False
        if ( cspad_image.max() > self.thr_low_dark ) and ( cspad_image.max() < self.thr_high_dark ) :
            isDark = True
            self.n_dark+=1
            if self.dark_data is None :
                self.dark_data = np.float_(cspad_image)
            else :
                self.dark_data += cspad_image

        # Draw this event. Background subtracted if possible.
        if self.draw_each_event and isGood :
            if self.n_dark > 0 :
                av_dark_img = self.dark_data/self.n_dark
                subimage = cspad_image - av_dark_img 
                title = "Event %d, background subtracted (avg of %d dark images)" % \
                        ( self.n_events, self.n_dark )
                self.drawframe( subimage, title )
            else :
                title = "Event %d " % self.n_events
                self.drawframe( cspad_image, title )

                


    # after last event has been processed. 
    def endjob( self, env ) :

        print "Done processing       ", self.n_events, " events"        

        print "Range defining images: %f (lower) - %f (upper)" % (self.thr_low_image, self.thr_high_image)
        print "Range defining darks: %f (lower) - %f (upper)" %  (self.thr_low_dark, self.thr_high_dark)
        print "# Signal images = ", self.n_img
        print "# Dark images = ", self.n_dark
        
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
        av_good_img = self.img_data/self.n_img
        av_bkgsubtracted = av_good_img 
        self.drawframe( av_good_img, "Average of images above threshold")

        if self.n_dark>0 :
            av_dark_img = self.dark_data/self.n_dark
            av_bkgsubtracted -= av_dark_img 
            self.drawframe( av_dark_img, "Average of images below threshold" )
            self.drawframe( av_bkgsubtracted, "Average background subtracted")

        plt.show()



    # -------------------------------------------------------------------
    # Additional functions
        
    def CsPadElement( self, data3d, qn ):
        # Construct one image for each quadrant, each with 8 sections
        # from a data3d = 3 x 2*194 x 185 data array

        print "make a big array from smaller ones"
        print "original data array shape for quad # %d: %s" % (qn, str(np.shape(data3d)) )
        print "original data types: ", data3d.dtype.name

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

        print "final Q%d shape: %s " % (qn,str(np.shape(e0)))
        if qn>0 : e0 = np.rot90( e0, 4-qn)
        return e0


    def drawframe( self, frameimage, title="", fig = None ):

        # plot image frame
        if fig is None :
            fig = plt.figure( 1 )

        self.canvas.set_title(title)
        # canvas is the main "Axes" object

        self.axes = plt.imshow( frameimage )#, origin='lower' )
        # axes is the are where the image is plotted
        
        self.colb = plt.colorbar(self.axes,pad=0.01)
        # colb is the colorbar object

        self.orglims = self.axes.get_clim()
        # min and max values in the axes are
        print "Original value limits: ", self.orglims

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
            
            colmin = lims[0]
            colmax = lims[1]
            range = colmax - colmin
            value = colmin + event.ydata * range
            #print colmin, colmax, range, value
            
            # left button
            if event.button is 1 :
                if value > colmin and value < colmax :
                    colmin = value
                    print "new mininum: ", colmin
                else :
                    print "min has not been changed (click inside the color bar to change the range)"
                        
            # middle button
            elif event.button is 2 :
                colmin, colmax = self.orglims
                print "reset"
                    
            # right button
            elif event.button is 3 :
                if value > colmin and value < colmax :
                    colmax = value
                    print "new maximum: ", colmax
                else :
                    print "max has not been changed (click inside the color bar to change the range)"

            plt.clim(colmin,colmax)
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

    
