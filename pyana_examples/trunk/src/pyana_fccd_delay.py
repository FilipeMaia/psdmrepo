#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z ofte $
#
# Description:
#  Pyana user analysis module pyana_fccd_delay:
#  Averages fccd images in bins of delay time, after selecting events based on ipimb intensity.
#  Delay time is determined from encoder ("crazy-scanner") and phase cavity fit time 1
#  Originally made for sxr27211 (based on myana code from sxr20510).
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

Averages fccd images in bins of delay time, after selecting events based on ipimb intensity.
Delay time is determined from encoder ('crazy-scanner') and phase cavity fit time 1
Originally made for sxr27211 (based on myana code from sxr20510).

@see RelatedModule

@version $Id: template!pyana-module!py 1095 2010-07-07 23:01:23Z ofte $

@author Ingrid Ofte
@author Hubertus Bromberger
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 1095 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging
import time

#-----------------------------
# Imports for other modules --
#-----------------------------
import numpy as np
import scipy as spy
import matplotlib.pyplot as plt

from pypdsdata import xtc

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class pyana_fccd_delay (object) :
    """Class whose instance will be used as a user analysis module. """

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self,
                   image_source = "SxrEndstation-0|Fccd-0",
                   encoder_source = "SxrBeamline-0|Encoder-0",
                   ipimb_source = "SxrBeamline-0|Ipimb-1",
                   start_time = "300", 
                   end_time = "500",   
                   num_bins = "100",   
                   path = "data",
                   ipimb_threshold_upper = "2.0",  # 0.55
                   ipimb_threshold_lower = "0.02", # 0.35
                   ipimb_offset = "0", # 1.22
                   trim_images = False,
                   ) :
        """Class constructor. The parameters to the constructor are passed
        from pyana configuration file. If parameters do not have default 
        values  here then the must be defined in pyana.cfg. All parameters 
        are passed as strings, convert to correct type before use.
        
        @param image_source          Address of FCCD image
        @param encoder_source        Address of encoder
        @param ipimb_source          Address of IPIMB
        @param start_time            delay time first bin
        @param end_time              delay time last bin
        @param num_bins              delay time number of bins
        @param path                  path directory for output files
        @param ipimb_threshold_upper 
        @param ipimb_threshold_lower 
        @param ipimb_offset          
        @param trim_images           Show only the trimmed (480x480) images
        """

        self.img_source = image_source
        self.enc_source = encoder_source
        self.ipimb_source = ipimb_source

        self.fStartTime = float(start_time)
        self.fEndTime   = float(end_time)
        self.iNumBins   = int(num_bins)
        self.FileFolder = path

        self.IpimbThrU   =  float(ipimb_threshold_upper)
        self.IpimbThrL   =  float(ipimb_threshold_lower)
        self.IpimbOffset = float(ipimb_offset)

        self.trim_images = trim_images

        # collect the total average (write to dark file if that is non-existent)
        self.avg_image = None
        self.nev = 0
        self.eventNr   = 0
        self.badEvents = []

        # Encoder Parameters to convert to picoseconds
        self.Delay_a = -80.0e-6;
        self.Delay_b = 0.52168;
        self.Delay_c = 299792458;
        self.Delay_0 = 0;

        # bitshift number for big positive encoder numbers
        self.bitshift = 2 << 24

        self.FoundStartTime = 1e12;
        self.FoundEndTime   = 0;
        self.DeltaTime      = (self.fEndTime - self.fStartTime)/self.iNumBins;

        # histograms
        self.hTime = []
        self.BinCounts = np.zeros(self.iNumBins, dtype=np.uint8)
        self.I0 = np.zeros(self.iNumBins)

        self.ImageArray = np.zeros((self.iNumBins, 500, 576)) # Raw image

        # load dark image file
        darkfile = self.FileFolder + "/dark.npy" 
        self.dark_array = None
        try: 
            self.dark_array = np.load( darkfile )
            print "Dark array %s loaded from %s"%(self.dark_array.shape,darkfile)
        except:
            print "No dark file... "
            try:
                print "... will make one: %s/dark.npy"%(self.FileFolder)
                # if it doesn't exist... make one
                os.mkdir(self.FileFolder)
                print self.fileFolder
            except:
                pass
            
                                   
        
    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job. It should
        do a one-time initialization possible extracting values from event
        data (which is a Configure object) or environment.

        @param evt    event data object
        @param env    environment object
        """

        # Preferred way to log information is via logging package
        logging.info( "pyana_fccd_delay.beginjob() called" )
        self.timer_start = time.clock()

    def beginrun( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "pyana_fccd_delay.beginrun() called" )

    def begincalibcycle( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new calibration cycle.

        @param evt    event data object
        @param env    environment object
        """

        logging.info( "pyana_fccd_delay.begincalibcycle() called" )

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """
        self.nev += 1

        #####################
        # Retreive the Ipimb information for normalization and filtering
        #####################
        try:
            ipmRaw = evt.get(xtc.TypeId.Type.Id_IpimbData, self.ipimb_source)
            i0value = ipmRaw.channel1Volts()
            if i0value < self.IpimbThrL:
                print "Failed ipimb threshold", i0value, self.IpimbThrL 
                self.badEvents.append(self.nev)
                return            
        except:
            print "No %s found in shot#%d" %( self.ipimb_source, self.nev)
            self.badEvents.append(self.nev)
            return

        ####################
        # Get Encoder to determine DelayTime and BinNumber
        ####################
        try:
            encoder = evt.get(xtc.TypeId.Type.Id_EncoderData, self.enc_source )
            Encoder_int = encoder.value()
            if Encoder_int > 5e6 :
                Encoder_int -= self.bitshift                
        except:
            print "No encoder found in shot#", self.nev
            self.badEvents.append(self.nev)
            return

        ###################
        # Phase Cavity to determine DelayTime and BinNumber
        ###################
        try: 
            pc = evt.getPhaseCavity()
            phasecav1 = pc.fFitTime1
            #phasecav2 = pc.fFitTime2 
            #charge1 = pc.fCharge1 
            #charge2 = pc.fCharge2 
        except: 
            print "No phase cavity found in shot#", self.nev
            self.badEvents.append(self.nev)
            return
            
        ####################
        # Calculate DelayTime and BinNumber
        ####################

        DelayTime = 2. * ((self.Delay_a * Encoder_int + self.Delay_b)*1.e-3 / self.Delay_c) \
                    / 1.e-12 + self.Delay_0 + phasecav1
        
        if DelayTime < self.FoundStartTime:
            self.FoundStartTime = DelayTime
        elif DelayTime > self.FoundEndTime:
            self.FoundEndTime = DelayTime
            
        self.hTime.append( DelayTime )

        if DelayTime >= self.fStartTime and DelayTime <= self.fEndTime:
            BinNumber = int((DelayTime-self.fStartTime)/self.DeltaTime)
        else:
            self.badEvents.append((self.nev, DelayTime))
            return


        ##################
        # Process FCCD Frame
        ##################
        try:
            # data from FCCD camera (uint8) 
            data = evt.getFrameValue(self.img_source).data()
            # data is now (500x2*576)
        except: 
            print "No image, %s ev#%d" % (self.img_source,self.nev)
            self.badEvents.append(self.nev)
            return

        # read out as uint16 due to DAQ trick. 
        data.dtype = np.uint16
        # data is now 500x576

        if self.avg_image is None:
            self.avg_image = np.float_(data)
        else:
            self.avg_image += data

        self.ImageArray[BinNumber] += data
        self.I0[BinNumber] += i0value
        self.BinCounts[BinNumber] += 1
            
    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the 
        calibration cycle.
        
        @param env    environment object
        """

        logging.info( "pyana_fccd_delay.endcalibcycle() called" )


    def endrun( self, env ) :
        """This optional method is called if present at the end of the run.
        
        @param env    environment object
        """
        
        logging.info( "pyana_fccd_delay.endrun() called" )

    def endjob( self, env ) :
        """This method is called at the end of the job. It should do 
        final cleanup, e.g. close all open files.
        
        @param env    environment object
        """
        logging.info( "pyana_fccd_delay.endjob() called" )

        print "Starttime: %f" % self.FoundStartTime
        print "Endtime: %f"   % self.FoundEndTime

        numImages = np.sum( self.BinCounts )
        if numImages < 1 :
            print "No images found! "
            return

        # trimmed images (480x480)
        if self.trim_images: 
            self.avg_image = self.trim_image( self.avg_image )
            try:
                self.dark_array = self.trim_image( self.dark_array )
            except:
                pass

            TrimmedImageArray = np.zeros((self.iNumBins, 480, 480)) # Trimmed images
            for bin in range (0, self.iNumBins):
                TrimmedImageArray[bin] = self.trim_image( self.ImageArray[bin] )

            del self.ImageArray
            self.ImageArray = TrimmedImageArray
        
        timer_stop = time.clock()
        duration = timer_stop - self.timer_start
        print "Job duration ", duration


        print "Have collected average image %s from %d events: " %(self.avg_image.shape,numImages)
        if self.dark_array is not None: print "Dark array: ", self.dark_array.shape
        print "Have images from %d bins, number of entries in each: "%self.iNumBins, self.BinCounts


        # compute average
        self.avg_image = self.avg_image / numImages

        #####################################################
        # Plotting
        #####################################################
        plt.ion()

        if self.dark_array is not None:
            plt.figure(200)
            ax200 = plt.imshow(self.dark_array)
            colb200 = plt.colorbar(ax200, pad=0.01)
            plt.title("background")
            plt.draw()

            # subtract background
            self.avg_image -= self.dark_array
        else : 
            darkfile = self.FileFolder + "/dark.npy"
            np.save(darkfile, self.avg_image)
            print "Average image has been saved to ", darkfile


        plt.figure(100)
        plt.title("average")
        ax100 = plt.imshow(self.avg_image)
        colb100 = plt.colorbar(ax100, pad=0.01)
        plt.draw()

        plt.figure(300)
        plt.title("I0 intensity")
        plt.plot(self.I0,'ro')
        plt.draw()


        ################
        # Normalize the averaged images
        ################
        spy.savetxt("%s/BinCounts.txt" % self.FileFolder, self.BinCounts)
        for bin in range (0, self.iNumBins):
            if self.BinCounts[bin] <= 0 : continue # skip to the next bin

            # normalize image to bin counts
            self.ImageArray[bin] = self.ImageArray[bin] / self.BinCounts[bin]
            if self.dark_array is not None:
                # background subtract image
                self.ImageArray[bin] -= self.dark_array

            # normalize i0 intensity to bin count
            self.I0[bin] = self.I0[bin] / self.BinCounts[bin]

            # normalize background-subtracted image to intensity
            if self.dark_array is not None: 
                # if dark, the intensity is likely to be zero
                self.ImageArray[bin] = self.ImageArray[bin] / self.I0[bin]

            
            plt.figure(1)
            plt.clf()
            axim = plt.imshow( self.ImageArray[bin] )            
            colb = plt.colorbar(axim, pad=0.01)
            plt.title("bin %d"%bin)
            plt.draw()
            
            if self.dark_array is not None: 
                self.ImageArray[bin] = self.ImageArray[bin] - self.dark_array
            
                saveeach = True
                if saveeach :
                    fName = "%s/Image%04d.npy"%(self.FileFolder,bin)
                    np.save( fName, self.ImageArray[bin] )
                    
        #plt.hist( np.float_(self.hTime), 100, histtype='stepfilled' )

        plt.ioff()
        plt.show()

        print "Bad events: ", self.badEvents
        print "Total number of events: ", self.nev
        print "Bad events: ", len(self.badEvents)
        print "Good events: ", self.nev - len(self.badEvents)
        

    def trim_image(self, image):
        # return if already trimmed
        if image.shape == (480,480):
            print "Request to trim an already trimmed image... return original image"
            return image

        """
        //  Height:
        //    500 rows   = 6 + 240 * 7 + 240 + 7
        //         Dark A: 6   Rows 0-5
        //       Data Top: 240 Rows 6-245
        //         Dark B: 7   Rows 246-252
        //    Data Bottom: 240 Rows 253-492
        //         Dark C: 7   Rows 493-249
        //
        //  Width (in 16-bit pixels):
        //    576 pixels = 12 * 48 outputs
        //            Top: (10 image pixels followed by 2 info pixels) * 48 outputs
        //         Bottom: (2 info pixels followed by 10 image pixels) * 48 outputs
        """

        DataTop = image[6:246,:]
        DataBottom = image[253:493,:]

        slicesTop = np.split(DataTop,48,axis=1)
        slicesBottom = np.split(DataBottom,48,axis=1)

        slicesTopTrimmed = []
        slicesBottomTrimmed = []
        for i in range( 0, 48 ):
            slicesTopTrimmed.append( slicesTop[i][:,0:10] )
            slicesBottomTrimmed.append( slicesBottom[i][:,2:12] )

        DataTop = np.concatenate( slicesTopTrimmed, axis=1 )
        DataBottom = np.concatenate( slicesBottomTrimmed, axis=1 )

        newimage = np.concatenate( (DataTop,DataBottom) )
        return newimage

