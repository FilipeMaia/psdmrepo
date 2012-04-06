#--------------------------------------------------------------------------
# File and Version Information:
# cspad_assembly
# : Fetch the data and create the CSPadAssembler for downstream modules to use
# : The name of the CsPadAssembler object in the event will be: CsPadAssembler:<source>
#------------------------------------------------------------------------
import sys, os, fnmatch, time
import logging
import numpy as np

from pypdsdata.xtc import TypeId
from cspad         import CsPadAssembler
from pyana.calib   import CalibFileFinder
from pyana import Skip

class cspad_assembly (object) :
    """Class whose instance will be used as a user analysis module. """

    def __init__ ( self,
                   source   = "CxiDs1-0|Cspad-0",
                   ) :
        """
        @param source    Address of detector/device in xtc file.
        """
        self.source = source

    def beginjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_assembly.beginjob() called" )

        cdir = env.calibDir()
        # experiment calibration directory, specified or default
        
        cversion = fnmatch.filter( os.listdir(cdir), "CsPad::CalibV*")[-1]
        # CsPad calibration directory, specified or highest version number

        self.cfinder = CalibFileFinder(cdir,cversion)

        self.starttime = time.time()
        self.n_proc = 0
        self.n_pass = 0
        self.npix_vetoed = []
                
    def beginrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_assembly.beginrun() called" )

        config = env.getConfig(TypeId.Type.Id_CspadConfig,self.source)
        if not config:
            print '*** %s config object is missing ***'%self.source
            return
        
        quads = range(4)
        self.sections = map(config.sections, quads)
        
        self.cspad = CsPadAssembler(devicename=self.source,
                                    config=config,
                                    calibfinder=self.cfinder,
                                    run=evt.run())
        self.cspad.load_pedestals()

        # hand a reference over to the event
        evt.put(self.cspad, "CsPadAssembler:%s"%self.source)


    def begincalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_assembly.begincalibcycle() called" )

        
    def event( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """

        # get CsPad elements from the datagram
        elements = evt.get(TypeId.Type.Id_CspadElement,self.source)
        self.n_proc+=1
        
        # make one large array (4x8x185x388) of pixels
        # (and fill in any blanks if needed)
        self.cspad.fill_pixel_array(elements)
        self.cspad.subtract_pedestals()

        # universal shot counter
        evt.put(self.n_proc,"shot_number")

    def endcalibcycle( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_assembly.endcalibcycle() called" )

    def endrun( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """
        logging.info( "cspad_assembly.endrun() called" )

    def endjob( self, evt, env ) :
        """
        @param evt    event data object
        @param env    environment object
        """        
        logging.info( "cspad_assembly.endjob() called" )

        duration = time.time() - self.starttime
        logging.info("cspad_assembly: Time elapsed: %.3f s"%duration)

