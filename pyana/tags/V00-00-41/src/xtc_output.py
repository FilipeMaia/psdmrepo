#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Pyana user analysis module xtc_output...
#
#------------------------------------------------------------------------

"""User analysis module for pyana framework.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andy Salnikov
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------
import pyana
from pypdsdata import xtc
from pypdsdata import io

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_defFileNameFmt = "e%(expNum)d-r%(run)04d-s%(stream)02d-c%(chunk)02d-filtered.xtc"

#---------------------
#  Class definition --
#---------------------
class xtc_output (object) :
    """Class which implements output module for XTC datagrams."""

    # special member which forces pyana to pass all events to this module
    # even those that are skipped. 
    pyana_get_all_events = True

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, chunk_size_MB = 200*1024, name_fmt = _defFileNameFmt, dir_name = "." ) :
        """
        Constructor takes set of parameters which can be changed in pyana configuration file
        """

        self.chunk_size_MB = int(chunk_size_MB)
        self.name_fmt = name_fmt
        self.dir_name = dir_name
        self.keep_epics = True

        self.chunk = 0
        self.run = 0
        self.expNum = 0
        self.stream = 0
        
        self.file = None
        self.filter = io.XtcFilter(io.XtcFilterTypeId([xtc.TypeId.Type.Id_Epics], []))
        
    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :

        logging.debug( "xtc_output.beginjob() called" )

        # run number should be known already at this point
        self.run = evt.run() or 0
        self.expNum = evt.expNum() or 0

        # stream is determined from subprocess index
        self.stream = env.subprocess()
        if self.stream < 0: self.stream = 0
        
        # open file
        self._openFile()

        # send current datagram to file 
        self._saveDg(evt.m_dg, "beginjob", xtc.TransitionId.Configure)

    def beginrun( self, evt, env ) :

        logging.debug( "xtc_output.beginrun() called" )

        # send current datagram to file 
        self._saveDg(evt.m_dg, "beginrun", xtc.TransitionId.BeginRun)

    def begincalibcycle( self, evt, env ) :

        logging.debug( "xtc_output.begincalibcycle() called" )

        # send current datagram to file 
        self._saveDg(evt.m_dg, "begincalibcycle", xtc.TransitionId.BeginCalibCycle)

    def event( self, evt, env ) :

        # send current datagram to file 
        if evt.status() == pyana.Normal:
            logging.debug("xtc_output.event(): saving complete datagram")
            self._saveDg(evt.m_dg, "event", xtc.TransitionId.L1Accept)
        else:
            if self.keep_epics:
                # filter out everything except Epics
                logging.debug("xtc_output.event(): filtering datagram")
                dg = self.filter.filter(evt.m_dg)
                if dg:
                    logging.debug("xtc_output.event(): saving epics datagram")
                    self._saveDg(dg, "event", xtc.TransitionId.L1Accept)

    def endcalibcycle( self, evt, env ) :

        logging.debug( "xtc_output.endcalibcycle() called" )

        # send current datagram to file 
        self._saveDg(evt.m_dg, "endcalibcycle", xtc.TransitionId.EndCalibCycle)

    def endrun( self, evt, env ) :

        logging.debug( "xtc_output.endrun() called" )

        # send current datagram to file 
        self._saveDg(evt.m_dg, "endrun", xtc.TransitionId.EndRun)

    def endjob( self, env ) :

        logging.debug( "xtc_output.endjob() called" )

        # close the file
        self.file.close()
        self.file = None


    def _openFile(self):
        """
        Open or re-open output file
        """
        
        # if it is open the close it first
        if self.file: 
            self.file.close()
            self.file = None

        # open file
        fname = os.path.join(self.dir_name, self.name_fmt % self.__dict__)
        logging.info( "xtc_output._openFile(): opening new file: %s", fname )
        self.file = open(fname, "wb")


    def _saveDg(self, dg, method, type):
        """
        Write datagram to output file after checking that it has expected type.
        If type is different from expected then datagram is not written and warning 
        message is produced
        """
        
        if dg is None:
            logging.warning("xtc_output.%s: no datagram provided", method)
            return
        
        # check that datagram has correct transition type
        if dg.seq.service() != type:
            logging.warning("xtc_output.%s: datagram has unexpected type: %s, will skip datagram", method, dg.seq.service())
            return

        # save datagram
        self.file.write(dg)
