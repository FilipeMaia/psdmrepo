#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module XtcScanner...
#
#------------------------------------------------------------------------

"""Command-line utility to summarize contents of an xtc file

Scans through the xtc file(s) provided and counts the information
within. At the end it outputs a summary of detectors/devices,
number of events and calibration cycles found within this data.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see pyxtcsummary

@version $Id: XtcScanner 0 2011-01-24 11:45:00 ofte $

@author Ingrid Ofte
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 0 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import time
from optparse import OptionParser

#---------------------------------
#  Imports of base class module --
#---------------------------------


#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import io
from pypdsdata.xtc import *
from pypdsdata.epics import *


#----------------------------------
# Local non-exported definitions --
#----------------------------------


#------------------------
# Exported definitions --
#------------------------


#---------------------
#  Class definition --
#---------------------
class XtcScanner ( object ) :
    """ Scans the xtc file(s) and makes a summary of what's there

    Crawls through the xtc file(s) provided and counts the information
    within. At the end it outputs a summary of detectors/devices,
    number of events and calibration cycles found within this data.
    
    @see pyxtcreader
    """

    #--------------------
    #  Class variables --
    #--------------------
    devices = {}
    counters = {}
    epicsPVs = []

    #----------------
    #  Constructor --
    #----------------
    def __init__(self, filenames = [], options={'ndatagrams':-1, 'l1_offset': 0, 'verbose': 0} ):
        """Constructor.

        Initialize list of files and options and creates
        several python lists to hold scan information.

        @param filenames    a python list of xtc file names
        @param options      a python dictionary of options
        """

        # define instance variables
        self.files = filenames
        self.options = options

        # counter for calibcycles and nevents/calibcycle
        self.ncalib = 0
        self.nevents = []
                        
        # keep track of current state
        self._state = ''
        self._counter = 0
        

    #-------------------
    #  Public methods --
    #-------------------

    def scan(self):
        """Scan each xtc datagram

        For each xtc datagram that is of type
        datagram.xtc, call _scan.
        """

        if len(self.files)==0 :
            print "You need to select an xtc file"
            return

        print "Start parsing files: \n", self.files
        start = time.clock()

        # read datagrams one by one

        # print progress bar
        progr = 0
        print ". ",

        xtciter = io.XtcMergeIterator( self.files, self.options['l1_offset'] )
        for dg in xtciter :

            if self.options['ndatagrams']>-1 :
                if progr > self.options['ndatagrams'] : break
            seq = dg.seq
            xtc = dg.xtc

            self._state = str(seq.service())

            if self._state == "BeginCalibCycle" :
                self.ncalib+=1;
                self.nevents.append(0)
            if self._state == "L1Accept" :
                self.nevents[(self.ncalib-1)]+=1


            # recursively dump Xtc info
            self._scan ( xtc )

            # progress bar
            progr+=1
            if (progr%100) == 1:
                print "\r  %d datagrams read " % progr ,
                sys.stdout.flush()
            if (progr%10) == 0:
                print " . " ,
                sys.stdout.flush()

        elapsed = ( time.clock() - start )
        print "\r  %d datagrams read in %f s " % (progr, elapsed)

        self.printSummary(opt_epics=0)


    def addCountInfo(self):
        pass



    def printSummary(self, opt_bld = 1, opt_det=1, opt_epics = 0):

        print "Here's what I find: "
        print "  - %d calibration cycles." % self.ncalib
        print "  - Events per calib cycle: \n  ", self.nevents
        print

        print "Information from ", len(self.devices), " items found"
        sortedkeys = sorted( self.devices.keys() )
        for d in sortedkeys :
            print d, ": \t    ",  
            for i in range ( len(self.devices[d] ) ):
                print " %s (%d) " % ( self.devices[d][i],  self.counters[d][i] ),
            print

        if opt_epics > 0 :
            print "Epics PVs: ", self.epicsPVs



    def setFiles(self, filenames):
        self.files = filenames

    def showFiles(self):
        print self.files


    # set one or more options
    def setOption(self, anoption):
        self.options.update(anoption)

    def showOptions(self):
        print self.options

    def ndev(self):
        return len(self.devices)

    #--------------------------------
    #  Static/class public methods --
    #--------------------------------


    #--------------------
    #  Private methods --
    #--------------------

    def _scan(self, xtc ) :
        """Scan this datagram

        Read a given Xtc object (datagram) recursively. 
        Add to the various lists
        """
        self._counter+=1

        # some kind of damage may break even 'contains'
        if xtc.damage.hasDamage(Damage.Value.IncompleteContribution) :
            print "damage found in scan (%d) of %s ", self._counter, xtc.src
            return

        if xtc.contains.id() == TypeId.Type.Id_Xtc :
            for x in xtc :
                self._scan( x )
        else :

            source = str(xtc.src)
            contents = str(xtc.contains)

            dtype = type(xtc.src).__name__
            dname = ''
            if ( dtype=='BldInfo' ):
                dname = str(xtc.src.type()).split('.')[1]
            if ( dtype=='DetInfo' ):
                dtn = str(xtc.src.detector()).split('.')[1]
                dti = str(xtc.src.detId())
                dvn = str(xtc.src.device()).split('.')[1]
                dvi = str(xtc.src.devId());
                dname = dtn+"-"+dti+"|"+dvn+"-"+dvi

            dkey = dtype + ':' + dname

            if self._state == "Configure" :
                if xtc.contains.id() == TypeId.Type.Id_Epics :
                    data = xtc.payload()
                    self.epicsPVs.append(data.sPvName)
                    
            if dkey not in self.devices :
                # first occurence of this detector/device
                self.devices[dkey] = []
                self.devices[dkey].append( contents )
                self.counters[dkey] = []
                self.counters[dkey].append( 1 )
            else :
                # new type of data contents
                if self.devices[dkey].count( contents )==0 :
                    self.devices[dkey].append( contents )
                    self.counters[dkey].append( 1 )
                else :
                    indx = self.devices[dkey].index( contents )
                    self.counters[dkey][indx]+=1




#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    parser = OptionParser(usage="%prog [options] xtc-files ...")
    parser.set_defaults(ndatagrams=-1, verbose=0, l1_offset=0)
    parser.add_option('-n', "--ndatagrams",  type='int')
    parser.add_option('-v', "--verbose", action='count')
    parser.add_option('-l', "--l1-offset", type='float')

    (options, args) = parser.parse_args()

    if not args :
        parser.error("at least one file name required")

        logging.basicConfig(level=logging.INFO)

    # turn options into a dictionary (from optparser.Values)
    option_dict = vars(options)

    main = XtcScanner()
    main.setFiles(args)
    main.setOption(option_dict)
    sys.exit( main.scan() )

