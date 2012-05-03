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

@id      $Id: XtcScanner 0 2011-01-24 11:45:00 ofte $
@version $Revision: $
@author  $Author: $ 
@data    $Date: $
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
    moreinfo = {}
    epicsPVs = []
    controls = []

    #----------------
    #  Constructor --
    #----------------
    def __init__(self,
                 filenames = [],
                 options={'nevents':-1, 'ncalibcycles':-1,'l1_offset': 0, 'verbose': 0, 'epics':False }
                 ):
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
        self._state = None
        self._counter = 0
        

    #-------------------
    #  Public methods --
    #-------------------

    def scan(self):
        """Scan each xtc datagram

        For each xtc datagram that is of type
        datagram.xtc, call _scan.
        """
        # reset:
        self.ncalib = 0
        self.nevents = []
        self.devices.clear()
        self.moreinfo.clear()
        self.counters.clear()
        self.epicsPVs = []
        self.controls = []

        print "Scanning...."
        
        if len(self.files)==0 :
            print "You need to select an xtc file"
            return

        self.fsize = 0
        for fname in self.files :
            self.fsize += os.path.getsize(fname)
            
        print "Start parsing files: \n", self.files
        start = time.clock()

        # read datagrams one by one
        

        # counters
        ndg = 0 # datagrams
        self.dgsize = 0 # size read so far

        xtciter = io.XtcMergeIterator( self.files, self.options['l1_offset'] )
        for dg in xtciter :
            ndg+=1
            self.dgsize += dg.xtc.sizeofPayload()


            self._state = dg.seq.service()

            if self._state == TransitionId.Configure :
                pass
            
            elif self._state == TransitionId.BeginCalibCycle:
                if self.options['ncalibcycles']>-1 :
                    if self.ncalib >= self.options['ncalibcycles'] : break 
                self.ncalib+=1
                self.nevents.append(0)

            elif self._state == TransitionId.L1Accept:
                if self.options['nevents']>-1 :
                    if sum(self.nevents) >= self.options['nevents'] : break 
                self.nevents[(self.ncalib-1)]+=1

            else :
                # ignore all other transitions
                continue

            # recursively dump Xtc info
            self._scan ( dg.xtc )


            # print progress bar
            frac = float(self.dgsize)/float(self.fsize)
            if (ndg%100) == 1:
                print "\r  %d datagrams read (%.0f%%)" % (ndg, 100*frac) ,
                sys.stdout.flush()
            if (ndg%10) == 0:
                print " . " ,
                sys.stdout.flush()

        elapsed = ( time.clock() - start )
        frac = float(self.dgsize)/float(self.fsize)
        print "\r  %d datagrams read (approx %.0f %% of run) in %f s " % (ndg, 100*frac, elapsed)

        self.printSummary(opt_epics=self.options['epics'])

    def addCountInfo(self):
        pass



    def printSummary(self, opt_bld=1, opt_det=1, opt_epics=False):

        print "-------------------------------------------------------------"
        print "XtcScanner information: "
        print "  - %d calibration cycles." % self.ncalib
        print "  - Events per calib cycle: \n  ", self.nevents
        print

        print "Information from ", len(self.controls), " control channels found:"
        for ctrl in self.controls :
            print ctrl
        print "Information from ", len(self.devices), " devices found"
        sortedkeys = sorted( self.devices.keys() )
        for d in sortedkeys :
            print "%35s: " % d,
            for info in self.moreinfo[d] :
                if info is not None:
                    print "(%s)\t" % info,
                else :
                    print "%s\t" % "   ",
            for i in range ( len(self.devices[d] ) ):
                print " %s (%d) " % ( self.devices[d][i],  self.counters[d][i] ),
            print

        if opt_epics :
            print "Epics PVs: ", len(self.epicsPVs)
            print self.epicsPVs


        print "XtcScanner is done!"
        print "-------------------------------------------------------------"


    def setFiles(self, filenames):
        self.files = filenames

    def showFiles(self):
        print self.files


    # set one or more options
    def setOption(self, anoption):
        self.options.update(anoption)
        print "What's here?"

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
            source = str(xtc.src).split('(')[1].strip(')')
            contents = str(xtc.contains)
            worthknowing = None

            dtype = type(xtc.src).__name__
            dname = ''
            if ( dtype=='BldInfo' ):
                #dname = str(xtc.src.type()).split('.')[1]
                dname = source
            elif ( dtype=='DetInfo' ):
                dtn = str(xtc.src.detector()).split('.')[1]
                dti = str(xtc.src.detId())
                dvn = str(xtc.src.device()).split('.')[1]
                dvi = str(xtc.src.devId());
                dname = dtn+"-"+dti+"|"+dvn+"-"+dvi
            elif ( dtype=='ProcInfo'):
                dname = source.split(",")[0]
            else:
                print dtype, xtc.src
            dkey = dtype + ':' + dname
            
            if self._state == TransitionId.Configure:
                if xtc.contains.id() == TypeId.Type.Id_ControlConfig :
                    data = xtc.payload()
                    for i in range(0,data.npvControls()):
                        pv_control = data.pvControl(i).name()
                        if pv_control not in self.controls :
                            self.controls.append( pv_control )

                elif xtc.contains.id() == TypeId.Type.Id_Epics :
                    try:
                        data = xtc.payload()
                        self.epicsPVs.append(data.sPvName)
                    except:
                        #print "An epics object with no payload (size %d)" % xtc.sizeofPayload()
                        pass
                    
                elif xtc.contains.id() == TypeId.Type.Id_AcqConfig :
                    data = xtc.payload()
                    worthknowing = "%s ch" % data.nbrChannels()

                elif xtc.contains.id() == TypeId.Type.Id_TM6740Config :
                    data = xtc.payload()
                    worthknowing = "%dx%d" % (data.Row_Pixels,data.Column_Pixels)

                elif xtc.contains.id() == TypeId.Type.Id_Opal1kConfig :
                    data = xtc.payload()
                    worthknowing = "%dx%d" % (data.Row_Pixels,data.Column_Pixels)
                    #no? worthknowing = "%dx%d" % (data.row,data.column)

                elif xtc.contains.id() == TypeId.Type.Id_FccdConfig :
                    data = xtc.payload()
                    worthknowing = "%dx%d" % (data.trimmedHeight(),data.trimmedWidth())
                    
                elif xtc.contains.id() == TypeId.Type.Id_PrincetonConfig :
                    data = xtc.payload()
                    worthknowing = "%dx%d" % (data.height(),data.width())

                elif xtc.contains.id() == TypeId.Type.Id_pnCCDconfig :
                    data = xtc.payload()
                    worthknowing = "%dx%d" % (data.numRows(),data.numChannels())


            if self._state == TransitionId.BeginCalibCycle:
                """ Look for run control to check nevents of a calibcycle
                """
                if xtc.contains.id() == TypeId.Type.Id_ControlConfig:
                    data = xtc.payload()
                    if data.uses_duration():
                        worthknowing = "Calibcycle Duration = %s "% str(data.duration())
                        print worthknowing
                    if data.uses_events():
                        worthknowing = "Each calibcycle a %d events"% data.events()
                        

            if dkey not in self.devices :
                # first occurence of this detector/device
                self.devices[dkey] = []
                self.devices[dkey].append( contents )
                self.moreinfo[dkey] = []
                self.moreinfo[dkey].append(worthknowing) 
                self.counters[dkey] = []
                self.counters[dkey].append( 1 ) 
            else :
                # new type of data contents
                if self.devices[dkey].count( contents )==0 :
                    self.devices[dkey].append( contents )
                    if worthknowing is not None:
                        self.moreinfo[dkey].append(worthknowing) 
                    self.counters[dkey].append( 1 )
                else :
                    indx = self.devices[dkey].index( contents )
                    self.counters[dkey][indx]+=1




#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    parser = OptionParser(usage="%prog [options] xtc-files ...")
    parser.set_defaults(nevents=-1, verbose=0, l1_offset=0, epics=False)
    parser.add_option('-n', "--nevents",  type='int')
    parser.add_option('-c', "--ndcalibcycles",  type='int')
    parser.add_option('-v', "--verbose", action='count')
    parser.add_option('-l', "--l1-offset", type='float')
    parser.add_option('-e', "--epics", action='store_true')
    
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

