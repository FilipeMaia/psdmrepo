from mpi4py import MPI
import logging
import os
import sys
import psana
import time
import math
import packunpack as pup
import hashlib
import pprint
import tarfile
import glob
import random
import traceback

__version__ = '00.00.06'


class job(object):
    def __init__(self):
        self.subjobs = []
        self.ds = None
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self._reducer_rank_selector = random.Random(1234) # this is the same amongst all ranks so they
                                                          # all choose the same reducer rank
        self._reducer_rank    = {}
        self.maxEventsPerNode = 5000 # arbitrarily chosen
        self.all_times        = []
        self.shared = {}
        self.output = []

        self.output_dir = None
        self._output_dir_base = os.path.expanduser('~/data-summary')

        self.gathered_output = []
        self.previous_versions = []

        self.x_axes = ['time',]
        self.eventN = 0
        self.count  = 0

        self.start_time = time.time()
        self.logger = logging.getLogger(__name__+'.r{:0.0f}'.format( self.rank ))
        self.logger.info('start time is {:}'.format(self.start_time))

        return

    @property
    def baseoutputdir(self):
        return self._output_dir_base

    @baseoutputdir.setter
    def baseoutputdir(self,value):
        self._output_dir_base = value
        return

    @baseoutputdir.getter
    def baseoutputdir(self):
        return self._output_dir_base


    def smart_rename(self,out):
        if out == './':
            return
        while out[-1] == '/':
            out = out[:-1]
        ii = 0
        while True:
            self.logger.info('checking if ({:}) exists ({:})'.format( out+'.{:02.0f}'.format(ii),os.path.isdir(out+'.{:02.0f}'.format(ii))))
            if os.path.isdir(out+'.{:02.0f}'.format(ii)):
                self.previous_versions.append( [out+'.{:02.0f}'.format(ii),] )
                if os.path.isdir(out+'.{:02.0f}'.format(ii)) and os.path.isfile( out+'.{:02.0f}'.format(ii) +'/report.html'):
                    self.previous_versions[-1].append( time.ctime( os.path.getctime(  out+'.{:02.0f}'.format(ii) +'/report.html' ) ) )
                else: 
                    self.previous_versions[-1].append( None )
                ii += 1
            else : 
                break
        self.output_dir_orig = self.output_dir
        self.output_dir = self.output_dir+'.{:02.0f}'.format(ii)
        self.logger.info('setting output dir to {:}'.format(self.output_dir))
        os.makedirs(self.output_dir)
        if os.path.exists( self.output_dir_orig + '.latest' ):
            os.unlink( self.output_dir_orig + '.latest' ) # remove the previous symlink
        os.symlink( self.output_dir, self.output_dir_orig + '.latest') # make a new symlink
        return
                     
    def set_outputdir(self,*args):
        if len(args) == 1:
            self.output_dir = args[0]
        if self.rank==0:
            self.smart_rename(self.output_dir)
        self.logger.info('waiting for rank 0 to set up directories..')
        outdir = self.comm.bcast(self.output_dir,root=0) # block and wait for rank0 to finish the directory stuff
        self.logger.info('waiting for rank 0 to set up directories..done')
        if outdir == self.output_dir:
            os.chdir(self.output_dir)
            self.logger.info('changing directory to {:} (output_dir)'.format(self.output_dir))
        else:
            os.chdir(outdir)
            self.output_dir = outdir
            self.logger.info('changing directory to {:} (outdir)'.format(outdir))
        self.logger_fh = logging.FileHandler('log_{:0.0f}.log'.format(self.rank))
        self.logger_fh.setLevel(logging.DEBUG)
        self.logger_fh.setFormatter( logging.Formatter( '%(asctime)s - %(levelname)s - %(name)s - %(message)s' ) )
        self.logger.addHandler( self.logger_fh )
        self.logger.info( "output directory is "+self.output_dir )
        self.version_info()
        if self.rank==0:
            self.package_python()
        return

    def version_info(self):
        # put a bunch of stuff in the log file about all the files
        thisdir = os.path.dirname( os.path.abspath( unicode( __file__,sys.getfilesystemencoding() ) ) )
        #self.logger.info('last modified {:}, file {:}'.format(time.ctime(os.path.getmtime(thisfile)),thisfile))

        def myfun(arg,dirname,fnames):
            self.logger.info('module version: {:}'.format(__version__) )
            for ff in sorted(fnames):
                if '.pyc' in ff:
                    continue
                self.logger.info('last modified {:}, file {:}'.format( 
                    time.ctime( os.path.getmtime( os.path.join( dirname, ff ) ) ) ,
                    os.path.join( dirname, ff ) 
                    ) )

        os.path.walk( thisdir, myfun, None)

        return

    def package_python(self):
        self.logger.info("opening tar file for datasummary code")
        thisdir = os.path.dirname( os.path.abspath( unicode( __file__,sys.getfilesystemencoding() ) ) )
        tar = tarfile.open(name="datasummary.tgz",mode="w:gz",dereference=True)
        for f in glob.glob(os.path.join(thisdir,"*.py")):
            self.logger.info("taring file: {:}".format(f))
            tar.add(f)
        self.logger.info("closing tar file")
        tar.close()
        return

    def set_maxEventsPerNode(self,n):
        self.maxEventsPerNode = n
        return

    def set_datasource(self,exp=None,run=None,srcdir=None):
        self.exp = exp
        self.run = run
        self.srcdir = srcdir
        instr, thisexp = exp.split('/')
        self.set_outputdir(os.path.join( self.baseoutputdir ,'{:}_run{:0.0f}'.format(thisexp,run)))

        self.logger.info('connecting to data source')
        if srcdir is not None :
            self.ds = psana.DataSource('exp={:}:run={:0.0f}:dir={:}:idx'.format(exp,run,srcdir))
        else :
            self.ds = psana.DataSource('exp={:}:run={:0.0f}:idx'.format(exp,run))
        if self.ds.empty():
            self.logger.error('data source is EMPTY!')
        if srcdir is not None :
            self.logger.info('preparing to analyze {:} run {:} dir {:}'.format(exp,run,srcdir))
        else :
            self.logger.info('preparing to analyze {:} run {:}'.format(exp,run))
        return

    def set_x_axes(self,xaxes):
        self.x_axes = xaxes
        return

    def add_event_process(self,sj):
        sj.set_parent(self)
        
        sj.logger = logging.getLogger( self.logger.name + '.' + sj.logger.name.split('.')[-1] )

        #sj.logger.addHandler(self.logger_fh)

        self.subjobs.append(sj)

    def check_subjobs(self, gsj):
        data = {}
        for ii in xrange(self.comm.size) :
            data[ii] = set(tuple(gsj[ii]))
        # make the unified list of jobs, that all the ranks should replicate.
        self.unified = None
        if len(data) > 1:
            self.unified = data[0].union( data[1:] )
        return pup.unpack(self.unified)

    def update_subjobs_before_endJob(self):
        print self.rank, self.scattered_subjobs
        # reorder subjobs if necessary
        # add subjobs if necessary so all can reduce
        return

    def unify_ranks(self):

        # make a dictionary of the id:sj
        self.mymap = {(sj.__class__.__name__, repr(sj.describe_self()) ):sj for sj in self.subjobs}

        # get all the other ranks subjobs identifiers to all the ranks
        self.reduced_sj_name_content_pairs = self.comm.allgather(self.mymap.keys())

        if self.rank == 0:
            self.logger.info( "the data processors that each rank has" )
            self.logger.info( pprint.pformat( self.reduced_sj_name_content_pairs, width=120 )) # for convenience

        self.dmap = {} # dmap contains the mapping id:list_of_ranks_with_this_id

        for ii,rj in enumerate(self.reduced_sj_name_content_pairs) :
            for nn in rj:
                if nn not in self.dmap:
                    self.dmap[nn] = []
                self.dmap[nn].append(ii) # fill up dmap

        # everyone has this information, so everyone knows what needs to be done
        if self.rank == 0:
            self.logger.info( "data reduction org chart" )

        for nn in sorted(self.dmap):   # a distributed reduction approach
            # don't always have rank 0 do the reduction, so this is stressed
            self._reducer_rank[nn] = self._reducer_rank_selector.choice(self.dmap[nn]) 
            if self.rank == 0:
                self.logger.info( "  {:} should reduce {:15s} from {:}".format(self._reducer_rank[nn], nn, self.dmap[nn] ) )
        return

    def gather_output(self):
        gathered_output = self.comm.gather( self.output, root=0 )
        timing = self.comm.gather(self.cputotal, root=0)
        if self.rank == 0:
            self.cpu_time = sum( timing )
            if len(self.gathered_output) != 0:
                return
            for go in gathered_output:
                self.gathered_output.extend( go )
        return

    def dojob(self): # the main loop
        self.cpustart = time.time()
        self.logger.info( "rank {:} starting".format( self.rank ) )
        # assign reducer_rank for the subjobs (just cycle through the available nodes)
        ranks = range(self.size)
        for ii,sj in enumerate(self.subjobs):
            #sj.reducer_rank = ranks[ ii % len(ranks) ]
            sj.reducer_rank = 0


        for sj in self.subjobs:
            try:
                sj.beginJob()
            except Exception as e:
                self.logger.error('some error at beginJob step!! {:}'.format(e) )
                self.logger.error(traceback.format_exc())

        for self.thisrun in self.ds.runs():
            times = self.thisrun.times()
            if self.rank == 0:
                self.all_times.extend(times)
            mylength = int(math.ceil(float(len(times))/self.size))
            if mylength > self.maxEventsPerNode:
                mylength = self.maxEventsPerNode
            self.mytimes = times[self.rank*mylength:(self.rank+1)*mylength]

            if mylength > len(self.mytimes):
                mylength = len(self.mytimes)

            for sj in self.subjobs:
                try: 
                    sj.beginRun()
                except Exception as e:
                    self.logger.error('some error at beginRun step!! {:}'.format(e) )
                    self.logger.error(traceback.format_exc())
                
            for ii in xrange(mylength):
                self.count += 1
                self.evt = self.thisrun.event(self.mytimes[ii])
                self.eventN = ii
                if self.evt is None:
                    self.logger.ERROR( "**** event fetch failed ({:}) : rank {:}".format(self.mytimes[ii],self.rank) )
                    continue

                for sj in self.subjobs:
                    try:
                        sj.event(self.evt)
                    except Exception as e:
                        self.logger.error('some error at event step!! {:}'.format(e) )
                        self.logger.error(traceback.format_exc())

            for sj in self.subjobs:
                try:
                    sj.endRun()
                except Exception as e:
                    self.logger.error('some error at endRun step!! {:}'.format(e) )
                    self.logger.error(traceback.format_exc())

        self.logger.info( "rank {:} finished event processing".format( self.rank ) )
        self.logger.info( "rank {:} total events processed: {:0.0f}".format( self.rank,self.count ) )
        self.cputotal = time.time() - self.cpustart

        # do a pre endJob check to make sure all jobs have the same subjobs (unfinished)
        self.subjobs[5:-1] = sorted(self.subjobs[5:-1]) # sort all jobs except first and last
        self.unify_ranks()

        for sj in self.subjobs:
            self.logger.info( 'rank {:} endJob {:}'.format(self.rank, repr(sj) ) )
            self.logger.info( '   --- rank {:} should send to {:} ({:})'.format(self.rank, self._reducer_rank[(sj.__class__.__name__, repr( sj.describe_self() ))], sj.__class__.__name__ ) )
            if sj != self.subjobs[-1]:
                sj.reducer_rank = self._reducer_rank[(sj.__class__.__name__, repr( sj.describe_self() ))]
                sj.reduce_ranks = self.dmap[(sj.__class__.__name__, repr( sj.describe_self() ))]
            try:
                sj.endJob()
            except Exception as e:
                self.logger.error('{:} some error at endJob step!! {:}'.format(sj.__class__.__name__,e) )
                self.logger.error(traceback.format_exc())

        #logger_flush()
        self.logger.info('rank {:} done!'.format( self.rank ) )
        for hdlr in self.logger.__dict__['handlers']: # this is a bad way to do this...
            hdlr.flush()
        return
