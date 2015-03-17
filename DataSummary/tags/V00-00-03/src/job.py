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

__version__ = 0.2


class job(object):
    def __init__(self):
        self.subjobs = []
        self.ds = None
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.maxEventsPerNode = 5000
        self.all_times = []
        self.shared = {}
        self.output = []
        self.output_dir = None
        self._output_dir_base = os.path.expanduser('~/data-summary')
        self.gathered_output = []
        self.previous_versions = []
        self.x_axes = ['time',]
        self.logger = logging.getLogger(__name__+'.r{:0.0f}'.format( self.rank ))
        self.start_time = time.time()
        self.logger.info('start time is {:}'.format(self.start_time))
        self.eventN = 0
        self.count = 0
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
            self.logger.info('changing directory to {:} (outdir)'.format(outdir))
        self.logger_fh = logging.FileHandler('log_{:0.0f}.log'.format(self.rank))
        self.logger_fh.setLevel(logging.DEBUG)
        self.logger_fh.setFormatter( logging.Formatter( '%(asctime)s - %(levelname)s - %(name)s - %(message)s' ) )
        self.logger.addHandler( self.logger_fh )
        self.logger.info( "output directory is "+self.output_dir )
        self.version_info()
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

    def set_maxEventsPerNode(self,n):
        self.maxEventsPerNode = n
        return

    def set_datasource(self,exp=None,run=None):
        self.exp = exp
        self.run = run
        instr, thisexp = exp.split('/')
        self.set_outputdir(os.path.join( self.baseoutputdir ,'{:}_run{:0.0f}'.format(thisexp,run)))

        self.logger.info('connecting to data source')
        self.ds = psana.DataSource('exp={:}:run={:0.0f}:idx'.format(exp,run))
        if self.ds.empty():
            self.logger.error('data source is EMPTY!')
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
        subjob_data = [sj.describe_self() for sj in self.subjobs]
        self.logger.info('subjobs at end: {:}'.format(subjob_data))

        self.gathered_subjobs = self.comm.gather( pup.pack(subjob_data) , root=0 )
        if self.rank == 0:
            tftable = [self.gathered_subjobs[0] == gsj for gsj in self.gathered_subjobs]
            allgood =  not False in tftable
            self.logger.info('all ranks have identical subjobs: {:}'.format(allgood  ))
            self.logger.debug('gathered subjobs: \n {:}'.format( pprint.pformat( self.gathered_subjobs) ) )
            if not allgood:
                self.logger.error.info('subjobs matching to sj 0: {:}'.format(repr(tftable)))
        #    self.scattered_subjobs = self.check_subjobs( self.gathered_subjobs[0] )
        #else:
        #    self.scattered_subjobs = None
        #self.scattered_subjobs = self.comm.scatter(self.scattered_subjobs, root=0 )
        ## instantiate the necessary subjobs in the right order (or reorder them, or whatever)
        ## and update the list of subjobs such that they are identical across all ranks.
        #self.update_subjobs_before_endJob()
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

            for sj in self.subjobs:
                try:
                    sj.endRun()
                except Exception as e:
                    self.logger.error('some error at endRun step!! {:}'.format(e) )

        self.logger.info( "rank {:} finished event processing".format( self.rank ) )
        self.logger.info( "rank {:} total events processed: {:0.0f}".format( self.rank,self.count ) )
        self.cputotal = time.time() - self.cpustart

        # do a pre endJob check to make sure all jobs have the same subjobs (unfinished)
        self.subjobs[5:-1] = sorted(self.subjobs[5:-1]) # sort all jobs except first and last
        self.unify_ranks()
        for sj in self.subjobs:
            self.logger.info( 'rank {:} endJob {:}'.format(self.rank, repr(sj) ) )
            try:
                sj.endJob()
            except Exception as e:
                self.logger.error('some error at endJob step!! {:}'.format(e) )

        #logger_flush()
        self.logger.info('rank {:} done!'.format( self.rank ) )
        for hdlr in self.logger.__dict__['handlers']: # this is a bad way to do this...
            hdlr.flush()
        return
