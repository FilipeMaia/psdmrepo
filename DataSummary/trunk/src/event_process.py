import logging
import os

__version__ = '00.00.06'

# copied from psana/Module.h
# https://www.slac.stanford.edu/~gapon/TALKS/2013_Oct_LCLS_UserMeeting/Batch_Psana_Intro_v2.pdf pg 7
class event_process(object):
    """
    An empty event process in the new style.
    See some working examples in event_process_lib.py
    They all inherit form this.
    """
    def __init__(self):
        """
        this will need to be overriden in a child class
        """
        self.output       = event_process_output()
        self.reducer_rank = 0
        self.reduce_ranks = []
        self.logger       = logging.getLogger(__name__+'.default_logger')
        return

    def set_parent(self,parent):
        """
        stores the parent name space for easy access by the subjob
        """
        self.parent = parent

    @property
    def output_dir(self):
        """
        set a unique output directory name (based on event process id and node rank)
        """
        if not hasattr(self,'_output_dir_id'):
            self._output_dir_id = repr(id(self)) + '-{:0.0f}'.format(self.parent.comm.Get_rank())
        outdir = os.path.join(self.parent.output_dir, self._output_dir_id )
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        return outdir

    @output_dir.setter
    def output_dir(self,val):
        """
        set a specific output directory instead of using the automatic one
        """
        self._output_dir_id = repr(val)
        outdir = os.path.join(self.parent.output_dir, self._output_dir_id )
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        return 
        

    def beginJob(self):
        """
        put setup tasks here
        """
        return

    def beginRun(self):
        """
        put run specific setup tasks here
        """
        return

    def event(self, evt):
        """
        put the main even processing tasks here
        this gets run on every single even in the data set
        """
        return

    def endRun(self):
        """
        put end of run tasks here
        """
        return

    def endJob(self):
        """
        put end of job tasks here
        this is also an important task since it is performed after all events have
        been processed, and where final analysis should be performed.  it's the "reduce" step,
        and as such it needs to be programmed parallel-y
        """
        return

    def replicate_info(self):
        """
        not used
        """
        return None

    def describe_self(self):
        """
        returns an object that can be used to create another identical (minus data)
        instance of the current object
        """
        # return a pickleable object (dictionary)
        # that can be used to reproduce this instance (minus the data)
        return (str(self.__class__).split('.')[-1].replace("'>",""), self.replicate_info() )

    def reduce(self,ranks,root=None,tag=None):
        """
        example of the philosophy necessary to carefully collect the data from
        only the ranks the encountered the data.  this isn't encountered often
        but it is possible.  there is probrably a better way to do it with mpi groups
        """
        self.gathered =[]
        if root is None and tag is None:
            # do your own singular reduction
            self.gathered.append( self.vals ) # replace vals with something appropriate
        elif root == rank and tag is not None:
            # recieve from the other guys
            for r in ranks:
                if r == rank:
                    self.gathered.append( self.vals ) # replace vals with something appropriate
                else :
                    self.gathered.append( comm.recv( source=r, tag=tag) ) # replace vals with something appropriate
        elif root != rank and tag is not None:
            # send to the root
            comm.send( self.vals, dest=root, tag=tag ) # replace vals with something appropriate
        return

#    def beginStep(self):
#        return
#    def endStep(self):
#        return

class event_process_output(object):
    """
    structured holder for results of event process.
    """

    def __init__(self):
        self.in_report       = None
        self.in_report_title = None
        self.table           = {}
        self.figures         = {}
        self.text            = []
        return

    def __contains__(self,item):
        """
        """
        if item == 'in_report' or item == 'in_report_title':
            return True
        return self.__dict__[item] and len(self.__dict__[item])>0

    def __getitem__(self,key):
        """
        """
        return self.__dict__[key]

    def __setitem__(self,key,val):
        """
        """
        self.__dict__[key] = val
        return

    def __repr__(self):
        """
        """
        return repr(self.__dict__)
