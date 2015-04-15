import logging
import os


# copied from psana/Module.h
# https://www.slac.stanford.edu/~gapon/TALKS/2013_Oct_LCLS_UserMeeting/Batch_Psana_Intro_v2.pdf pg 7
class event_process(object):
    """
    An empty event process in the new style.
    See some working examples in event_process_lib.py
    They all inherit form this.
    """
    def __init__(self):
        self.output       = event_process_output()
        self.reducer_rank = 0
        self.reduce_ranks = []
        self.logger       = logging.getLogger(__name__+'.default_logger')
        return

    def set_parent(self,parent):
        self.parent = parent
        #if 'r{:}'.format(self.parent.rank) not in self.logger.name:
            #self.logger.name = '{:}.r{:}'.format(self.logger.name,self.parent.rank)

    @property
    def output_dir(self):
        if not hasattr(self,'_output_dir_id'):
            self._output_dir_id = repr(id(self)) + '-{:0.0f}'.format(self.parent.comm.Get_rank())
        outdir = os.path.join(self.parent.output_dir, self._output_dir_id )
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        return outdir

    @output_dir.setter
    def output_dir(self,val):
        self._output_dir_id = repr(val)
        outdir = os.path.join(self.parent.output_dir, self._output_dir_id )
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        return 
        

    def beginJob(self):
        #print "rank {:}".format(self.parent.rank)
        return

    def beginRun(self):
        return

    def event(self, evt):
        return

    def endRun(self):
        return

    def endJob(self):
        return

    def replicate_info(self):
        return None

    def describe_self(self):
        # return a pickleable object (dictionary)
        # that can be used to reproduce this instance (minus the data)
        return (str(self.__class__).split('.')[-1].replace("'>",""), self.replicate_info() )

    def reduce(self,ranks,root=None,tag=None):
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
        if item == 'in_report' or item == 'in_report_title':
            return True
        return self.__dict__[item] and len(self.__dict__[item])>0

    def __getitem__(self,key):
        return self.__dict__[key]

    def __setitem__(self,key,val):
        self.__dict__[key] = val
        return

    def __repr__(self):
        return repr(self.__dict__)
