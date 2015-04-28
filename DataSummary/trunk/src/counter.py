import logging
import numpy
from mpi4py import MPI
import event_process

__version__ = '00.00.06'

class counter(event_process.event_process):
    def __init__(self ):
        self.data  = numpy.array([0,])
        self.mergeddata = numpy.array([0,]) 
        self.logger = logging.getLogger(__name__+'.counter')
        self.N = 100
        return

    def event(self,evt):
        #id = evt.get(psana.EventId)
        #print 'rank', pp.rank, 'analyzed event with fiducials', id.fiducials()
        self.data[0] += 1
        if self.data[0] % self.N == 0:
            self.logger.info('processed {:} events'.format(self.data[0]))
        return

    def endJob(self):
        self.parent.comm.Reduce([self.data, MPI.DOUBLE], [self.mergeddata, MPI.DOUBLE],op=MPI.SUM,root=0)
        self.logger.info( "rank {:} events processed: {:}".format(self.parent.rank,self.data[0]) )
        if self.parent.rank == 0:
            self.logger.info( "total events processed: {:}".format(self.mergeddata[0]) )
            self.parent.shared['total_processed'] = self.mergeddata[0]
        return

