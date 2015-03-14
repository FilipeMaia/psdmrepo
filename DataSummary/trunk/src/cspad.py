import os
import psana
import numpy
import logging
import event_process
import pylab
from mpi4py import MPI
from common import strtype

class cspad(event_process.event_process):
    def __init__(self):
        self.logger                      = logging.getLogger(__name__+'.cspad')
        self.output = event_process.event_process_output()
        self.output['in_report']         = None
        self.output['in_report_title']   = None
        self.frame                       = None
        self.nframes                     = numpy.array([0])
        self.reducer_rank                = 0
        return

    def beginJob(self):
        return

    def add_frame(self,frame):
        if self.frame == None:
            self.frame = numpy.zeros_like(frame,dtype='float64')

        self.frame += frame
        self.nframes[0] += 1
        return

    def set_stuff(self,psana_src,psana_device,in_report=None,in_report_title=None):
        self.src         = psana.Source(psana_src)
        self.dev         = psana_device
        self.output['in_report']         = in_report
        self.output['in_report_title']   = in_report_title

    def replicate_info(self):
        args = ( str(self.src), strtype(self.dev) )
        kwargs = { 'in_report': self.output['in_report'], 'in_report_title': self.output['in_report_title'] }
        self.logger.info('args: {:}'.format(repr(args)))
        self.logger.info('kwargs: {:}'.format(repr(kwargs)))
        return ('set_stuff',args,kwargs)

    def event(self,evt):
        cspad = evt.get(self.dev, self.src)
        a = []
        for i in range(0,4):
            quad = cspad.quads(i)
            d    = quad.data()
            a.append(numpy.vstack([ d[j] for j in range(0,8) ]))

        frame_raw = numpy.hstack(a)
        self.add_frame(frame_raw)
        return

    def endJob(self):
        self.logger.info('mpi reducing cspad')

        self.mergedframe = numpy.zeros_like( self.frame, dtype='float64' )
        self.mergednframes = numpy.array([0])
        self.parent.comm.Reduce(self.frame,   self.mergedframe,  op=MPI.SUM, root=self.reducer_rank)
        self.parent.comm.Reduce(self.nframes,self.mergednframes, op=MPI.SUM, root=self.reducer_rank)


        if self.parent.rank == self.reducer_rank:
            self.output['figures'] = {'mean': {}, 'mean_hist': {}, }
            fig = pylab.figure()
            self.avg = self.mergedframe/float(self.mergednframes[0])
            pylab.imshow(self.avg)
            pylab.colorbar()
            self.flat = self.avg.flatten()
            pylab.clim(self.flat.mean()-2.*self.flat.std(),self.flat.mean()+2.*self.flat.std())
            pylab.title('CSPAD average of {:} frames'.format(self.nframes))
            pylab.savefig( os.path.join( self.parent.output_dir, 'figure_cspad.png' ))
            self.output['figures']['mean']['png'] = os.path.join( self.parent.output_dir, 'figure_cspad.png')

            fig.clear()
            pylab.hist(self.flat,1000)
            pylab.xlim(self.flat.mean()-2.*self.flat.std(),self.flat.mean()+2.*self.flat.std())
            pylab.title('histogram')
            pylab.savefig( os.path.join( self.parent.output_dir, 'figure_cspad_hist.png' ))
            self.output['figures']['mean_hist']['png'] = os.path.join( self.parent.output_dir, 'figure_cspad_hist.png')

            del fig
            self.parent.output.append(self.output)
        return
