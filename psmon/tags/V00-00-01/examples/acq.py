from psana import *

from psmon import pszmq
import sys
from psmon.psdata import XYPlotData

ds = DataSource('exp=CXI/cxitut13:run=22')

acqsrc  =Source('DetInfo(CxiEndstation.0:Acqiris.0)')

pszmq.socket_init(12323, 12324, 10)

nbad = 0
ngood = 0
for evt in ds.events():
    if (nbad+ngood)%10 == 0:
        print 'nbad',nbad,'ngood',ngood
    acq   = evt.get(Acqiris.DataDescV1,acqsrc)
    if acq == None:
        nbad+=1
        continue
    ngood+=1

    acqnumchannels = acq.data_shape()
    chan=0
    # the final "0" here is acq segment.  LCLS has always only used segment 0.
    wf = acq.data(chan).waveforms()[0]

    if (nbad+ngood)%10 == 0:
        ax = range(0,len(wf))
        acqSingleTrace = XYPlotData(ngood, "ACQIRIS SINGLE TRACE", ax, wf)
        pszmq.send_data("ACQSINGLETRACE", acqSingleTrace)
