#display plot with "psplot -s psanacs031 -p 12301 ACQSINGLETRACE"

from psana import *

from psmon import publish
from psmon.plots import XYPlot,Image
import sys

ds = DataSource('exp=cxitut13:run=22')

acqsrc  =Source('DetInfo(CxiEndstation.0:Acqiris.0)')
pulnixsrc  =Source('DetInfo(CxiDg4.0:Tm6740.0)')

#optional port, buffer-depth arguments.
publish.init()

nbad = 0
ngood = 0
for evt in ds.events():
    if (nbad+ngood)%10 == 0:
        print 'nbad',nbad,'ngood',ngood
    acq = evt.get(Acqiris.DataDescV1,acqsrc)
    pulnix = evt.get(Camera.FrameV1,pulnixsrc)
    if acq == None or pulnix is None:
        nbad+=1
        continue
    ngood+=1

    acqnumchannels = acq.data_shape()
    chan=0
    # the final "0" here is acq segment.  LCLS has always only used segment 0.
    wf = acq.data(chan).waveforms()[0]

    if (nbad+ngood)%10 == 0:
        ax = range(0,len(wf))
        acqSingleTrace = XYPlot(ngood, "ACQIRIS SINGLE TRACE", ax, wf)
        publish.send("ACQSINGLETRACE", acqSingleTrace)

        cam = Image(ngood, "Pulnix", pulnix.data16())
        publish.send("PULNIX", cam)
