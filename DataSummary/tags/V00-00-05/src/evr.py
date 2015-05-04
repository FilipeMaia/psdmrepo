import psana
import logging
import event_process

class evr(event_process.event_process):
    def __init__(self):
        self.evr = []
        self.src = psana.Source('DetInfo(NoDetector.0:Evr.0)')
        self.logger = logging.getLogger(__name__+'.evr')
        return

    def beginJob(self):
        self.parent.evr = self.evr #maybe there is a better way? some sort of registry?
        self.parent.shared['evr'] = self.evr
        return

    def event(self,evt):
        self.raw_evr = evt.get(psana.EvrData.DataV3, self.src)
        if self.raw_evr is None:
            self.evr = []
        else:
            self.evr = [ff.eventCode() for ff in self.raw_evr.fifoEvents()]

        self.parent.shared['evr'] = self.evr
        return

