import psana
import time
import logging
import event_process

class time_fiducials(event_process.event_process):
    def __init__(self):
        self.timestamp = ()
        self.src = psana.EventId
        self.logger = logging.getLogger(__name__+'.time_fiducials')

    def beginJob(self):
        self.parent.timestamp = self.timestamp
        self.parent.shared['timestamp'] = self.timestamp

    def event(self,evt):
        ts = evt.get(self.src)
        if ts is None:
            self.timestamp = ()
        else :
            self.timestamp = ts.time()

        self.parent.shared['timestamp'] = self.timestamp
        return

