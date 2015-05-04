import logging

__version__ = '00.00.06'

class epics_scatter(event_process.event_process):
    def __init__(self):
        self.logger                      = logging.getLogger(__name__+'.epics_scatter')
        self.scatter_x_channel           = None
        self.scatter_y_channel           = None
        self.output = event_process.event_process_output()
        self.output['in_report']         = None
        self.output['in_report_title']   = None

        return

    def replicate_info(self):
        return None

    def beginJob(self):
        self.epics = self.parent.ds.env().epicsStore()
        self.allPvs = self.epics.names()
        self.scatters = {}
        return

    def add_pv_x(self,chan):
        self.scatter_x_channel = chan
        return

    def add_pv_y(self,chan):
        self.scatter_y_channel = chan
        return

    def event(self,evt):
        return

    def endJob(self):
        return

