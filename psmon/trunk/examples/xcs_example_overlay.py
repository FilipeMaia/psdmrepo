import time
import numpy
from psana import *
from psmon.psutil import XYPlotData
from psmon.psmonserver import ServerScript

class MyAnalysis(ServerScript):
    def __init__(self):
        super(MyAnalysis, self).__init__()

    def run(self, expname, *args, **kwargs):
        counter = 0
        myrate = .2 # sleep time between data sends - crudely limit rate to < 5 Hz
        # A good run for this is xcs84213 run 4
        events = DataSource(expname).events()
        ipimb_srcs = [
            (Source('BldInfo(XCS-IPM-02)'), Lusi.IpmFexV1, Lusi.IpmFexV1.channel, 'xcs-ipm-02', [[], []], [[], []]),
            (Source('DetInfo(XcsBeamline.1:Ipimb.4)'), Lusi.IpmFexV1, Lusi.IpmFexV1.channel, 'xcs-ipm-04', [[], []], [[], []]),
        ]

        for evt in events:
            evt_data = evt.get(EventId)
            evt_ts = evt_data.time()
            # convert the ts
            evt_ts_str = '%.4f'%(evt_ts[0] + evt_ts[1] / 1e9)

            # clear accumulated ipm event data if flag is sent
            if self.get_reset_flag():
                for _, _, _, _, xvals, yvals in ipimb_srcs:
                    for xval, yval in zip(xvals, yvals):
                        xval[:] = []
                        yval[:] = []
                self.clear_reset_flag()

            for src, data_type, data_func, topic, xvals, yvals in ipimb_srcs:
                ipm = evt.get(data_type, src)
                xvals[0].append(data_func(ipm)[0])
                yvals[0].append(data_func(ipm)[2])
                xvals[1].append(data_func(ipm)[1])
                yvals[1].append(data_func(ipm)[3])
                ipm_data_over = XYPlotData(evt_ts_str,
                                           topic+' chan 0 vs 2 and 1 vs 3',
                                           [numpy.array(xvals[0]), numpy.array(xvals[1])],
                                           [numpy.array(yvals[0]), numpy.array(yvals[1])],
                                           formats='.')
                self.send_data(topic+'-over', ipm_data_over)
            counter += 1
            time.sleep(myrate)
