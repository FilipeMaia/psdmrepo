import time
from psana import *
from psmon.psutil import ImageHelper, MultiImageHelper, XYPlotHelper, HistHelper
from psmon.psmonserver import ServerScript

class MyAnalysis(ServerScript):
    def __init__(self):
        super(MyAnalysis, self).__init__()

    def run(self, expname):
        counter = 0
        myrate = 1 # data publish rate in Hz
        # A good run for this is xcs84213 run 4
        events = DataSource(expname).events()
        input_srcs = [
            (Source('DetInfo(XcsBeamline.1:Tm6740.5)'), Camera.FrameV1, Camera.FrameV1.data16, ImageHelper(self.send_data, 'yag5', pubrate=myrate)),
            (Source('DetInfo(XcsEndstation.1:Opal1000.1)'), Camera.FrameV1, Camera.FrameV1.data16, ImageHelper(self.send_data, 'xcs-spectrometer', pubrate=myrate)),
        ]
        ipimb_srcs = [
            (
                Source('BldInfo(XCS-IPM-02)'),
                Lusi.IpmFexV1,
                Lusi.IpmFexV1.channel,
                XYPlotHelper(
                    self.send_data,
                    'xcs-ipm-02-0v2',
                    xlabel={'axis_title': 'chan 0', 'axis_units': 'V'},
                    ylabel={'axis_title': 'chan 2', 'axis_units': 'V'},
                    pubrate=myrate
                ),
                HistHelper(
                    self.send_data,
                    'xcs-ipm-02-hist0',
                    100,
                    0.0,
                    0.0005,
                    xlabel={'axis_title': 'chan 0', 'axis_units': 'V'},
                    ylabel={'axis_title': 'shots'},
                    pubrate=myrate
                ),
            ),
            (
                Source('DetInfo(XcsBeamline.1:Ipimb.4)'),
                Lusi.IpmFexV1,
                Lusi.IpmFexV1.channel,
                XYPlotHelper(
                    self.send_data,
                    'xcs-ipm-04-0v2',
                    xlabel={'axis_title': 'chan 0', 'axis_units': 'V'},
                    ylabel={'axis_title': 'chan 2', 'axis_units': 'V'},
                    pubrate=myrate
                ),
                HistHelper(
                    self.send_data,
                    'xcs-ipm-04-hist0',
                    100,
                    0.0,
                    0.0005,
                    xlabel={'axis_title': 'chan 0', 'axis_units': 'V'},
                    ylabel={'axis_title': 'shots'},
                    pubrate=myrate
                ),
            ),
        ]
        multi_image_helper = MultiImageHelper(self.send_data, 'xcs-multi-image', 2, pubrate=myrate)

        for evt in events:
            evt_data = evt.get(EventId)
            evt_ts = evt_data.time()
            # convert the ts
            evt_ts_str = '%.4f'%(evt_ts[0] + evt_ts[1] / 1e9)

            img_index = 0

            for src, data_type, data_func, helper in input_srcs:
                frame = evt.get(data_type, src)
                multi_image_helper.set_image(img_index, data_func(frame), evt_ts_str)
                helper.set_image(data_func(frame), evt_ts_str)
                helper.publish()
                img_index += 1
            for src, data_type, data_func, helper, hist_helper in ipimb_srcs:
                ipm = evt.get(data_type, src)
                helper.add(data_func(ipm)[0], data_func(ipm)[2], evt_ts_str)
                helper.publish()
                hist_helper.add(data_func(ipm)[0], evt_ts_str)
                hist_helper.publish()
            multi_image_helper.publish()
            counter += 1
            time.sleep(myrate/10.0)
