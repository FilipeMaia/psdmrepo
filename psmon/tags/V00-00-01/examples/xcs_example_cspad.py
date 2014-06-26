import time
from psana import *
from psmon.psutil import MultiData, ImageData
from psmon.psmonserver import ServerScript

class MyAnalysis(ServerScript):
    def __init__(self):
        super(MyAnalysis, self).__init__()

    def run(self, expname, *args, **kwargs):
        counter = 0
        myrate = .2 # sleep time between data sends - crudely limit rate to < 5 Hz
        # set the psana config file
        setConfigFile(os.path.join(os.path.dirname(__file__), 'cfg', 'xcs_cspad.cfg'))
        # A good run for this is xcs84213 run 115
        events = DataSource(expname).events()
        input_srcs = [
            (Source('DetInfo(XcsEndstation.0:Cspad.0)'), ndarray_int16_2, 'image0', 'cspad'),
        ]

        for evt in events:
            evt_data = evt.get(EventId)
            evt_ts = evt_data.time()

            # convert the ts
            evt_ts_str = '%.4f'%(evt_ts[0] + evt_ts[1] / 1e9)

            for src, data_type, data_key, topic in input_srcs:
                frame = evt.get(data_type, src, data_key)
                image_data = ImageData(evt_ts_str, topic, frame)
                self.send_data(topic, image_data)
            counter += 1
            time.sleep(myrate)
