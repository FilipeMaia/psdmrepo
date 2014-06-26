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
        # A good run for this is xcs84213 run 4
        events = DataSource(expname).events()
        input_srcs = []

        # if the camera name is in *args publish images for it
        if 'yag5' in args:
            input_srcs.append((Source('DetInfo(XcsBeamline.1:Tm6740.5)'), Camera.FrameV1, Camera.FrameV1.data16, 'yag5'))
        if 'xcs-spectrometer' in args:
            input_srcs.append((Source('DetInfo(XcsEndstation.1:Opal1000.1)'), Camera.FrameV1, Camera.FrameV1.data16, 'xcs-spectrometer'))

        # check what the multi-image topic name should be
        multi_image_topic = kwargs.get('multi-image-name', 'xcs-multi-image')

        for evt in events:
            evt_data = evt.get(EventId)
            evt_ts = evt_data.time()
            # convert the ts
            evt_ts_str = '%.4f'%(evt_ts[0] + evt_ts[1] / 1e9)

            img_index = 0

            multi_image_data = MultiData(evt_ts_str, multi_image_topic)

            for src, data_type, data_func, topic in input_srcs:
                frame = evt.get(data_type, src)
                image_data = ImageData(evt_ts_str, topic, data_func(frame))
                multi_image_data.add(image_data)
                self.send_data(topic, image_data)
                img_index += 1
            if len(input_srcs) > 0:
                self.send_data(multi_image_topic, multi_image_data)
            counter += 1
            time.sleep(myrate)
