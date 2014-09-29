#!/usr/bin/env python
import time
import numpy
from psana import *
from psmon import publish
from psmon.plots import MultiPlot, Image, XYPlot


def main():
    exp = 'xcsc0114'
    run = '9'
    counter = 0
    status_rate = 100
    myrate = .2 # sleep time between data sends - crudely limit rate to < 5 Hz

    # create expname string for psana
    if run == 'online':
        expname='shmem=%s.0:stop=no'%exp
    else:
        expname='exp=%s:run=%s'%(exp, run)

    input_srcs = [
        (Source('DetInfo(XcsBeamline.1:Tm6740.5)'), Camera.FrameV1, Camera.FrameV1.data16, 'yag5'),
        (Source('DetInfo(XcsEndstation.1:Opal1000.1)'), Camera.FrameV1, Camera.FrameV1.data16, 'xcs-spectrometer'),
    ]
    ipimb_srcs = [
        (Source('BldInfo(XCS-IPM-02)'), Lusi.IpmFexV1, Lusi.IpmFexV1.channel, 'xcs-ipm-02', [], []),
        #(Source('DetInfo(XcsBeamline.1:Ipimb.4)'), Lusi.IpmFexV1, Lusi.IpmFexV1.channel, 'xcs-ipm-04', [], []),
    ]
    multi_plot_topic = 'xcs-multi-plot'

    # initialize socket connections
    publish.init()

    # Start processing events
    if run == 'online':
        print "Running psana example script: shared-mem %s" % exp
    else:
        print "Running psana example script: experiment %s, run %s" % (exp, run)
    events = DataSource(expname).events()

    for evt in events:
        evt_data = evt.get(EventId)
        evt_ts = evt_data.time()
        # convert the ts
        evt_ts_str = '%.4f'%(evt_ts[0] + evt_ts[1] / 1e9)

        img_index = 0

        multi_plot_data = MultiPlot(evt_ts_str, multi_plot_topic)
        # flag for indicating if all needed pieces of the multi_data are there
        multi_data_good = True

        # clear accumulated ipm event data if flag is sent
        if publish.get_reset_flag():
            for _, _, _, _, xvals, yvals in ipimb_srcs:
                xvals[:] = []
                yvals[:] = []
            publish.clear_reset_flag()

        for src, data_type, data_func, topic in input_srcs:
            frame = evt.get(data_type, src)
            if frame is None:
                multi_data_good = False
                continue
            image_data = Image(evt_ts_str, topic, data_func(frame))
            multi_plot_data.add(image_data)
            publish.send(topic, image_data)
            img_index += 1
        for src, data_type, data_func, topic, xvals, yvals in ipimb_srcs:
            ipm = evt.get(data_type, src)
            if ipm is None:
                multi_data_good = False
                continue
            xvals.append(data_func(ipm)[0])
            yvals.append(data_func(ipm)[2])
            ipm_data0v2 = XYPlot(
                evt_ts_str,
                topic+' chan 0 vs 2',
                numpy.array(xvals),
                numpy.array(yvals),
                xlabel={'axis_title': 'chan 0', 'axis_units': 'V'},
                ylabel={'axis_title': 'chan 2', 'axis_units': 'V'},
                formats='.'
            )
            multi_plot_data.add(ipm_data0v2)
            publish.send(topic+'-0v2', ipm_data0v2)
        if multi_data_good:
            publish.send(multi_plot_topic, multi_plot_data)
        counter += 1
        if counter % status_rate == 0:
            print "Processed %d events so far" % counter
        time.sleep(myrate)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print '\nExitting script!'
