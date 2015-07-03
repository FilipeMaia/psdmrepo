#!/usr/bin/env python
import time
from psana import *
from psmon import publish
from psmon.helper import ImageHelper, MultiImageHelper, XYPlotHelper, HistHelper, HistOverlayHelper


def main():
    exp = 'xcsc0114'
    run = '9'
    counter = 0
    status_rate = 100
    myrate = 1 # data publish rate in Hz

    # create expname string for psana
    if run == 'online':
        expname='shmem=%s.0:stop=no'%exp
    else:
        expname='exp=%s:run=%s'%(exp, run)

    input_srcs = [
        (Source('DetInfo(XcsBeamline.1:Tm6740.5)'), Camera.FrameV1, Camera.FrameV1.data16, ImageHelper(publish.send, 'yag5', pubrate=myrate)),
        (Source('DetInfo(XcsEndstation.1:Opal1000.1)'), Camera.FrameV1, Camera.FrameV1.data16, ImageHelper(publish.send, 'xcs-spectrometer', pubrate=myrate)),
    ]
    ipimb_srcs = [
        (
            Source('BldInfo(XCS-IPM-02)'),
            Lusi.IpmFexV1,
            Lusi.IpmFexV1.channel,
            XYPlotHelper(
                publish.send,
                'xcs-ipm-02-0v2',
                xlabel={'axis_title': 'chan 0', 'axis_units': 'V'},
                ylabel={'axis_title': 'chan 2', 'axis_units': 'V'},
                format='.',
                pubrate=myrate
            ),
            HistHelper(
                publish.send,
                'xcs-ipm-02-hist0',
                100,
                0.0,
                0.6,
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
                publish.send,
                'xcs-ipm-04-0v2',
                xlabel={'axis_title': 'chan 0', 'axis_units': 'V'},
                ylabel={'axis_title': 'chan 2', 'axis_units': 'V'},
                format='.',
                pubrate=myrate
            ),
            HistHelper(
                publish.send,
                'xcs-ipm-04-hist0',
                100,
                0.0,
                1.2,
                xlabel={'axis_title': 'chan 0', 'axis_units': 'V'},
                ylabel={'axis_title': 'shots'},
                pubrate=myrate
            ),
        ),
    ]
    multi_image_helper = MultiImageHelper(publish.send, 'xcs-multi-image', 2, pubrate=myrate)
    over_histo_helper = HistOverlayHelper(
        publish.send,
        'xcs-overlay-histo',
        xlabel={'axis_title': 'chan 0', 'axis_units': 'V'},
        ylabel={'axis_title': 'shots'},
        pubrate=myrate
    )
    # Create histograms in the overlay for each ipm
    while over_histo_helper.nhist < len(ipimb_srcs):
        over_histo_helper.make_hist(100, 0.0, 1.2)

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
        ipm_index = 0

        for src, data_type, data_func, helper in input_srcs:
            frame = evt.get(data_type, src)
            if frame is None:
                continue
            multi_image_helper.set_image(img_index, data_func(frame), evt_ts_str)
            helper.set_image(data_func(frame), evt_ts_str)
            helper.publish()
            img_index += 1
        for src, data_type, data_func, helper, hist_helper in ipimb_srcs:
            ipm = evt.get(data_type, src)
            if ipm is None:
                continue
            helper.add(data_func(ipm)[0], data_func(ipm)[2], evt_ts_str)
            helper.publish()
            hist_helper.add(data_func(ipm)[0], evt_ts_str)
            hist_helper.publish()
            over_histo_helper.add(ipm_index, data_func(ipm)[0], evt_ts_str)
            ipm_index += 1
        multi_image_helper.publish()
        over_histo_helper.publish()
        counter += 1
        if counter % status_rate == 0:
            print "Processed %d events so far" % counter
        time.sleep(myrate/10.0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print '\nExitting script!'
