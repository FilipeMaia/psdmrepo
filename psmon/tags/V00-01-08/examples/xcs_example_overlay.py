#!/usr/bin/env python
import time
import numpy
from psana import *
from psmon import publish
from psmon.plots import XYPlot


def main():
    exp = 'xcsi0314'
    run = '12'
    counter = 0
    status_rate = 100
    myrate = .2 # sleep time between data sends - crudely limit rate to < 5 Hz

    # create expname string for psana
    if run == 'online':
        expname='shmem=%s.0:stop=no'%exp
    else:
        expname='exp=%s:run=%s'%(exp, run)

    ipimb_srcs = [
        (Source('BldInfo(XCS-IPM-02)'), Lusi.IpmFexV1, Lusi.IpmFexV1.channel, 'xcs-ipm-02', [[], []], [[], []]),
        (Source('BldInfo(XCS-IPM-04)'), Lusi.IpmFexV1, Lusi.IpmFexV1.channel, 'xcs-ipm-04', [[], []], [[], []]),
    ]

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

        # clear accumulated ipm event data if flag is sent
        if publish.get_reset_flag():
            for _, _, _, _, xvals, yvals in ipimb_srcs:
                for xval, yval in zip(xvals, yvals):
                    xval[:] = []
                    yval[:] = []
            publish.clear_reset_flag()

        for src, data_type, data_func, topic, xvals, yvals in ipimb_srcs:
            ipm = evt.get(data_type, src)
            if ipm is None:
                continue
            xvals[0].append(data_func(ipm)[0])
            yvals[0].append(data_func(ipm)[2])
            xvals[1].append(data_func(ipm)[1])
            yvals[1].append(data_func(ipm)[3])
            ipm_data_over = XYPlot(
                evt_ts_str,
                topic+' chan 0 vs 2 and 1 vs 3',
                [numpy.array(xvals[0]), numpy.array(xvals[1])],
                [numpy.array(yvals[0]), numpy.array(yvals[1])],
                formats='.'
            )
            publish.send(topic+'-over', ipm_data_over)
        counter += 1
        if counter % status_rate == 0:
            print "Processed %d events so far" % counter
        time.sleep(myrate)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print '\nExitting script!'
