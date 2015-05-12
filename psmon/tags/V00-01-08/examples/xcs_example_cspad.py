#!/usr/bin/env python
import time
from psana import *
from psmon import publish
from psmon.plots import MultiPlot, Image


def main():
    exp = 'xcsc9114'
    run = '7'
    counter = 0
    status_rate = 100
    myrate = .2 # sleep time between data sends - crudely limit rate to < 5 Hz
    # set the psana config file
    setConfigFile(os.path.join(os.path.dirname(__file__), 'cfg', 'xcs_cspad.cfg'))

    # create expname string for psana
    if run == 'online':
        expname='shmem=%s.0:stop=no'%exp
    else:
        expname='exp=%s:run=%s'%(exp, run)

    input_srcs = [
        (Source('DetInfo(XcsEndstation.0:Cspad.0)'), ndarray_int16_2, 'image0', 'cspad'),
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

        for src, data_type, data_key, topic in input_srcs:
            frame = evt.get(data_type, src, data_key)
            if frame is None:
                continue
            image_data = Image(evt_ts_str, topic, frame)
            publish.send(topic, image_data)
        counter += 1
        if counter % status_rate == 0:
            print "Processed %d events so far" % counter
        time.sleep(myrate)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print '\nExitting script!'
