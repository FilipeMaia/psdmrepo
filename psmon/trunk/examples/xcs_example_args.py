#!/usr/bin/env python
import re
import time
import argparse
from psana import *
from psmon import publish, config
from psmon.plots import MultiPlot, Image


def parse_args(*args):
    keyval_re = re.compile('\s*=+\s*')

    pos_args = []
    keyval_args = {}

    for arg in args:
        tokens = keyval_re.split(arg)

        if len(tokens) > 2:
            LOG.warning('Invalid input argument format: %s', arg)
            continue

        try:
            key, value = tokens
            keyval_args[key] = value
        except ValueError:
            pos_args.append(arg)

    return pos_args, keyval_args


def parse_cmdline():
    default_exp = 'xcsc0114'
    default_run = '9'

    parser = argparse.ArgumentParser(
        description='Psana example script using cli arguments'
    )

    parser.add_argument(
        'script_args',
        metavar='SCRIPT_ARGS',
        nargs='*',
        help='A list of arguments to pass to the script. Single values and key value pairs '\
             ' of the form <key>=<value>. Passed to the run function of the script as *arg '\
             'and **kwarg respectively. Example: foo bar baz=value'
    )

    parser.add_argument(
        '-p',
        '--port',
        metavar='PORT',
        type=int,
        default=config.APP_PORT,
        help='the tcp port the server listens on (default: %d)'%config.APP_PORT
    )

    parser.add_argument(
        '-b',
        '--buffer',
        metavar='BUFFER',
        type=int,
        default=config.APP_BUFFER,
        help='the size in messages of send buffer (default: %d)'%config.APP_BUFFER
    )

    parser.add_argument(
        '-e',
        '--expname',
        metavar='EXPNAME',
        default=default_exp,
        help='the experiment name or shmem server (default: %s)'%default_exp
    )

    parser.add_argument(
        '-r',
        '--run',
        metavar='RUN',
        default=default_run,
        help='the run number (default: %s)'%default_run
    )

    parser.add_argument(
        '--rate',
        metavar='RATE',
        type=float,
        default=0.2,
        help='the publish interval in seconds (default: %f)'%0.2
    )

    parser.add_argument(
        '--status',
        metavar='STATUS',
        type=int,
        default=1000,
        help='the number of events between status printouts (default: %d)'%1000
    )

    return parser.parse_args()


def main():
    # get the cli arguements
    cli_args = parse_cmdline()
    # create expname str for psana
    if cli_args.run == 'online':
        expname='shmem=%s.0:stop=no'%cli_args.expname
    else:
        expname='exp=%s:run=%s'%(cli_args.expname, cli_args.run)
    # parse the extra cli arguments
    script_args, script_kwargs = parse_args(*cli_args.script_args)
    # printout status rate
    status_rate = cli_args.status

    counter = 0
    input_srcs = []

    # if the camera name is in *args publish images for it
    if 'yag5' in script_args:
        input_srcs.append((Source('DetInfo(XcsBeamline.1:Tm6740.5)'), Camera.FrameV1, Camera.FrameV1.data16, 'yag5'))
    if 'xcs-spectrometer' in script_args:
        input_srcs.append((Source('DetInfo(XcsEndstation.1:Opal1000.1)'), Camera.FrameV1, Camera.FrameV1.data16, 'xcs-spectrometer'))

    # check what the multi-image topic name should be
    multi_image_topic = script_kwargs.get('multi-image-name', 'xcs-multi-image')

    # initialize socket connections
    publish.init(port=cli_args.port, bufsize=cli_args.buffer)

    # Start processing events
    if cli_args.run == 'online':
        print "Running psana example script: shared-mem %s" % cli_args.expname
    else:
        print "Running psana example script: experiment %s, run %s" % (cli_args.expname, cli_args.run)
    events = DataSource(expname).events()

    for evt in events:
        evt_data = evt.get(EventId)
        evt_ts = evt_data.time()
        # convert the ts
        evt_ts_str = '%.4f'%(evt_ts[0] + evt_ts[1] / 1e9)

        img_index = 0

        multi_image_data = MultiPlot(evt_ts_str, multi_image_topic)
        # flag for indicating if all needed pieces of the multi_data are there
        multi_data_good = True

        for src, data_type, data_func, topic in input_srcs:
            frame = evt.get(data_type, src)
            if frame is None:
                multi_data_good = False
                continue
            image_data = Image(evt_ts_str, topic, data_func(frame))
            multi_image_data.add(image_data)
            publish.send(topic, image_data)
            img_index += 1
        if len(input_srcs) > 1 and multi_data_good:
            publish.send(multi_image_topic, multi_image_data)
        counter += 1
        if counter % status_rate == 0:
            print "Processed %d events so far" % counter
        time.sleep(cli_args.rate)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print '\nExitting script!'
