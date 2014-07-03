#!/usr/bin/env python
import os
import sys
import time
import logging
import argparse
import multiprocessing as mp

from psmon import app, config


LOG = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


def parse_cmdline():
    parser = argparse.ArgumentParser(
        description='Psmon plot client application'
    )

    parser.add_argument(
        'topics',
        nargs='+',
        help='The topic from the server to suscribe too'
    )

    parser.add_argument(
        '-s',
        '--server',
        metavar='SERVER',
        default=config.APP_SERVER,
        help='the host name of the server (default: %s)'%config.APP_SERVER
    )

    parser.add_argument(
        '-p',
        '--port',
        metavar='PORT',
        type=int,
        default=config.APP_PORT,
        help='the tcp port of the server (default: %d)'%config.APP_PORT
    )

    parser.add_argument(
        '-r',
        '--rate',
        metavar='RATE',
        type=float,
        default=config.APP_RATE,
        help='update rate of the histogram in Hz (default: %.2fHz)'%config.APP_RATE
    )

    parser.add_argument(
        '-b',
        '--buffer',
        metavar='BUFFER',
        type=int,
        default=config.APP_BUFFER,
        help='the size in messages of recieve buffer (default: %d)'%config.APP_BUFFER
    )

    parser.add_argument(
        '-x',
        '--x-range',
        metavar='X_RANGE',
        type=float,
        nargs=2,
        default=None,
        help='the fixed x range for any plots'
    )

    parser.add_argument(
        '-y',
        '--y-range',
        metavar='Y_RANGE',
        type=float,
        nargs=2,
        default=None,
        help='the fixed y range for any plots'
    )

    parser.add_argument(
        '-z',
        '--z-range',
        metavar='Z_RANGE',
        type=float,
        nargs=2,
        default=None,
        help='the fixed z range for any plots'
    )

    parser.add_argument(
        '-a',
        '--aspect',
        metavar='ASPECT',
        type=float,
        default=None,
        help='the aspect ratio for the plot'
    )

    parser.add_argument(
        '-i',
        '--interpolation',
        metavar='INTERPOLATION',
        default='none',
        help='the interpolation type for images (default: \'none\')'
    )

    parser.add_argument(
        '--bkg-color',
        metavar='BKG_COLOR',
        default=None,
        help='the background color for plots'
    )

    parser.add_argument(
        '--text-color',
        metavar='TEXT_COLOR',
        default=None,
        help='the text color for plots'
    )

    parser.add_argument(
        '--palette',
        metavar='PALETTE',
        default=None,
        help='the color palette to use for images'
    )

    parser.add_argument(
        '--client',
        metavar='CLIENT',
        default=config.APP_CLIENT,
        help='the client backend used for rendering (default: %s)'%config.APP_CLIENT
    )

    parser.add_argument(
        '--log',
        metavar='LOG',
        default=config.LOG_LEVEL,
        help='the logging level of the client (default %s)'%config.LOG_LEVEL
    )

    return parser.parse_args()


def mpl_client(renderer, client_info, plot_info):
    render_mod = __import__('psmon.client%s'%renderer, fromlist=['main'])

    render_mod.main(client_info, plot_info)

def main():
    try:
        args = parse_cmdline()

        # set levels for loggers that we care about
        LOG.setLevel(app.log_level_parse(args.log))

        # create the plot info object from cmd args
        plot_info = app.PlotInfo(
            xrange=args.x_range,
            yrange=args.y_range,
            zrange=args.z_range,
            aspect=args.aspect,
            bkg_col=args.bkg_color,
            fore_col=args.text_color,
            interpol=args.interpolation,
            palette=args.palette
        )

        proc_list = []
        for topic in args.topics:
            client_info = app.ClientInfo(args.server, args.port, args.buffer, args.rate, topic)
            proc = mp.Process(name='%s-client'%topic, target=mpl_client, args=(args.client, client_info, plot_info))
            proc.daemon = True
            LOG.info('Starting client for topic: %s', topic)
            proc.start()
            proc_list.append(proc)
    
        # wait for all the children to exit
        failed_client = False
        for proc in proc_list:
            proc.join()
            if proc.exitcode == 0:
                LOG.info('%s exited successfully', proc.name)
            else:
                failed_client = True
                LOG.error('%s exited with non-zero status code: %d', proc.name, proc.exitcode)

        LOG.info('All clients have exited')

        # return a non-zero status code if any clients died
        if failed_client:
            return 1
    except KeyboardInterrupt:
        print '\nExitting client!'


if __name__ == '__main__':
    sys.exit(main())
