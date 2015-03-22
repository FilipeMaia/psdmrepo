#!/usr/bin/env python
import sys
import zmq
import logging
import argparse

from psmon import app, config, log_level_parse


LOG = logging.getLogger(config.LOG_BASE_NAME)


def _print_wrapper(func):
    def wrapped_func(*args, **kwargs):
        print func(*args, **kwargs)
    return wrapped_func


def _parse_cmdline():
    parser = argparse.ArgumentParser(
        description='Psmon console client for communicating with servers'
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
        default=config.APP_PORT+config.APP_COMM_OFFSET,
        help='the tcp port of the server (default: %d)'%(config.APP_PORT+config.APP_COMM_OFFSET)
    )

    parser.add_argument(
        '--log',
        metavar='LOG',
        default=config.LOG_LEVEL,
        help='the logging level of the client (default %s)'%config.LOG_LEVEL
    )

    return parser.parse_args()


def _main(context):
    args = _parse_cmdline()

    # set levels for loggers that we care about
    LOG.setLevel(log_level_parse(args.log))

    # start zmq requester
    LOG.info('Starting request client for host \'%s\' on port \'%d\'', args.server, args.port)
    try:
        comm_socket = context.socket(zmq.REQ)
        comm_socket.connect('tcp://%s:%d' % (args.server, args.port))
        zmqreq = app.ZMQRequester(comm_socket)
        LOG.info('Request client started successfully')
        return zmqreq
    except zmq.ZMQError as err:
        LOG.error('Failed to connect to server: %s', err)


if __name__ == '__main__':
    _zmqcontext = zmq.Context()
    _requester = _main(_zmqcontext)
    request = _print_wrapper(_requester.send_request)
    reset = _print_wrapper(_requester.send_reset_signal)
    LOG.info('Available commands: \'request\', \'reset\'')
