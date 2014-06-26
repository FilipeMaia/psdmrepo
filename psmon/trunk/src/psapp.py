import os
import re
import sys
import zmq
import pwd
import socket
import logging

from psmon import psconfig


LOG = logging.getLogger(__name__)


def socket_recv(sock):
    topic = sock.recv()
    return sock.recv_pyobj()


def get_socket_gen(sock):
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    while True:
        socks = dict(poller.poll(25))
        if socks.get(sock) == zmq.POLLIN:
            yield socket_recv(sock)
        else:
            yield


def reset_signal(sock, pending_flag):
    # check to see if there is another pending reset req
    if not pending_flag.is_set():
        pending_flag.set()
        sock.send(psconfig.RESET_REQ_STR%socket.gethostname())
        reply = sock.recv()
        if reply != psconfig.RESET_REP_STR%socket.gethostname():
            LOG.error(reply)
        pending_flag.clear()


def init_client_sockets(client_info):
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.SUBSCRIBE, client_info.topic + psconfig.ZMQ_TOPIC_DELIM_CHAR)
    sub_socket.set_hwm(client_info.buffer)
    sub_socket.connect("tcp://%s:%d" % (client_info.server, client_info.port))
    reset_socket = context.socket(zmq.REQ)
    reset_socket.connect("tcp://%s:%d" % (client_info.server, client_info.port+1))

    return context, sub_socket, reset_socket


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


def log_init():
    logging.basicConfig(format=psconfig.LOG_FORMAT, level=psconfig.LOG_LEVEL_ROOT)


def log_level_parse(log_level):
    return getattr(logging, log_level.upper(), psconfig.LOG_LEVEL_ROOT)


def default_run_chooser():
    # get offline defaults
    exp = psconfig.APP_EXP_DEFAULT
    run = psconfig.APP_RUN_DEFAULT

    # check if the current user is one of the opr accounts - if so default to online
    match_obj = re.search('^(.*)opr$', pwd.getpwuid(os.getuid()).pw_name)
    if match_obj:
        exp = match_obj.group(1)
        run = 'online'
    return exp, run


class ClientInfo(object):
    def __init__(self, server, port, buffer, rate, topic):
        self.server = server
        self.port = port
        self.buffer = buffer
        self.rate = rate
        self.topic = topic


class PlotInfo(object):
    def __init__(self, xrange=None, yrange=None, zrange=None, aspect=None, fore_col=None, bkg_col=None, interpol=None, palette=None):
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange
        self.aspect = aspect
        self.fore_col = fore_col
        self.bkg_col = bkg_col
        self.interpol = interpol
        self.palette = palette
