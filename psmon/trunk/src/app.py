import os
import re
import sys
import zmq
import pwd
import socket
import logging
import threading

from psmon import config


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
        sock.send(config.RESET_REQ_STR%socket.gethostname())
        reply = sock.recv()
        if reply != config.RESET_REP_STR%socket.gethostname():
            LOG.error(reply)
        pending_flag.clear()


def init_client_sockets(client_info):
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.SUBSCRIBE, client_info.topic + config.ZMQ_TOPIC_DELIM_CHAR)
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


def log_level_parse(log_level):
    return getattr(logging, log_level.upper(), config.LOG_LEVEL_ROOT)


def default_run_chooser():
    # get offline defaults
    exp = config.APP_EXP_DEFAULT
    run = config.APP_RUN_DEFAULT

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


class ZMQPublisher(object):
    def __init__(self, comm_offset):
        self.context = zmq.Context()
        self.data_socket = self.context.socket(zmq.PUB)
        self.comm_socket = self.context.socket(zmq.REP)
        self.comm_offset = comm_offset
        self.initialized = False

    def initialize(self, port, bufsize):
        if self.initialized:
            LOG.debug('Publisher is already initialized - Nothing to do')
            return

        offset = 0
        while offset < config.APP_BIND_ATTEMPT and not self.initialized:
            port_try = port + offset
            result = self.attempt_bind(port_try)
            offset += result
            if result == 0:
                output_str = 'Initialized publisher%s. Data port: %d, Comm port: %d'
                if offset == 0:
                    LOG.info(output_str, '', port_try, port_try + self.comm_offset)
                else:
                    LOG.warning(output_str, ' (alternate ports)', port_try, port_try + self.comm_offset)
            elif result == 1:
                LOG.warning('Unable to bind publisher to data port: %d', port_try)
            else:
                LOG.warning('Unable to bind publisher to communication port: %d', (port_try+self.comm_offset))

        # some logging output on the status of the port initialization attempts
        if not self.initialized:
            LOG.warning('Unable to initialize publisher after %d attempts - disabling!' % offset)

    def attempt_bind(self, port):
        try:
            self.bind(self.data_socket, port)
            try:
                self.bind(self.comm_socket, port + self.comm_offset)
                self.initialized = True
                return 0
            except zmq.ZMQError:
                # make sure to clean up the first bind which succeeded in this case
                self.unbind(self.data_socket, port)
                return 2
        except zmq.ZMQError as e:
            return 1

    def send(self, topic, data):
        if self.initialized:
            self.data_socket.send(topic + config.ZMQ_TOPIC_DELIM_CHAR, zmq.SNDMORE)
            self.data_socket.send_pyobj(data)

    def bind(self, sock, port):
        sock.bind('tcp://*:%d' % port)

    def unbind(self, sock, port):
        sock.unbind('tcp://*:%d' % port)


class ZMQListener(object):
    def __init__(self, request_pattern, reply_str, comm_socket):
        self.signal = re.compile(request_pattern)
        self.reply = reply_str
        self.comm_socket = comm_socket
        self.reset_lock = threading.Lock()
        self.reset_flag = threading.Event()
        self.thread = threading.Thread(target=self.comm_listener)
        self.thread.daemon = True

    def comm_listener(self):
        while not self.comm_socket.closed:
            msg = self.comm_socket.recv()
            signal_matcher = self.signal.match(msg)
            if signal_matcher is not None:
                self.set_flag()
                self.comm_socket.send(self.reply%signal_matcher.group(1))
                if LOG.isEnabledFor(logging.INFO):
                    LOG.info('Received valid comm request: %s', msg)
            else:
                self.comm_socket.send("invalid request from client")
                if LOG.isEnabledFor(logging.WARN):
                    LOG.warning('Invalid request received on comm port: %s', msg)

    def get_flag(self):
        with self.reset_lock:
            return self.reset_flag.is_set()

    def set_flag(self):
        with self.reset_lock:
            self.reset_flag.set()

    def clear_flag(self):
        with self.reset_lock:
            self.reset_flag.clear()

    def start(self):
        self.thread.start()
