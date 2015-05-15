import os
import re
import sys
import zmq
import pwd
import Queue
import socket
import logging
import threading
from collections import namedtuple

from psmon import config


LOG = logging.getLogger(__name__)


class ClientInfo(object):
    def __init__(self, server, port, buffer, rate, recvlimit, topic):
        self.server = server
        self.port = port
        self.buffer = buffer
        self.rate = rate
        self.recvlimit = recvlimit
        self.topic = topic


class PlotInfo(object):
    def __init__(self, xrange=None, yrange=None, zrange=None, logx=False, logy=False, aspect=None, fore_col=None, bkg_col=None, interpol=None, palette=None, grid=False):
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange
        self.logx = logx
        self.logy = logy
        self.aspect = aspect
        self.fore_col = fore_col
        self.bkg_col = bkg_col
        self.interpol = interpol
        self.palette = palette
        self.grid = grid


class MessageHandler(object):
    def __init__(self, name, qlimit, is_pyobj):
        self.name = name
        self.is_pyobj = is_pyobj
        self.__mqueue = Queue.Queue(maxsize=qlimit)

    def get(self):
        return self.__mqueue.get_nowait()

    def put(self, msg):
        self.__mqueue.put_nowait(msg)

    @property
    def size(self):
        return self.__mqueue.qsize()

    @property
    def empty(self):
        return self.__mqueue.empty()

    @property
    def full(self):
        return self.__mqueue.full() 


class ZMQPublisher(object):
    def __init__(self, comm_offset=config.APP_COMM_OFFSET):
        self.context = zmq.Context()
        self.data_socket = self.context.socket(zmq.PUB)
        self.comm_socket = self.context.socket(zmq.REP)
        self.comm_offset = comm_offset
        self.initialized = False

    def initialize(self, port, bufsize):
        if self.initialized:
            LOG.debug('Publisher is already initialized - Nothing to do')
            return

        # set the hwm for the socket to the specified buffersize
        LOG.debug('Publisher data socket buffer size set to %d', bufsize)
        self.data_socket.set_hwm(bufsize)

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


class ZMQSubscriber(object):
    def __init__(self, client_info, comm_offset=config.APP_COMM_OFFSET, connect=True):
        self.client_info = client_info
        self.context = zmq.Context()
        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.setsockopt(zmq.SUBSCRIBE, self.client_info.topic + config.ZMQ_TOPIC_DELIM_CHAR)
        self.data_socket.set_hwm(self.client_info.buffer)
        self.comm_socket = self.context.socket(zmq.REQ)
        self.comm_offset = comm_offset
        self.connected = False
        if connect:
            self.connect()

    def connect(self):
        if not self.connected:
            self.sock_init(self.data_socket, self.client_info.server, self.client_info.port)
            self.sock_init(self.comm_socket, self.client_info.server, self.client_info.port + self.comm_offset)
            self.connected = True

    def sock_init(self, sock, server, port):
        sock.connect('tcp://%s:%d' % (server, port))

    def data_recv(self, flags=0):
        topic = self.data_socket.recv(flags)
        return self.data_socket.recv_pyobj(flags)

    def get_socket_gen(self):
        while True:
            count = 0
            data = None
            while count < self.client_info.recvlimit:
                try:
                    data = self.data_recv(flags=zmq.NOBLOCK)
                    count += 1
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        if LOG.isEnabledFor(logging.DEBUG):
                            LOG.debug('Number of queued messages discarded: %d', count)
                        break
                    else:
                        raise

            if count >= self.client_info.recvlimit and LOG.isEnabledFor(logging.WARN):
                LOG.warn('Number of queued messages exceeds the discard limit: %d', self.client_info.recvlimit)

            yield data


class ZMQListener(object):
    MessageHandle = namedtuple('MessageHandle', 'msg type')

    def __init__(self, comm_socket):
        self._reset = config.RESET_REQ_HEADER
        self._signal = re.compile(config.RESET_REQ_STR%'(.*)')
        self._reply = config.RESET_REP_STR
        self.__comm_socket = comm_socket
        self.__reset_flag = threading.Event()
        self.__message_handler = {}
        self.__thread = threading.Thread(target=self.comm_listener)
        self.__thread.daemon = True

    def send_reply(self, header, msg, send_py_obj=False):
        self.__comm_socket.send_string(header, zmq.SNDMORE)
        if send_py_obj:
            self.__comm_socket.send_pyobj(msg)
        else:
            self.__comm_socket.send_string(msg)

    def register_handler(self, name, limit=0, is_pyobj=True):
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug('Attempting to register message handler: name=%s, limit=%s, pyobj=%s', (name, limit, is_pyobj))
        if name in self.__message_handler:
            if LOG.isEnabledFor(logging.WARN):
                LOG.warning('Attempted to register message handler which already exists: %s', name)
            raise ValueError('Message handler \'%s\' already registered'%name)
        handler = self.__message_handler[name] = MessageHandler(name, limit, is_pyobj)
        if LOG.isEnabledFor(logging.INFO):
            LOG.info('Sucessfully registered message handler: %s'%name)
        return handler

    def comm_listener(self):
        while not self.__comm_socket.closed:
            header = self.__comm_socket.recv_string()
            if header == self._reset:
                msg = self.__comm_socket.recv_string()
                signal_matcher = self._signal.match(msg)
                if signal_matcher is not None:
                    self.set_flag()
                    self.send_reply(self._reset, self._reply%signal_matcher.group(1))
                    if LOG.isEnabledFor(logging.INFO):
                        LOG.info('Received valid reset request: %s', msg)
                else:
                    self.send_reply(self._reset, "invalid request from client")
                    if LOG.isEnabledFor(logging.WARN):
                        LOG.warning('Invalid request received on comm port: %s', msg)
            else:
                if header in self.__message_handler:
                    if self.__message_handler[header].is_pyobj:
                        msg = self.__comm_socket.recv_pyobj()
                    else:
                        msg = self.__comm_socket.recv_string()
                    try:
                        self.__message_handler[header].put(msg)
                        if LOG.isEnabledFor(logging.DEBUG):
                            LOG.debug('Message for handler \'%s\' processed', header)
                        self.send_reply(header, 'Message for handler processed')
                    except Queue.Full:
                        if LOG.isEnabledFor(logging.WARN):
                            LOG.warning('Message handler \'%s\' is full - request dropped', header)
                        self.send_reply(header, 'Message handler full - request dropped')
                else:
                    if LOG.isEnabledFor(logging.DEBUG):
                        LOG.debug('Received message for unregistered handler: %s', header)

    def get_flag(self):
        return self.__reset_flag.is_set()

    def set_flag(self):
        self.__reset_flag.set()

    def clear_flag(self):
        self.__reset_flag.clear()

    def start(self):
        if not self.__thread.isAlive():
            self.__thread.start()


class ZMQRequester(object):
    def __init__(self, comm_socket):
        self._reset = config.RESET_REQ_HEADER
        self._request = config.RESET_REQ_STR%socket.gethostname()
        self._req_reply = config.RESET_REP_STR%socket.gethostname()
        self.__comm_socket = comm_socket
        self.__comm_lock = threading.Lock()
        self.__pending_flag = threading.Event()
        self.__thread = None

    def send_request(self, header, msg, send_py_obj=True, recv_py_obj=False):
        with self.__comm_lock:
            self.__comm_socket.send_string(header, zmq.SNDMORE)
            if send_py_obj:
                self.__comm_socket.send_pyobj(msg)
            else:
                self.__comm_socket.send_string(msg)
            rep_header = self.__comm_socket.recv_string()
            if header != rep_header and LOG.isEnabledFor(logging.WARN):
                LOG.warning('Request header does not match repy header: \'%s\' and \'%s\'', header, rep_header)
            if recv_py_obj:
                rep_msg = self.__comm_socket.recv_pyobj()
            else:
                rep_msg = self.__comm_socket.recv_string()

            return rep_msg
        
    def reset_signal(self):
        # check to see if there is another pending reset req
        if not self.__pending_flag.is_set():
            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug('Sending reset request to server')
            self.__pending_flag.set()
            reply = self.send_request(self._reset, self._request, False)
            if reply != self._req_reply and LOG.isEnabledFor(logging.ERROR):
                LOG.error('Server returned unexpected reply to reset request: %s', reply)
            self.__pending_flag.clear()

    def send_reset_signal(self, *args):
        self.__thread = threading.Thread(target=self.reset_signal)
        self.__thread.daemon = True
        self.__thread.start()
