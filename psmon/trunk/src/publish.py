#!/usr/bin/env python
import re
import zmq
import logging
import threading

from psmon import config


LOG = logging.getLogger(__name__)


LOG.info('Importing publish')
__context = zmq.Context()
__socket = __context.socket(zmq.PUB)
__reset_socket = __context.socket(zmq.REP)
__initialized = False
__signal = re.compile(config.RESET_REQ_STR%'(.*)')
__reset_lock = threading.Lock()
__reset_flag = threading.Event()


def is_initialized():
    return __initialized


def init(port=config.APP_PORT, bufsize=config.APP_BUFFER):
    global __initialized
    __socket.set_hwm(bufsize)
    __socket.bind("tcp://*:%d" % port)
    __socket.set_hwm(bufsize)
    __reset_socket.bind("tcp://*:%d" % (port+1))
    __initialized = True
    LOG.info('Initialized publish. Data port: %d', port)


def send(topic, data):
    if __initialized:
        __socket.send(topic + config.ZMQ_TOPIC_DELIM_CHAR, zmq.SNDMORE)
        __socket.send_pyobj(data)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug('Published data to topic: %s', topic)


def get_reset_flag():
    with __reset_lock:
        return __reset_flag.is_set()


def set_reset_flag():
    with __reset_lock:
        __reset_flag.set()


def clear_reset_flag():
    with __reset_lock:
        __reset_flag.clear()


def reset_listener():
    while __initialized:
        reset_msg = __reset_socket.recv()
        signal_matcher = __signal.match(reset_msg)
        if signal_matcher is not None:
            set_reset_flag()
            __reset_socket.send(config.RESET_REP_STR%signal_matcher.group(1))
            LOG.debug('Received reset request')
        else:
            __reset_socket.send("invalid request from client")
            LOG.warning('Invalid request received on reset port')

def start_listener():
    listener_thread = threading.Thread(target=server_script.reset_listener) #, args=(server_script,)
    listener_thread.daemon = True
    listener_thread.start()
