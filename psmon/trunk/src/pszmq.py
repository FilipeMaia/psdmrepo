#!/usr/bin/env python
import re
import zmq
import socket
import threading

from psmon import psconfig

print '*** initializing pszmq'
context = zmq.Context()
socket = context.socket(zmq.PUB)
reset_socket = context.socket(zmq.REP)
initialized = False
__signal = re.compile(psconfig.RESET_REQ_STR%'(.*)')
__reset_lock = threading.Lock()
__reset_flag = threading.Event()

def socket_init(port, reset_port, bufsize):
    global initialized
    socket.set_hwm(bufsize)
    socket.bind("tcp://*:%d" % port)
    socket.set_hwm(bufsize)
    reset_socket.bind("tcp://*:%d" % reset_port)
    initialized = True
    print 'initialized',initialized

def send_data(topic, data):
    if initialized:
        socket.send(topic + psconfig.ZMQ_TOPIC_DELIM_CHAR, zmq.SNDMORE)
        socket.send_pyobj(data)

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
    while initialized:
        reset_msg = reset_socket.recv()
        signal_matcher = __signal.match(reset_msg)
        if signal_matcher is not None:
            set_reset_flag()
            reset_socket.send(psconfig.RESET_REP_STR%signal_matcher.group(1))
        else:
            reset_socket.send("invalid request from client")
