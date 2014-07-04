#!/usr/bin/env python
import logging

from psmon import app, config


LOG = logging.getLogger(__name__)


LOG.info('Importing publish')
__publisher = app.ZMQPublisher(config.APP_COMM_OFFSET)
__reset_listener = app.ZMQListener(config.RESET_REQ_STR%'(.*)', config.RESET_REP_STR, __publisher.comm_socket)


def initialized():
    return __publisher.initialized


def init(port=config.APP_PORT, bufsize=config.APP_BUFFER):
    __publisher.initialize(port, bufsize)


def send(topic, data):
    if initialized():
        __publisher.send(topic, data)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug('Published data to topic: %s', topic)


def get_reset_flag():
    return __reset_listener.get_flag()


def clear_reset_flag():
    __reset_listener.clear_flag()


def start_reset_listener():
    __reset_listener.start()
