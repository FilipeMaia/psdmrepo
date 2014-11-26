#!/usr/bin/env python
import logging

from psmon import app, config


LOG = logging.getLogger(__name__)


LOG.info('Importing publish')
__publisher = app.ZMQPublisher()
__reset_listener = app.ZMQListener(__publisher.comm_socket)


def initialized():
    return __publisher.initialized


def init(port=config.APP_PORT, bufsize=config.APP_BUFFER):
    __publisher.initialize(port, bufsize)
    __reset_listener.start()


def send(topic, data):
    if initialized():
        __publisher.send(topic, data)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug('Published data to topic: %s', topic)


def register_handler(name, **kwargs):
    return __reset_listener.register_handler(name, **kwargs)


def get_handler(name):
   return __reset_listener.message_handler.get(name)


def get_reset_flag():
    return __reset_listener.get_flag()


def clear_reset_flag():
    __reset_listener.clear_flag()
