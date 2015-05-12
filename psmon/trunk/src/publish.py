import logging

from psmon.pubutils import PublishManager as __PubManager


LOG = logging.getLogger(__name__)


LOG.info('Importing publish')


# Publish object for managing the module
__publish = __PubManager()


def initialized():
    """
    Returns the initialization state of the publish module.
    """
    return __publish.initialized


def get_port():
    """
    Returns the integer tcp port number used by the publish module.
    """
    return __publish.port


def get_bufsize():
    """
    Returns the zmq buffer size used by the publish module.
    """
    return __publish.bufsize


def get_local():
    """
    Returns the local publishing state of the publish module.
    """
    return __publish.local


def get_disable():
    """
    Returns the disabled status of autoconnecting.
    """
    return __publish.disable


def port(port):
    """
    Sets the starting tcp port number used by the publish module.

    Does nothing if attempting to change the port number after the module has 
    been initialized.
    """
    if not __publish.initialized:
        __publish.port = port
    elif LOG.isEnabledFor(logging.WARN):
        LOG.warn('Cannot change the tcp port number of the publish module after it has been initialized')


def bufsize(bufsize):
    """
    Sets the zmq buffer size used by the publish module.

    Does nothing if attempting to change the buffer size after the module has 
    been initialized.
    """
    if not __publish.initialized:
        __publish.bufsize = bufsize
    elif LOG.isEnabledFor(logging.WARN):
        LOG.warn('Cannot change the buffer size of the publish module after it has been initialized')


def local(local=True):
    """
    Used to set the local publishing state of the module. If the local flag of 
    the publish module is set then all plots are published to a client 
    launched locally.

    Does nothing if attempting to change publishing state after the module has 
    been initialized.
    """
    if not __publish.initialized:
        __publish.local = local
    elif LOG.isEnabledFor(logging.WARN):
        LOG.warn('Cannot change the local publishing state of the publish module after it has been initialized')


def plot_opts(**kwargs):
    """
    Plot options that are used when spawning local plots. Passed as keyword args.

    Unsupported options are ignored.
    """
    failed_opts = __publish.add_plot_opts(**kwargs)
    for failed_opt in failed_opts:
        if LOG.isEnabledFor(logging.WARN):
            LOG.warn('Unable to set plot option \'%s\': unsupported option', failed_opt)


def get_plot_opts():
    """
    Returns a dictionary of the currently set plot options. Can be modified.
    """
    return __publish.plot_opts


def disable(disable=True):
    """
    Used to disable autoconnection on send attempts for the publish module.

    Does nothing if attempting to disable autoconnection after the module has 
    been initialized.
    """
    if not __publish.initialized:
        __publish.disable = disable
    elif LOG.isEnabledFor(logging.WARN):
        LOG.warn('Cannot disable autoconnect for the publish module after it has been initialized')    


def init(port=None, bufsize=None, local=None):
    """
    Initializes the publish module.

    Optional arguments
    - port: the tcp port number to use with the publish module
    - bufsize: the zmq buffer size to use with the publish module
    - local: when true all plots are published to a client launched locally
    """
    if port is not None:
        __publish.port = port
    if bufsize is not None:
        __publish.bufsize = bufsize
    if local is not None:
        __publish.local = local
    __publish.init()


def reset():
    """
    Resets the publish module to its initial state.
    """
    __publish.reset()


def send(topic, data):
    """
    Publishes a data object to all clients suscribed to the topic.

    Arguments
    - topic: the name of the topic to which the data is being published
    - data: the data object to be published to suscribers
    """
    __publish.send(topic, data)
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug('Published data to topic: %s', topic)


def register_handler(name, **kwargs):
    """
    Registers a message handler for recieving messages from suscribed clients.

    Arguments
    - name: all messages sent from clients with this header will be handled by
    this message handler

    Returns a refernce to the newly created handler.
    """
    return __publish.reset_listener.register_handler(name, **kwargs)


def get_handler(name):
    """
    Returns a referenced to the named message handler.

    Arguments:
    - name: the header string/identifier of the requested handler
    """
    return __publish.reset_listener.message_handler.get(name)


def get_reset_flag():
    """
    Gets the state of the client reset flag. This will be set if any client 
    has sent a reset message, and will remain set until cleared.
    """
    return __publish.reset_listener.get_flag()


def clear_reset_flag():
    """
    Clears any set reset flags.
    """
    __publish.reset_listener.clear_flag()
