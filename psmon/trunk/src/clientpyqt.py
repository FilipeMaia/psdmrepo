import sys
import socket
import logging
import threading

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

import psmon.plotpyqt as psplot
from psmon import app, config


LOG = logging.getLogger(__name__)


def set_color_opt(option, value):
    if value is not None:
        try:
            pg.setConfigOption(option, pg.functions.colorTuple(pg.functions.Color(value)))
        except:
             LOG.warning('Inavlid %s color for pyqtgraph: %s', option, value)


def main(client_info, plot_info):
    # initial all the socket connections
    zmqsub = app.ZMQSubscriber(client_info)

    # grab an initial datagram from the server
    init_data = zmqsub.data_recv()

    # attempt to decode its type and go from there
    try:
        data_type = psplot.type_getter(type(init_data))
    except TypeError:
        LOG.exception('Server returned an unknown datatype: %s', type(init_data))
        return 1

    # Disable/enable scipy.weave based on configuration settings
    pg.setConfigOption("useWeave", config.PYQT_USE_WEAVE)

    # start the QtApp
    qtapp = QtGui.QApplication([])
    # set widget background/foreground color if specified
    set_color_opt('background', plot_info.bkg_col)
    set_color_opt('foreground', plot_info.fore_col)

    # start the plotting rendering routine
    plot = data_type(init_data, zmqsub.get_socket_gen(), plot_info, rate=1.0/client_info.rate)
    plot.animate()

    # define signal sender function
    reset_req = app.ZMQRequester(
        config.RESET_REQ_STR%socket.gethostname(),
        config.RESET_REP_STR%socket.gethostname(),
        zmqsub.comm_socket
    )

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
