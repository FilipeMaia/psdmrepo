import sys
import logging
import threading

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

import psmon.psplotpyqt as psplot
from psmon import psapp, psconfig


LOG = logging.getLogger(__name__)


def set_color_opt(option, value):
    if value is not None:
        try:
            pg.setConfigOption(option, pg.functions.colorTuple(pg.functions.Color(value)))
        except:
             LOG.warning('Inavlid %s color for pyqtgraph: %s', option, value)


def main(client_info, plot_info):
    # initial all the socket connections
    context, sub_socket, reset_socket = psapp.init_client_sockets(client_info)

    # grab an initial datagram from the server
    init_data = psapp.socket_recv(sub_socket)

    # attempt to decode its type and go from there
    try:
        data_type = psplot.type_getter(type(init_data))
    except TypeError:
        LOG.exception('Server returned an unknown datatype: %s', type(init_data))
        return 1

    # start the QtApp
    app = QtGui.QApplication([])
    # set widget background/foreground color if specified
    set_color_opt('background', plot_info.bkg_col)
    set_color_opt('foreground', plot_info.fore_col)

    # start the plotting rendering routine
    plot = data_type(init_data, psapp.get_socket_gen(sub_socket), plot_info, rate=1.0/client_info.rate)
    plot.animate()

    # define signal sender function
    pending_req = threading.Event()
    def send_reset_signal(*args):
        sender_thread = threading.Thread(target=psapp.reset_signal, args=(reset_socket, pending_req))
        sender_thread.daemon = True
        sender_thread.start()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
