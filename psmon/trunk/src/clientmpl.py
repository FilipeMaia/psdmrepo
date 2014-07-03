import logging
import threading

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import psmon.plotmpl as psplot
from psmon import app, config


LOG = logging.getLogger(__name__)


def main(client_info, plot_info):
    # initial all the socket connections
    context, sub_socket, reset_socket = app.init_client_sockets(client_info)

    # grab an initial datagram from the server
    init_data = app.socket_recv(sub_socket)

    # attempt to decode its type and go from there
    try:
        data_type = psplot.type_getter(type(init_data))
    except TypeError:
        LOG.exception('Server returned an unknown datatype: %s', type(init_data))
        return 1

    plot = data_type(init_data, app.get_socket_gen(sub_socket), plot_info, rate=1.0/client_info.rate)
    plot_ani = plot.animate()

    # auto zoom button params
    az_xpos = 0.01
    az_ypos = 0.015
    az_xlen = 0.12
    az_ylen = 0.035
    az_xlen_multi = 0.14
    # add some border to the bottom of the plot for the buttons
    plot.figure.subplots_adjust(bottom=0.12)
    if plot.multi_plot:
        button_cnt = 0
        auto_zoom_buttons = []
        for sub_ax in plot.ax:
            auto_zoom_button = Button(plt.axes([az_xpos + button_cnt * (az_xpos + az_xlen_multi) , az_ypos, az_xlen_multi, az_ylen]),
                                      'Auto Zoom %d' % (button_cnt+1))
            auto_zoom_button.on_clicked(sub_ax.autoscale)
            auto_zoom_buttons.append(auto_zoom_button)
            button_cnt += 1
    else:
        auto_zoom_button = Button(plt.axes([az_xpos, az_ypos, az_xlen, az_ylen]), 'Auto Zoom')
        auto_zoom_button.on_clicked(plot.ax.autoscale)

    # define signal sender function
    pending_req = threading.Event()
    def send_reset_signal(*args):
        sender_thread = threading.Thread(target=app.reset_signal, args=(reset_socket, pending_req))
        sender_thread.daemon = True
        sender_thread.start()

    reset_plots_button = Button(plt.axes([0.87, 0.015, 0.12, 0.035]), 'Reset Plots')
    reset_plots_button.on_clicked(send_reset_signal)

    try:
        plt.show()
    except:
        # sort of ugly but this can throw all kinds of errors when closing a window
        pass

    # return process status code
    return 0
