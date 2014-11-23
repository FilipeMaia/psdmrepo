import sys
import math
import logging
import collections

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
from matplotlib.axes import _process_plot_format

from psmon import config
from psmon.util import is_py_iter, arg_inflate_flat, arg_inflate_tuple, inflate_input
from psmon.util import window_ratio
from psmon.plots import Hist, Image, XYPlot, MultiPlot


LOG = logging.getLogger(__name__)


TypeMap = {
    Hist: 'HistClient',
    Image: 'ImageClient',
    XYPlot: 'XYPlotClient',
    MultiPlot: 'MultiPlotClient',
}


def type_getter(data_type, mod_name=__name__):
    plot_type_name = TypeMap.get(data_type)
    return getattr(sys.modules[__name__], plot_type_name)


class Plot(object):
    def __init__(self, init, framegen, info, rate, **kwargs):
        if 'figax' in kwargs:
            self.figure, self.ax = kwargs['figax']
        else:
            self.figure, self.ax = plt.subplots(facecolor=info.bkg_col, edgecolor=info.bkg_col, subplot_kw={'axis_bgcolor': info.bkg_col or config.MPL_AXES_BKG_COLOR})
            self.figure.canvas.set_window_title(init.title)
        self.info = info
        self.set_title(init.ts)
        self.set_labels(init.xlabel, init.ylabel)
        self.set_ax_col(self.ax)
        self.framegen = framegen
        self.rate_ms = rate * 1000
        self.multi_plot = False

    def update_sub(self, data):
        pass

    def update(self, data):
        if data is not None:
            self.set_title(data.ts)
            self.set_labels(data.xlabel, data.ylabel)
        return self.update_sub(data)

    def animate(self):
        return animation.FuncAnimation(self.figure, self.update, self.ani_func, interval=self.rate_ms)

    def ani_func(self):
        yield self.framegen.next()

    def set_title(self, title):
        if title is not None:
            if self.info.fore_col is not None:
                self.ax.set_title(title, loc='right', color=self.info.fore_col)
            else:
                self.ax.set_title(title, loc='right')

    def set_labels(self, xlabel=None, ylabel=None):
        self.set_axis_label(self.ax.set_xlabel, xlabel)
        self.set_axis_label(self.ax.set_ylabel, ylabel)

    def set_axis_label(self, axis_label_func, axis_label_data):
        if isinstance(axis_label_data, collections.Mapping):
            if 'axis_title' in axis_label_data:
                label_str = axis_label_data['axis_title']
                if 'axis_units' in axis_label_data:
                    if 'axis_units_prefix' in axis_label_data:
                        label_str += ' [%s%s]'%(axis_label_data['axis_units'], axis_label_data['axis_units_prefix']) 
                    else:
                        label_str += ' [%s]'%axis_label_data['axis_units']
                axis_label_func(label_str)
        elif axis_label_data is not None:
            axis_label_func(axis_label_data)

    def set_xy_ranges(self):
        if self.info.xrange is not None:
            self.ax.set_xlim(self.info.xrange)
        if self.info.yrange is not None:
            self.ax.set_ylim(self.info.yrange)

    def set_aspect(self):
        if self.info.aspect is not None:
            self.ax.set_aspect(self.info.aspect)

    def set_ax_col(self, ax):
        if self.info.fore_col is not None:
            for ax_name in ['bottom', 'top', 'right', 'left']:
                ax.spines[ax_name].set_color(self.info.fore_col)
            for ax_name in ['x', 'y']:
                ax.tick_params(axis=ax_name, colors=self.info.fore_col)
            ax.yaxis.label.set_color(self.info.fore_col)
            ax.xaxis.label.set_color(self.info.fore_col)

    def update_plot_data(self, plots, x_vals, y_vals, new_fmts, old_fmts):
        for index, (plot, data_tup, old_fmt) in enumerate(zip(plots, arg_inflate_tuple(1, x_vals, y_vals, new_fmts), old_fmts)):
            x_val, y_val, new_fmt = data_tup
            plot.set_data(x_val, y_val)
            if new_fmt != old_fmt:
                # parse the format string
                linestyle, marker, color = _process_plot_format(new_fmt)
                linestyle = linestyle or rcParams['lines.linestyle']
                marker = marker or rcParams['lines.marker']
                color = color or rcParams['lines.color']
                plot.set_linestyle(linestyle)
                plot.set_marker(marker)
                plot.set_color(color)
                old_fmts[index] = new_fmt


class MultiPlotClient(object):
    def __init__(self, init, framegen, info, rate):
        # set default column and row values
        ncols = init.size
        nrows = 1
        # if any column organization data is passed try to use it
        if init.ncols is not None:
            if isinstance(init.ncols, int) and 0 < init.ncols <= init.size:
                ncols = init.ncols
                nrows = int(math.ceil(init.size/float(init.ncols)))
            else:
                LOG.warning('Invalid column number specified: %s - Must be a positive integer less than the number of plots: %s', init.ncols, init.size)
        if init.use_windows:
            LOG.warning('Separate windows for subplots is not supported in the matplotlib client')
        ratio_calc = window_ratio(config.MPL_SMALL_WIN, config.MPL_LARGE_WIN)
        self.figure, self.ax = plt.subplots(nrows=nrows, ncols=ncols, facecolor=info.bkg_col, edgecolor=info.bkg_col, figsize=ratio_calc(nrows, ncols))
        # flatten the axes array returned by suplot
        self.ax = self.ax.flatten() if init.size > 1 else [self.ax]
        self.figure.canvas.set_window_title(init.title)
        self.plots = [type_getter(type(data_obj))(data_obj, None, info, rate, figax=(self.figure, subax)) for data_obj, subax in zip(init.data_con, self.ax)]
        self.framegen = framegen
        self.rate_ms = rate * 1000
        self.info = info
        self.multi_plot = True

    def update(self, data):
        if data is not None:
            for plot, plot_data in zip(self.plots, data.data_con):
                plot.update(plot_data)

    def animate(self):
        return animation.FuncAnimation(self.figure, self.update, self.ani_func, interval=self.rate_ms)

    def ani_func(self):
        yield self.framegen.next()


class ImageClient(Plot):
    def __init__(self, init_im, framegen, info, rate=1, **kwargs):
        super(ImageClient, self).__init__(init_im, framegen, info, rate, **kwargs)
        # if a color palette is specified check to see if it valid
        cmap = plt.get_cmap(config.MPL_COLOR_PALETTE)
        if self.info.palette is not None:
            try:
                cmap = plt.get_cmap(self.info.palette)
            except ValueError:
                LOG.warning('Inavlid color palette for matplotlib: %s - Falling back to default: %s', self.info.palette, cmap.name)
        self.im = self.ax.imshow(init_im.image, interpolation=self.info.interpol, cmap=cmap)
        self.im.set_clim(self.info.zrange)
        self.cb = self.figure.colorbar(self.im, ax=self.ax)
        self.set_cb_col()
        self.set_aspect()
        self.set_xy_ranges()

    def update_sub(self, data):
        """
        Updates the data in the image - none means their was no update for this interval
        """
        if data is not None:
            self.im.set_data(data.image)
        return self.im

    def set_cb_col(self):
        if self.info.fore_col is not None:
            self.cb.outline.set_color(self.info.fore_col)
            self.set_ax_col(self.cb.ax)


class HistClient(Plot):
    def __init__(self, init_hist, datagen, info, rate=1, **kwargs):
        super(HistClient, self).__init__(init_hist, datagen, info, rate, **kwargs)
        # pyqtgraph needs a trailing bin edge that mpl doesn't so check for that
        if init_hist.bins.size > init_hist.values.size:
            plot_args = arg_inflate_flat(1, init_hist.bins[:-1], init_hist.values, init_hist.formats)
        else:
            plot_args = arg_inflate_flat(1, init_hist.bins, init_hist.values, init_hist.formats)
        self.hists = self.ax.plot(*plot_args, drawstyle=config.MPL_HISTO_STYLE)
        self.formats = inflate_input(init_hist.formats, init_hist.values)
        self.set_aspect()
        self.set_xy_ranges()

    def update_sub(self, data):
        if data is not None:
            # pyqtgraph needs a trailing bin edge that mpl doesn't so check for that
            if data.bins.size > data.values.size:
                self.update_plot_data(self.hists, data.bins[:-1], data.values, data.formats, self.formats)
            else:
                self.update_plot_data(self.hists, data.bins, data.values, data.formats, self.formats)
            self.ax.relim()
            self.ax.autoscale_view()
        return self.hists


class XYPlotClient(Plot):
    def __init__(self, init_plot, datagen, info, rate=1, **kwargs):
        super(XYPlotClient, self).__init__(init_plot, datagen, info, rate, **kwargs)
        plot_args = arg_inflate_flat(1, init_plot.xdata, init_plot.ydata, init_plot.formats)
        self.plots = self.ax.plot(*plot_args)
        self.formats = inflate_input(init_plot.formats, init_plot.ydata)
        self.set_aspect()
        self.set_xy_ranges()

    def update_sub(self, data):
        if data is not None:
            self.update_plot_data(self.plots, data.xdata, data.ydata, data.formats, self.formats)
            self.ax.relim()
            self.ax.autoscale_view()
        return self.plots
