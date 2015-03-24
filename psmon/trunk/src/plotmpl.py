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
    if plot_type_name is None:
        raise MplClientTypeError('No plotting client for datatype: %s' % data_type)
    return getattr(sys.modules[mod_name], plot_type_name)


class MplClientTypeError(Exception):
    pass


class PlotClient(object):
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

    def set_aspect(self, lock=True, ratio=None):
        if ratio is None:
            ratio=self.info.aspect
        if lock:
            if ratio is not None:
                self.ax.set_aspect(ratio)
        else:
            self.ax.set_aspect('auto')

    def set_ax_col(self, ax):
        if self.info.fore_col is not None:
            for ax_name in ['bottom', 'top', 'right', 'left']:
                ax.spines[ax_name].set_color(self.info.fore_col)
            for ax_name in ['x', 'y']:
                ax.tick_params(axis=ax_name, colors=self.info.fore_col)
            ax.yaxis.label.set_color(self.info.fore_col)
            ax.xaxis.label.set_color(self.info.fore_col)

    def set_grid_lines(self, show=None):
        if show is None:
            show = self.info.grid
        self.ax.grid(show)

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

    def add_legend(self, plots, plot_data, leg_label, leg_offset):
        if leg_label is not None:
            self.legend_labels = inflate_input(leg_label, plot_data)
            for plot, label in zip(plots, self.legend_labels):
                plot.set_label(label)
            self.legend = self.ax.legend()


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
        self.figure, self.ax = plt.subplots(nrows=nrows, ncols=ncols, facecolor=info.bkg_col, edgecolor=info.bkg_col, figsize=ratio_calc(ncols, nrows), squeeze=False)
        # flatten the axes array returned by suplot
        self.ax = self.ax.flatten()
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


class ImageClient(PlotClient):
    def __init__(self, init_im, framegen, info, rate=1, **kwargs):
        super(ImageClient, self).__init__(init_im, framegen, info, rate, **kwargs)
        # if a color palette is specified check to see if it valid
        cmap = plt.get_cmap(config.MPL_COLOR_PALETTE)
        # deal with custom axis ranges if requested
        if init_im.pos is None and init_im.scale is None:
            extent = None
        else:
            x1 = 0 if init_im.pos is None else init_im.pos[0]
            xscale = 1 if init_im.scale is None else init_im.scale[0]
            y1 = 0 if init_im.pos is None else init_im.pos[1]
            yscale = 1 if init_im.scale is None else init_im.scale[1]
            extent = [x1, x1 + xscale * init_im.image.shape[1], y1 + yscale * init_im.image.shape[0], y1]
        if self.info.palette is not None:
            try:
                cmap = plt.get_cmap(self.info.palette)
            except ValueError:
                LOG.warning('Inavlid color palette for matplotlib: %s - Falling back to default: %s', self.info.palette, cmap.name)
        self.im = self.ax.imshow(init_im.image, interpolation=self.info.interpol, cmap=cmap, extent=extent)
        self.im.set_clim(self.info.zrange)
        self.cb = self.figure.colorbar(self.im, ax=self.ax)
        self.set_cb_col()
        self.set_aspect(init_im.aspect_lock, init_im.aspect_ratio)
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


class HistClient(PlotClient):
    def __init__(self, init_hist, datagen, info, rate=1, **kwargs):
        super(HistClient, self).__init__(init_hist, datagen, info, rate, **kwargs)
        # pyqtgraph needs a trailing bin edge that mpl doesn't so check for that
        plot_args = arg_inflate_flat(1, self.correct_bins(init_hist.bins, init_hist.values), init_hist.values, init_hist.formats)
        self.hists = self.ax.plot(*plot_args, drawstyle=config.MPL_HISTO_STYLE)
        self.formats = inflate_input(init_hist.formats, init_hist.values)
        self.set_aspect()
        self.set_xy_ranges()
        self.set_grid_lines()
        self.add_legend(self.hists, init_hist.values, init_hist.leg_label, init_hist.leg_offset)

    def update_sub(self, data):
        if data is not None:
            # pyqtgraph needs a trailing bin edge that mpl doesn't so check for that
            self.update_plot_data(self.hists, self.correct_bins(data.bins, data.values), data.values, data.formats, self.formats)
            self.ax.relim()
            self.ax.autoscale_view()
        return self.hists

    def correct_bins(self, bins, values):
        """
        Checks that number of bins is correct for matplotlib. pyqtgraph needs a 
        trailing bin edge that mpl doesn't so check for that and remove if 
        needed.

        Takes the 'bins' numpy array (single or list of) and compares to the 
        'values' numpy array (single or list of) and trims trailing entry from 
        'bins' if its size is greater than that of the mathcing 'values'.

        Returns the corrected 'bins'.
        """
        if is_py_iter(bins) or is_py_iter(values):
            corrected_bins = []
            for bin, value in zip(inflate_input(bins, values), values):
                if bin.size > value.size:
                    corrected_bins.append(bin[:-1])
                else:
                    corrected_bins.append(bin)
            return corrected_bins
        elif bins.size > values.size:
            return bins[:-1]
        else:
            return bins


class XYPlotClient(PlotClient):
    def __init__(self, init_plot, datagen, info, rate=1, **kwargs):
        super(XYPlotClient, self).__init__(init_plot, datagen, info, rate, **kwargs)
        plot_args = arg_inflate_flat(1, init_plot.xdata, init_plot.ydata, init_plot.formats)
        self.plots = self.ax.plot(*plot_args)
        self.formats = inflate_input(init_plot.formats, init_plot.ydata)
        self.set_aspect()
        self.set_xy_ranges()
        self.set_grid_lines()
        self.add_legend(self.plots, init_plot.ydata, init_plot.leg_label, init_plot.leg_offset)

    def update_sub(self, data):
        if data is not None:
            self.update_plot_data(self.plots, data.xdata, data.ydata, data.formats, self.formats)
            self.ax.relim()
            self.ax.autoscale_view()
        return self.plots
