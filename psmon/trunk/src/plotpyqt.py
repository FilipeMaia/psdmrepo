import sys
import math
import logging
import collections
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from psmon import config
from psmon.util import is_py_iter, arg_inflate_tuple, window_ratio
from psmon.plots import Hist, Image, XYPlot, MultiPlot
from psmon.format import parse_fmt_xyplot, parse_fmt_hist


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
        raise PyQtClientTypeError('No plotting client for datatype: %s' % data_type)
    return getattr(sys.modules[mod_name], plot_type_name)


class PyQtClientTypeError(Exception):
    pass


class PlotClient(object):
    def __init__(self, init, framegen, info, rate, **kwargs):
        if 'figwin' in kwargs:
            self.fig_win = kwargs['figwin']
            self.fig_layout = self.fig_win.addLayout()
            self.title_layout = self.fig_layout.addLayout()
            self._set_row_stretch(0)
            self.title_layout.addLabel(init.title, size='11pt', bold=True)
            self.fig_layout.nextRow()
        else:
            self.fig_win = pg.GraphicsLayoutWidget()
            self.fig_win.setWindowTitle(init.title)
            self.fig_win.show()
            self.fig_layout = self.fig_win.ci
        # create a sublayout for the plot itself
        self.plot_layout = self.fig_layout.addLayout()
        self._set_row_stretch(1)
        self.plot_view = self.plot_layout.addPlot()
        # creat a sublayout under the plot for info/buttons
        self.fig_layout.nextRow()
        self.info_layout = self.fig_layout.addLayout()
        self._set_row_stretch(0)
        self.info_label = self.info_layout.addLabel('', justify='right')
        # set labels
        self.set_title(init.ts)
        self.set_title_axis('bottom', init.xlabel)
        self.set_title_axis('left', init.ylabel)
        # specific to this class
        self.framegen = framegen
        self.rate_ms = rate * 1000
        self.info = info
        self.multi_plot = False
        # set any user specified default axis ranges
        self.set_xy_ranges()
        # show grid lines if requested
        self.set_grid_lines()
        # create cursor event listener
        self.proxy = pg.SignalProxy(
            self.plot_view.scene().sigMouseMoved,
            rateLimit=config.PYQT_MOUSE_EVT_RATELIMIT,
            slot=self.cursor_hover_evt
        )

    def update_sub(self, data):
        pass

    def update(self, data):
        """
        Base update function - meant for basic functionality that should happen for all plot/image updates.

        Calls update_sub(self, data) which should be implement an Plot subclass specific update behavior
        """
        if data is not None:
            self.set_title(data.ts)
            self.set_title_axis('bottom', data.xlabel)
            self.set_title_axis('left', data.ylabel)
        return self.update_sub(data)

    def animate(self):
        self.ani_func()

    def ani_func(self):
        # call the data update function
        self.update(self.framegen.next())
        # setup timer for calling next update call
        QtCore.QTimer.singleShot(self.rate_ms, self.ani_func)

    def set_title(self, title):
        if title is not None:
            self.plot_view.setTitle(title, size='10pt', justify='right')

    def set_title_axis(self, axis_name, axis_label_data):
        """
        Function for setting a label on the axis specified by 'axis_name'. The label data can be either a simple
        string or a dictionary of keywords that is passed on to the pyqtgraph setLabel function.

        Supported keywords:
        - axis_title
        - axis_units
        - axis_unit_prefix
        """
        if isinstance(axis_label_data, collections.Mapping):
            self._set_title_axis(axis_name, **axis_label_data)
        else:
            self._set_title_axis(axis_name, axis_label_data)

    def _set_title_axis(self, axis_name, axis_title, axis_units=None, axis_unit_prefix=None):
        """
        Implementation function for creating a label for a specific axis - takes an axis_name, axis_title and optional
        axis_units, and axis_unit_prefix keyword args, which match to those for pyqtgraph's set label
        """
        if axis_title is not None:
            self.plot_view.setLabel(axis_name, text=axis_title, units=axis_units, unitPrefix=axis_unit_prefix)

    def set_aspect(self, ratio):
        """
        Set the ascept ratio of the viewbox of the plot/image to the specified ratio.

        If no ratio is passed it uses the client side default.

        Note: this is disabled if explicit x/y ranges are set for view box since the
        two options fight each other.
        """
        if ratio is None:
            ratio=self.info.aspect

        # Since the images are technically transposed this is needed for the ratio to work the same as mpl
        if ratio is not None:
            ratio = 1.0 / ratio

        if self.info.xrange is None and self.info.yrange is None:
            self.plot_view.getViewBox().setAspectLocked(lock=True, ratio=ratio)

    def set_xy_ranges(self):
        if self.info.xrange is not None:
            self.plot_view.setXRange(*self.info.xrange)
        if self.info.yrange is not None:
            self.plot_view.setYRange(*self.info.yrange)

    def set_grid_lines(self, show=None, alpha=config.PYQT_GRID_LINE_ALPHA):
        if show is None:
            show = self.info.grid
        self.plot_view.showGrid(x=show, y=show, alpha=alpha)

    def _set_row_stretch(self, val):
        self.fig_layout.layout.setRowStretchFactor(self.fig_layout.currentRow, val)

    def cursor_hover_evt_sub(self, x_pos, y_pos):
        self.info_label.setText('x=%d, y=%d' % (x_pos, y_pos), size='10pt')

    def cursor_hover_evt(self, evt):
        pos = evt[0]
        if self.plot_view.sceneBoundingRect().contains(pos):
            mouse_pos = self.plot_view.getViewBox().mapSceneToView(pos)
            self.cursor_hover_evt_sub(int(mouse_pos.x()), int(mouse_pos.y()))


class ImageClient(PlotClient):
    def __init__(self, init_im, framegen, info, rate=1, **kwargs):
        super(ImageClient, self).__init__(init_im, framegen, info, rate, **kwargs)
        if init_im.aspect_lock:
            self.set_aspect(init_im.aspect_ratio)
        self.set_grid_lines(False)
        self.im = pg.ImageItem(image=init_im.image.T, border=config.PYQT_BORDERS)
        self.cb = pg.HistogramLUTItem(self.im, fillHistogram=True)

        # Setting up the color map to use
        cm = config.PYQT_COLOR_PALETTE
        if self.info.palette is not None:
            if self.info.palette in pg.graphicsItems.GradientEditorItem.Gradients:
                cm = self.info.palette
            else:
                LOG.warning('Inavlid color palette for pyqtgraph: %s - Falling back to default: %s', self.info.palette, cm)
        self.cb.gradient.loadPreset(cm)

        # Set up colorbar ranges if specified
        if self.info.zrange is not None:
            self.cb.setLevels(*self.info.zrange)
            self.cb.setHistogramRange(*self.info.zrange)
        else:
            self.cb.setHistogramRange(*self.cb.getLevels())

        if config.PYQT_USE_ALT_IMG_ORIGIN:
            self.plot_view.invertY()
        self.plot_view.addItem(self.im)
        self.plot_layout.addItem(self.cb)

    def update_sub(self, data):
        """
        Updates the data in the image - none means their was no update for this interval
        """
        if data is not None:
            self.im.setImage(data.image.T, autoLevels=False)
        return self.im

    def cursor_hover_evt_sub(self, x_pos, y_pos):
        if 0 <= x_pos < self.im.image.shape[0] and 0 <= y_pos < self.im.image.shape[1]:
            z_val = self.im.image[x_pos][y_pos]
            # for image of float type show decimal places
            if hasattr(z_val, 'dtype') and np.issubdtype(z_val, np.integer):
                label_str = 'x=%d, y=%d, z=%d'
            else:
                label_str = 'x=%d, y=%d, z=%.3f'
            self.info_label.setText(label_str % (x_pos, y_pos, z_val), size='10pt')


class XYPlotClient(PlotClient):
    def __init__(self, init_plot, framegen, info, rate=1, **kwargs):
        super(XYPlotClient, self).__init__(init_plot, framegen, info, rate, **kwargs)
        self.plots = []
        self.formats = []
        for xdata, ydata, format_val in arg_inflate_tuple(1, init_plot.xdata, init_plot.ydata, init_plot.formats):
            cval = len(self.plots)
            self.formats.append((format_val, cval))
            self.plots.append(
                self.plot_view.plot(
                    x=xdata,
                    y=ydata,
                    **parse_fmt_xyplot(format_val, cval)
                )
            )

    def update_sub(self, data):
        """
        Updates the data in the plot - none means their was no update for this interval
        """
        if data is not None:
            for index, (plot, data_tup, format_tup) in enumerate(zip(self.plots, arg_inflate_tuple(1, data.xdata, data.ydata, data.formats), self.formats)):
                xdata, ydata, new_format = data_tup
                old_format, cval = format_tup
                if new_format != old_format:
                    self.formats[index] = (new_format, cval)
                    plot.setData(x=xdata, y=ydata, **parse_fmt_xyplot(new_format, cval))
                else:
                    plot.setData(x=xdata, y=ydata)
        return self.plots


class HistClient(PlotClient):
    def __init__(self, init_hist, framegen, info, rate=1, **kwargs):
        super(HistClient, self).__init__(init_hist, framegen, info, rate, **kwargs)
        self.hists = []
        self.formats = []
        for bins, values, format_val in arg_inflate_tuple(1, init_hist.bins, init_hist.values, init_hist.formats):
            cval = len(self.hists)
            self.formats.append((format_val, cval))
            hist = pg.PlotCurveItem(
                x=bins,
                y=values,
                stepMode=True,
                fillLevel=0,
                **parse_fmt_hist(format_val, cval)
            )
            self.plot_view.addItem(hist)
            self.hists.append(hist)

    def update_sub(self, data):
        """
        Updates the data in the histogram - none means their was no update for this interval
        """
        if data is not None:
            for index, (hist, data_tup, format_tup) in enumerate(zip(self.hists, arg_inflate_tuple(1, data.bins, data.values, data.formats), self.formats)):
                bins, values, new_format = data_tup
                old_format, cval = format_tup
                if new_format != old_format:
                    self.formats[index] = (new_format, cval)
                    hist.setData(x=bins, y=values, **parse_fmt_hist(new_format, cval))
                else:
                    hist.setData(x=bins, y=values)
        return self.hists


class MultiPlotClient(object):
    def __init__(self, init, framegen, info, rate):
        if init.use_windows:
            self.plots = [type_getter(type(data_obj))(data_obj, None, info, rate) for data_obj in init.data_con]
        else:
            self.fig_win = pg.GraphicsLayoutWidget()
            self.fig_win.setWindowTitle(init.title)
            self.fig_win.show()
            ratio_calc = window_ratio(config.PYQT_SMALL_WIN, config.PYQT_LARGE_WIN)
            if init.ncols is None:
                self.fig_win.resize(*ratio_calc(init.size, 1))
                self.plots = [type_getter(type(data_obj))(data_obj, None, info, rate, figwin=self.fig_win) for data_obj in init.data_con]
            else:
                self.plots = []
                if not isinstance(init.ncols, int) or not 0 < init.ncols <= init.size:
                    ncols = init.size
                    nrows = 1
                    LOG.warning('Invalid column number specified: %s - Must be a positive integer less than the number of plots: %s',
                                    init.ncols,
                                    init.size)
                else:
                    ncols = init.ncols
                    nrows = math.ceil(init.size/float(init.ncols))
                self.fig_win.resize(*ratio_calc(ncols, nrows))
                for index, data_obj in enumerate(init.data_con):
                    if index > 0 and index % ncols == 0:
                        self.fig_win.nextRow()
                    self.plots.append(type_getter(type(data_obj))(data_obj, None, info, rate, figwin=self.fig_win))
        self.framegen = framegen
        self.rate_ms = rate * 1000
        self.info = info
        self.multi_plot = True

    def update(self, data):
        if data is not None:
            for plot, plot_data in zip(self.plots, data.data_con):
                plot.update(plot_data)

    def animate(self):
        self.ani_func()

    def ani_func(self):
        # call the data update function
        self.update(self.framegen.next())
        # setup timer for calling next update call
        QtCore.QTimer.singleShot(self.rate_ms, self.ani_func)
