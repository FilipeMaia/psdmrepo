import time
import numpy as np
from itertools import chain, izip

from psmon.plots import Image, MultiPlot, Hist, XYPlot


def is_py_iter(obj):
    """
    Check if the object is an iterable python object excluding ndarrays
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, np.ndarray)


def arg_inflate(index, *args):
    args = list(args)
    for i in range(len(args)):
        if i == index:
            continue
        if not is_py_iter(args[i]):
            args[i] = [args[i]] * len(args[index])
    return args


def arg_inflate_flat(index, *args):
    if is_py_iter(args[index]):
        return list(chain.from_iterable(izip(*arg_inflate(index, *args))))
    else:
        return args


def arg_inflate_tuple(index, *args):
    if is_py_iter(args[index]):
        return zip(*arg_inflate(index, *args))
    else:
        return [args]


def inflate_input(input, input_ref):
    if is_py_iter(input_ref):
        return arg_inflate(1, input, input_ref)[0]
    else:
        return [input]


def make_bins(nbins, bmin, bmax):
    step = (bmax - bmin)/nbins
    return np.arange(bmin, bmax + step, step)[:nbins+1]


def window_ratio(min_res, max_res):
    def window_ratio_calc(ncols, nrows):
        pref_x = min_res.x * ncols
        pref_y = min_res.y * nrows

        if pref_x > max_res.x or pref_y > max_res.y:
            num = min(max_res.x/float(ncols), max_res.y/float(nrows))
            pref_x = max(ncols * num, min_res.x)
            pref_y = max(nrows * num, min_res.y)

        return int(pref_x), int(pref_y)
    return window_ratio_calc


class Helper(object):
    def __init__(self, publisher, topic, title=None, pubrate=None):
        self.publisher = publisher
        self.topic = topic
        self.data = None
        self.title = title or self.topic
        self.pubrate = pubrate
        self.__last_pub = time.time()

    def publish(self):
        current_time = time.time()
        if self.pubrate is None or self.pubrate * (current_time - self.__last_pub) >= 1:
            self.__last_pub = current_time
            self.publisher(self.topic, self.data)


class MultiHelper(Helper):
    def __init__(self, publisher, topic, num_data, title=None, pubrate=None):
        super(MultiHelper, self).__init__(publisher, topic, title, pubrate)
        self.data = MultiPlot(None, self.title, [None] * num_data)

    def set_data(self, index, type, *args, **kwargs):
        self.data.data_con[index] = type(*args, **kwargs)


class ImageHelper(Helper):
    def __init__(self, publisher, topic, title=None, pubrate=None):
        super(ImageHelper, self).__init__(publisher, topic, title, pubrate)
        self.data = Image(None, self.title, None)

    def set_image(self, image, image_title=None):
        if image_title is not None:
            self.data.ts = image_title
        self.data.image = image


class MultiImageHelper(MultiHelper):
    def __init__(self, publisher, topic, num_image, title=None, pubrate=None):
        super(MultiImageHelper, self).__init__(publisher, topic, num_image, title=None, pubrate=None)

    def set_image(self, index, image, image_title=None):
        self.set_data(index, Image, image_title, None, image)


class XYPlotHelper(Helper):
    DEFAULT_ARR_SIZE = 100

    def __init__(self, publisher, topic, title=None, xlabel=None, ylabel=None, format='-', pubrate=None):
        super(XYPlotHelper, self).__init__(publisher, topic, title, pubrate)
        self.index = 0
        self.xdata = np.zeros(XYPlotHelper.DEFAULT_ARR_SIZE)
        self.ydata = np.zeros(XYPlotHelper.DEFAULT_ARR_SIZE)
        self.data = XYPlot(
            None,
            self.title,
            None,
            None,
            xlabel=xlabel,
            ylabel=ylabel,
            formats=format
        )

    def add(self, xval,  yval, entry_title=None):
        if entry_title is not None:
            self.data.ts = entry_title
        if self.index == self.xdata.size:
            # double the size if we need to reallocate
            self.xdata = np.resize(self.xdata, 2*self.index)
            self.ydata = np.resize(self.ydata, 2*self.index)
        self.xdata[self.index] = xval
        self.ydata[self.index] = yval
        self.index += 1
        self.data.xdata = self.xdata[:self.index]
        self.data.ydata = self.ydata[:self.index]

    def clear(self):
        self.index = 0


class HistHelper(Helper):
    def __init__(self, publisher, topic, nbins, bmin, bmax, title=None, xlabel=None, ylabel=None, format='-', pubrate=None):
        super(HistHelper, self).__init__(publisher, topic, title, pubrate)
        self.nbins = int(nbins)
        self.bmin = float(bmin)
        self.bmax = float(bmax)
        self.range = (bmin, bmax)
        self.data = Hist(
            None,
            self.title,
            make_bins(self.nbins, self.bmin, self.bmax),
            np.zeros(self.nbins),
            xlabel=xlabel,
            ylabel=ylabel,
            formats=format
        )

    def add(self, value, entry_title=None):
        if entry_title is not None:
            self.data.ts = entry_title
        self.data.values += np.histogram(value, self.nbins, range=self.range)[0]

    def clear(self):
        self.data.values[:] = 0


class HistOverlayHelper(Helper):
    def __init__(self, publisher, topic, title=None, xlabel=None, ylabel=None, pubrate=None):
        super(HistOverlayHelper, self).__init__(publisher, topic, title, pubrate)
        self.nhist = 0
        self.nbins = []
        self.ranges = []
        self.bins = []
        self.values = []
        self.formats = []
        self.data = Hist(
            None,
            self.title,
            self.bins,
            self.values,
            xlabel=xlabel,
            ylabel=ylabel,
            formats=self.formats
        )

    def make_hist(self, nbins, bmin, bmax, format='-'):
        index = self.nhist
        self.nbins.append(nbins)
        self.ranges.append((bmin, bmax))
        self.bins.append(make_bins(nbins, bmin, bmax))
        self.values.append(np.zeros(nbins))
        self.formats.append(format)
        self.nhist += 1
        return index

    def add(self, index, value, entry_title=None):
        if entry_title is not None:
            self.data.ts = entry_title
        self.values[index] += np.histogram(value, self.nbins[index], range=self.ranges[index])[0]

    def clear(self, index=None):
        if index is None:
            for value in self.values:
                value[:] = 0
        else:
            self.values[index][:] = 0
