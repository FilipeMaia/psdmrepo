class Data(object):
    def __init__(self, ts, title, xlabel, ylabel):
        self.ts = ts
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel


class MultiData(object):
    def __init__(self, ts, title, data_con=None):
        self.ts = ts
        self.title = title
        if data_con is None:
            self.data_con = []
        else:
            self.data_con = data_con

    def add(self, data):
        self.data_con.append(data)

    def get(self, index):
        return self.data_con[index]

    @property
    def size(self):
        return len(self.data_con)


class ImageData(Data):
    """
    A data container for image data
    """

    def __init__(self, ts, title, image, xlabel=None, ylabel=None):
        super(ImageData, self).__init__(ts, title, xlabel, ylabel)
        self.image = image


class HistData(Data):
    """
    A data container for 1-d histogram data
    """

    def __init__(self, ts, title, bins, values, xlabel=None, ylabel=None, formats='.'):
        super(HistData, self).__init__(ts, title, xlabel, ylabel)
        self.bins = bins
        self.values = values
        self.formats = formats


class XYPlotData(Data):

    def __init__(self, ts, title, xdata, ydata, xlabel=None, ylabel=None, formats='-'):
        super(XYPlotData, self).__init__(ts, title, xlabel, ylabel)
        self.xdata = xdata
        self.ydata = ydata
        self.formats = formats
