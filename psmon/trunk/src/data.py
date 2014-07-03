class Data(object):
    def __init__(self, ts, title, xlabel, ylabel):
        self.ts = ts
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel


class MultiPlot(object):
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


class Image(Data):
    """
    A data container for image data
    """

    def __init__(self, ts, title, image, xlabel=None, ylabel=None):
        super(Image, self).__init__(ts, title, xlabel, ylabel)
        self.image = image


class Hist(Data):
    """
    A data container for 1-d histogram data
    """

    def __init__(self, ts, title, bins, values, xlabel=None, ylabel=None, formats='.'):
        super(Hist, self).__init__(ts, title, xlabel, ylabel)
        self.bins = bins
        self.values = values
        self.formats = formats


class XYPlot(Data):

    def __init__(self, ts, title, xdata, ydata, xlabel=None, ylabel=None, formats='-'):
        super(XYPlot, self).__init__(ts, title, xlabel, ylabel)
        self.xdata = xdata
        self.ydata = ydata
        self.formats = formats
