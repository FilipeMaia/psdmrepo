"""
Histogram

General purpose histogram classes. 
"""


# Histogramming classes
class histaxis:
    def __init__(self,nbin,low,high):
        self.low = low
        self.high = high
        self.nbin = nbin
        self.binsize = (high-low)/float(nbin)
    def bin(self,val):
        return int(math.floor((val-self.low)/self.binsize))

class hist1d:
    def __init__(self,nbinx,xlow,xhigh):
        self.data = np.zeros(nbinx)
        self.nbinx = nbinx
        self.xaxis = histaxis(nbinx,xlow,xhigh)
    def fill(self,xval,weight=1.0):
        xbin=self.xaxis.bin(xval)
        if xbin>=0 and xbin<self.xaxis.nbin:
            self.data[xbin] += weight
