# you can view these plots with a command similar to:
# "psplot -s psanacs057 TEST MULTI"

import psana
from psmon import publish
from psmon.plots import XYPlot, MultiPlot
import numpy as np
import time

def main():
    publish.init()

    x = np.linspace(0, np.pi*2, 100)
    for i in range(100):

        y = np.sin(x + np.pi/10*i)

        multi = MultiPlot(i, 'Some Plots')
        sinPlot = XYPlot(i, 'a sine plot', x, y, formats='bs')
        multi.add(sinPlot)
        multi.add(sinPlot)
        publish.send('TEST', sinPlot)
        publish.send('MULTI', multi)
        time.sleep(0.4)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print '\nExitting script!'
