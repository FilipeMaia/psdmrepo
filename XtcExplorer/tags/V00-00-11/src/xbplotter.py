import numpy as np
import matplotlib.pyplot as plt

def draw_on(fignr):
    fig = plt.figure(num=200)
    axes = fig.add_subplot(111)
    axes.set_title("Hello MatPlotLib")
    
    plt.show()
    
    dark_image = np.load("pyana_cspad_average_image.npy")
    axim = plt.imshow( dark_image )#, origin='lower' )
    colb = plt.colorbar(axim,pad=0.01)
    
    plt.draw()
    
    print "Done drawing"
    
    axim = plt.imshow( dark_image[500:1000,1000:1500] )#, origin='lower' )
    
    return fig

def draw_on_simple(fignr):
    fig = plt.figure(num=200)
    axes = fig.add_subplot(111)
    axes.set_title("Hello MatPlotLib")
    
    plt.show()
    
    dark_image = np.load("pyana_cspad_average_image.npy")
    axim = plt.imshow( dark_image )#, origin='lower' )
    colb = plt.colorbar(axim,pad=0.01)
    
    plt.draw()
    
    print "Done drawing"
    
    axim = plt.imshow( dark_image[500:1000,1000:1500] )#, origin='lower' )
    
    return fig
