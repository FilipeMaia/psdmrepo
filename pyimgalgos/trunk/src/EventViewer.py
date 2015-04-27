#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import pyimgalgos.GlobalGraphics as gg
#import matplotlib.patches as patches # for patches.Circle
#from time import sleep

import psana

#------------------------------

def list_of_times_default(run):
    print 'Default list of times for entire run'
    return run.times()

#------------------------------

def fig_axes(figsize=(13,12), title='Image', dpi=80, \
             win_axim=(0.05,  0.03, 0.87, 0.93), \
             win_axcb=(0.923, 0.03, 0.02, 0.93)) :
    """ Creates and returns figure, and axes for image and color bar
    """
    fig  = plt.figure(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='w', frameon=True)
    axim = fig.add_axes(win_axim)
    axcb = fig.add_axes(win_axcb)
    fig.canvas.set_window_title(title)
    return fig, axim, axcb

#------------------------------

class EventViewer :
    list_of_dtypes = [psana.ndarray_float32_2, \
                      psana.ndarray_float64_2, \
                      psana.ndarray_int16_2, \
                      psana.ndarray_int32_2, \
                      psana.ndarray_int64_2, \
                      psana.ndarray_uint8_2, \
                      psana.ndarray_uint16_2, \
                      psana.ndarray_uint32_2, \
                      psana.ndarray_uint64_2]


    def __init__(self, run, list_of_times=None, src=None, key=None):

        self.count_msg = 0
        self.plot_first(run, list_of_times, src, key)

#------------------------------

    def plot_first(self, run, list_of_times=None, src=None, key=None) :

        self.fig, self.axim, self.axcb = fig_axes() # or gg.fig_axes()
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.run = run
        self.src = src
        self.key = key
        self.list_times = list_of_times_default(run) if list_of_times is None else list_of_times
        self.list_ind_time = [(i,t) for i,t in enumerate(self.list_times)]
        self.list_index = 0
        self.list_len = len(self.list_ind_time)
        self.plot_current('first')

#------------------------------

    def plot_current(self, cmt='current'):

        i,t = self.list_ind_time[self.list_index]
        tit = 'Event in the list %d (of %d), in run: %d, fiducial: %d' % \
              (self.list_index, self.list_len, i, t.fiducial())
        #print 'Plot %s %s,' % (cmt, tit) 
        #plt.title("Fid: %d" % (t.fiducial()))
        self.fig.canvas.set_window_title(tit)
        evt = self.run.event(t)

        #print 80*"=", '\n', evt.keys()

        img = self.get_img(evt, self.src, self.key)

        self.plot_img(img)

#------------------------------

    def plot_next(self):
        self.list_index += 1
        if self.list_index >= self.list_len :
            self.list_index = self.list_len-1
            print 'Counter reached the last event in the list'
            return    
        self.plot_current('next')

#------------------------------

    def plot_previous(self):
        self.list_index -= 1
        if self.list_index < 0 :
            self.list_index = 0
            print 'Counter reached the first event in the list'
            return
        self.plot_current('previous')

#------------------------------

    def on_key_press(self, event):
        """ Switch for input control key
        """    
        #print 'You pressed: ', event.key #, event.xdata, event.ydata
    
        if   event.key == 'escape'  \
          or event.key == 'e' :
            self.fig.canvas.mpl_disconnect(self.cid)
            sys.exit('The End')

        elif event.key == 'left'  \
          or event.key == 'down'  \
          or event.key == 'b' :
            self.plot_previous()
            return

        elif event.key == 'right' \
          or event.key == 'up'    \
          or event.key == 'n':
            self.plot_next()
            return

        else :
            self.print_help()

#------------------------------

    def plot_img(self, img) :
    
        fig, axim, axcb = self.fig, self.axim, self.axcb
    
        axim.cla()
        axcb.cla()

        imsh = axim.imshow(img, interpolation='nearest', aspect='auto', origin='upper') # extent=img_range)
        colb = fig.colorbar(imsh, cax=axcb) # , orientation='horizontal')

        ave = np.mean(img)
        rms = np.std(img)
        #print 'img ave = %f, rms = %f' % (ave, rms)
        imsh.set_clim(ave-1*rms,ave+5*rms)
        #imsh.set_clim(0,200)

        self.print_help(1)

        fig.canvas.draw()

        plt.ioff()
        plt.show()

#------------------------------

    def print_help(self, mode=None):
        try:    self.click_counter += 1
        except: self.click_counter = 1

        if mode==1 and self.click_counter >1 : return

        print 'Navigation keys: n-next, b-previous, e-exit'

#------------------------------

    def get_img(self, evt, src=None, key=None) :
        """Interface method can be re-implemented in sub-class
        """
        self.count_msg += 1
        if self.count_msg < 5 : print 'EventViewer.get_img(...)'        

        for dtype in self.list_of_dtypes :
            img = evt.get(dtype, src, key)
            if img is not None : return img

        return gg.getRandomImage(mu=200, sigma=25, shape=(100,100))

#------------------------------
#------------------------------
#------------------------------
#------------------------------

import CalibManager.AppDataPath as apputils

def do_test() :

    path_psana_cfg = apputils.AppDataPath('pyimgalgos/scripts/psana-cspad-ds2-image-producer.cfg').path()
    print 'Path to psana cfg file: %s' % path_psana_cfg
    psana.setConfigFile(path_psana_cfg)

    ds = psana.DataSource('exp=cxif5315:run=169:idx')
    run = ds.runs().next()
    src = psana.Source('DetInfo(CxiDs2.0:Cspad.0)')
    key ='cspad_img'
    
    list_of_times = None # list_of_times_selected(run)
    EventViewer(run, list_of_times, src, key)
    
#------------------------------

if __name__ == '__main__' :
    proc_name = os.path.basename(sys.argv[0])

    do_test()

    sys.exit('The End')

#------------------------------
