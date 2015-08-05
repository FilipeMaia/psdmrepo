
#!/usr/bin/env python
#--------------------

import numpy as np
import matplotlib.pyplot as plt
import sys
import math

#---------------------
# Define graphical methods

def plot_image (arr, img_range=None, zrange=None, title='',figsize=(12,12), dpi=80) :    # img_range = (left, right, low, high), zrange=(zmin,zmax)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='w',edgecolor='w',frameon=True)
    fig.subplots_adjust(left=0.10, bottom=0.08, right=0.98, top=0.92, wspace=0.2, hspace=0.1)
    figAxes = fig.add_subplot(111)
    imAxes = figAxes.imshow(arr, origin='upper', interpolation='nearest', aspect='auto', extent=img_range)
    if zrange != None : imAxes.set_clim(zrange[0],zrange[1])
    colbar = fig.colorbar(imAxes, pad=0.03, fraction=0.04, shrink=1.0, aspect=40, orientation='horizontal')
    fig.canvas.set_window_title(title)

#def plot_histogram(arr, amp_range=None, figsize=(6,6), bins=100) :
#    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
#    axhi = fig.add_axes([0.14, 0.06, 0.81, 0.90])
#    weights, bins, patches = axhi.hist(arr.flatten(), bins, range=amp_range)
#    add_stat_text(axhi, weights, bins)

def plot_histogram(arr, amp_range=None, figsize=(6,6), bins=40) :
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    axhi = fig.add_axes([0.15, 0.1, 0.8, 0.85])
    weights, binedges, patches = axhi.hist(arr.flatten(), bins=bins, range=amp_range)
    axhi.set_xlim([binedges[0],binedges[-1]]) 
    add_stat_text(axhi, weights, binedges)


    #fig.canvas.manager.window.move(500,10)

def saveHRImageInFile(arr, ampRange=None, fname='cspad-arr-hr.png', figsize=(12,12), dpi=300) :
    print 'SAVE HIGH RESOLUTION IMAGE IN FILE', fname
    plot_image(arr, zrange=ampRange, figsize=figsize, dpi=dpi)
    title = ''
    for q in range(4) : title += ('Quad %d'%(q) + 20*' ')  
    plt.title(title,color='b',fontsize=20)
    plt.savefig(fname,dpi=dpi)
    #plt.imsave('test.png', format='png',dpi=300)

#---------------------

def add_stat_text(axhi, weights, bins) :
    #mean, rms, err_mean, err_rms, neff = proc_stat(weights,bins)
    mean, rms, err_mean, err_rms, neff, skew, kurt, err_err = proc_stat(weights,bins)
    pm = r'$\pm$' 
    txt  = 'Mean=%.2f%s%.2f\nRMS=%.2f%s%.2f\n' % (mean, pm, err_mean, rms, pm, err_rms)
    txt += r'$\gamma1$=%.3f  $\gamma2$=%.3f' % (skew, kurt)
    #txt += '\nErr of err=%8.2f' % (err_err)
    xb,xe = axhi.get_xlim()     
    yb,ye = axhi.get_ylim()     
    x = xb + (xe-xb)*0.8
    y = yb + (ye-yb)*0.86
    axhi.text(x, y, txt, fontsize=12, color='k', ha='center', rotation=0)

#---------------------

def proc_stat(weights, bins) :
    center = np.array([0.5*(bins[i] + bins[i+1]) for i,w in enumerate(weights)])

    sum_w  = weights.sum()
    if sum_w == 0 : return  0, 0, 0, 0, 0, 0, 0, 0
    
    sum_w2 = (weights*weights).sum()
    neff   = sum_w*sum_w/sum_w2
    sum_1  = (weights*center).sum()
    mean = sum_1/sum_w
    d      = center - mean
    d2     = d * d
    wd2    = weights*d2
    m2     = (wd2)   .sum() / sum_w
    m3     = (wd2*d) .sum() / sum_w
    m4     = (wd2*d2).sum() / sum_w

    #sum_2  = (weights*center*center).sum()
    #err2 = sum_2/sum_w - mean*mean
    #err  = math.sqrt(err2)

    rms  = math.sqrt(m2)
    rms2 = m2
    
    err_mean = rms/math.sqrt(neff)
    err_rms  = err_mean/math.sqrt(2)    

    skew, kurt, var_4 = 0, 0, 0

    if rms>0 and rms2>0 :
        skew  = m3/(rms2 * rms) 
        kurt  = m4/(rms2 * rms2) - 3
        var_4 = (m4 - rms2*rms2*(neff-3)/(neff-1))/neff
    err_err = math.sqrt( math.sqrt( var_4 ) )
    #print  'mean:%f, rms:%f, err_mean:%f, err_rms:%f, neff:%f' % (mean, rms, err_mean, err_rms, neff)
    #print  'skew:%f, kurt:%f, err_err:%f' % (skew, kurt, err_err)
    return mean, rms, err_mean, err_rms, neff, skew, kurt, err_err

#---------------------
#---------------------
#---------------------
