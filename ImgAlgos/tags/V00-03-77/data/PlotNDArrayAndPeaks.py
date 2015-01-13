#!/usr/bin/env python
#--------------------
import os
import sys
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # for patches.Circle
# import pyimgalgos.GlobalGraphics as gg # move, show, etc.

#--------------------
from PSCalib.GeometryAccess import GeometryAccess, img_from_pixel_arrays
from PSCalib.GeometryObject import data2x2ToTwo2x1
#--------------------

class Storage :
    def __init__(self) :
        print 'Storage object is created in order to pass common parameters between methods.'
        self.iX = None

#--------------------
store = Storage() # singleton
#--------------------

def plot_peaks_for_arr(arr_peaks, axim, color='w') :  
    """ Plots peaks as circles in coordinates of 2-d shaped ndarray
    """
    anorm = np.average(arr_peaks,axis=0)[4]
    #print 'ampmax=', ampmax

    for peak in arr_peaks :
        s, r, c, amax, atot, npix = peak
        #print 's, r, c, amax, atot, npix=', s, r, c, amax, atot, npix
        x=c
        y=s*185 + r
        xy0 = (x,y)
        r0 = 4
        #r0 = 2+2*atot/amnorm
        circ = patches.Circle(xy0, radius=r0, linewidth=2, color=color, fill=False)
        axim.add_artist(circ)

#--------------------

def plot_peaks_for_img(arr_peaks, axim, color='w') :  
    """ Plots peaks as circles in coordinates of image
    """
    iX, iY = store.iX, store.iY

    #anorm = np.average(arr_peaks,axis=0)[4]
    #print 'ampmax=', ampmax

    for peak in arr_peaks :
        s, r, c, amax, atot, npix = peak
        #print 's, r, c, amax, atot, npix=', s, r, c, amax, atot, npix
        x=iX[s,int(r),int(c)]
        y=iY[s,int(r),int(c)]
        #print ' x,y=',x,y        
        xyc = (y,x)
        r0  = 4
        #r0  = 2+2*atot/anorm
        circ = patches.Circle(xyc, radius=r0, linewidth=2, color=color, fill=False)
        axim.add_artist(circ)

#--------------------

def print_peaks(arr_peaks) :  
    # arr_peaks : 1         4         311       32.999    32.999    1
    for peak in arr_peaks :
        s, r, c, amax, atot, npix = peak
        print 's:%2d  r:%3d  c:%3d  amax:%8.1f  atot:%8.1f  npix:%2d' % \
               (s, r, c, amax, atot, npix) 

#--------------------

def get_array_from_file(fname) :
    """ Loads and returns numpy array from file
    """    
    if store.verbos : print 'get_array_from_file:', fname
    arr = np.loadtxt(fname, dtype=np.float32)

    if len(arr.shape)<2 : return [arr]
    return arr

#--------------------
#--------------------
#--------------------
#--------------------
#--------------------

def list_of_files_as(ifname='./xxx-r0081-e000002-raw.txt') :
    """This module makes the list of files like ./xxx-r0081-*-raw.txt
    """
    print 'input file name: %s' % ifname

    dname = os.path.dirname(ifname)
    dname = './' if dname=='' else dname
    bname  = os.path.basename(ifname)
    pref, medi, suff = bname.rsplit('-',2)
    print 'Splited fields: ', dname, pref, medi, suff

    lst = os.listdir(dname)
    #if patrn is None :
    #    return lst
    lst_sel = [os.path.join(dname,fname) for fname in lst if pref in fname and suff in fname]
    return sorted(lst_sel)

#--------------------

def list_of_files_in_dir(dname='./', patrn=None) :
    """ Returns a list of files in the directory for with names containing pattern
    """    
    lst = os.listdir(dname)
    if patrn is None :
        return lst
    lst_sel = [fname for fname in lst if patrn in fname]
    return sorted(lst_sel)

#--------------------

def print_list_of_files(dname='./', patrn=None) :
    """ Prints a list of files in the directory for with names containing pattern
    """    
    print 'List of files in the dir.', dname
    for name in list_of_files_in_dir(dname, patrn) :
        print name
    print '\n'

#--------------------

def fig_axes(figsize=(12,10), title='Default-title') :
    """ Creates and returns figuure, and axes for image and color bar
    """
    fig  = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    axes = fig.add_axes([0.05,  0.03, 0.88, 0.94])
    axcb = fig.add_axes([0.93,  0.03, 0.02, 0.94])
    fig.canvas.set_window_title(title)
    return fig, axes, axcb

#--------------------

def intensity_range() :
    """ Returns the intensity range from input pars or None if they are not defined
    """
    if store.amplo is None \
    or store.amphi is None :
        return None
    return store.amplo, store.amphi

#--------------------

def is_cspad2x2(geometry) :
    """ Check if the geometry file for CSPAD2X2
    """    
    if geometry.get_geo('CSPAD2X2:V1',0) is None :
        return False
    return True

#--------------------

def nda_3d_shape(nda) :
    """If ndarray has more than 2 dimensions, the shape is returned for 3-d 
    """
    if len(nda.shape) > 2 :
        rows, cols = nda.shape[-2:]
        return (nda.size/rows/cols, rows, cols)
    return nda.shape    
    #return (2, 185, 388)

#--------------------

def get_index_arrays(gfname) :
    """ Reconstruct image from input array and geometry file
    """
    if store.iX is None :

        print 'It would be nice to reconstruct image for available geometry file:\n  %s' % gfname
        geometry = GeometryAccess(gfname, 0377)
        iX_asdata, iY_asdata = geometry.get_pixel_coord_indexes()
        print 'Geometry array original shape for iX, iY:', str(iX_asdata.shape), str(iY_asdata.shape)
        
        if is_cspad2x2(geometry) :
            # case of cspad2x2
            iX_asdata.shape = (185, 388, 2)
            iY_asdata.shape = (185, 388, 2)
        
            store.iX = data2x2ToTwo2x1(iX_asdata)
            store.iY = data2x2ToTwo2x1(iY_asdata)
        else :
            # all other detectors
            store.iX = iX_asdata
            store.iY = iY_asdata
        
        print 'iX, iY shapes:', str(store.iX.shape), str(store.iY.shape)
        
        shape_3d = nda_3d_shape(store.iX)
        print '3-d shape of ndarrays:', str(shape_3d)
        
        store.iX.shape = store.iY.shape = shape_3d

    return store.iX, store.iY

#--------------------

def get_image(arr, gfname) :
    """ Reconstruct image from input array and geometry file
    """
    iX, iY = get_index_arrays(gfname)
    if store.verbos : print 'Original arr.shape: %s' % str(arr.shape),
    arr.shape = iX.shape
    img = img_from_pixel_arrays(iX,iY,W=arr)
    if store.verbos : print '  img.shape:', str(img.shape)
    return img

#--------------------

def plot_one(fname) :
    """ Plot image from file and peaks from associated file
    """    
    print 'Plot for file: %s' % fname

    fig, axim, axcb = store.fig, store.axim, store.axcb
    gfname, verbos, colring = store.gfname, store.verbos, store.color

    arr = get_array_from_file(fname)
    arrim = arr if store.gfname is None else get_image(arr, gfname)
    
    axim.cla()
    imsh = axim.imshow(arrim, interpolation='nearest', aspect='auto', origin='upper') # extent=img_range)

    #colb = fig.colorbar(imsh, pad=0.005, fraction=0.09, shrink=1, aspect=40)
    colb = fig.colorbar(imsh, cax=axcb) # , orientation='horizontal')

    amp_range = intensity_range()

    if amp_range is None :
        rms = np.std(arrim)
        if verbos : print 'arrim.rms = %f' % rms
        imsh.set_clim(-3*rms,6*rms)
    else :
        imsh.set_clim(amp_range[0],amp_range[1])

    fname_peaks = fname.rsplit('-',1)[0] + '-peaks.txt'
    if os.path.lexists(fname_peaks) :
        arr_peaks = get_array_from_file(fname_peaks)

        if arr_peaks is not None :

            if verbos : print_peaks(arr_peaks)

            if store.gfname is None :
                plot_peaks_for_arr(arr_peaks, axim, color=colring)
            else :
                plot_peaks_for_img(arr_peaks, axim, color=colring)

    else :
        print 'File with peaks: %s does not exist' % fname_peaks        

    fig.canvas.set_window_title('File: %s'%fname)
    fig.canvas.draw()

#--------------------

def plot_next() :
    """ Plot image from the next file
    """    
    store.count += 1
    if store.count >= store.lst_len :
        store.count = store.lst_len-1
        print 'Counter reached the last file in the list: %s' % store.lst[store.count]
        return
    #print 'plot_next(): %d' % store.count
    plot_one(store.lst[store.count])

#--------------------

def plot_previous() :
    """ Plot image from previous file
    """    
    store.count -= 1
    if store.count < 0 :
        store.count = 0 
        print 'Counter reached the first file in the list: %s' % store.lst[store.count] 
        return

    #print 'plot_previous: %d' % store.count
    plot_one(store.lst[store.count])
   
#--------------------

def on_key_press(event):
    """ Switch for input control key
    """
    if store.verbos : print 'You pressed: ', event.key #, event.xdata, event.ydata

    if   event.key == 'escape'  \
      or event.key == 'e' : sys.exit('The End')

    elif event.key == 'left'  \
      or event.key == 'down'  \
      or event.key == 'b'     : plot_previous()

    elif event.key == 'right' \
      or event.key == 'up'    \
      or event.key == 'n'     : plot_next()

    else                      : plot_next()
        
#--------------------
#def on_click(event):
#    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
#        event.button, event.x, event.y, event.xdata, event.ydata)
#--------------------
#def input_char() :
#    key = raw_input('"e"-end or click "Enter" for next event > ')
#    return key
#--------------------

def move_window(x0=200,y0=100) :
    move_str = '+' + str(x0) + '+' + str(y0)
    plt.get_current_fig_manager().window.geometry(move_str)
    #plt.get_current_fig_manager().window.geometry("+50+50")

#--------------------
                    
def do_plots() :
    """ Plot image and update using keybord control keys
    """
    store.fig, store.axim, store.axcb = fig_axes()
    ifname = store.ifname

    #store.lst = list_of_files_in_dir(dname, patrn)

    store.lst = list_of_files_as(ifname)

    print 'List of files to plot:'
    for fname in store.lst : print fname

    store.lst_len = len(store.lst)
    store.count = 0

    if store.lst_len < 1 :
        sys.exit('The list of input files is empty')
        
    cid = store.fig.canvas.mpl_connect('key_press_event', on_key_press)

    plot_one(store.lst[store.count])
    print 'Control buttons: arrows or n/b - next/previous file, e/Escape-exit, s-save, f-full screen, p-pan/zoom'
    move_window(500,10)
    plt.show()

    store.fig.canvas.mpl_disconnect(cid)

#--------------------
#--------------------
#--------------------
#--------------------

def do_plot_one() :
    """ Plots a single image w/o updates
    """
    store.fig, store.axim, store.axcb = fig_axes()
    plot_one(store.ifname)
    move_window(500,10)
    plt.show()

#--------------------

def usage() :
    return '\n\nExamples:' + \
           '  TBA'

#--------------------

from optparse import OptionParser

def input_options_parser() :

    #gfname_def = '/reg/d/psdm/CXI/cxitut13/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'

    gfname_def = None
    gfname_def = '/reg/neh/home1/dubrovin/LCLS/' \
                 'CSPad2x2Alignment/calib-cspad2x2-01-2013-02-13/' \
                 'calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/geometry/1-end.data'
    ifname_def = 'xxx-r0081-e000002-raw.txt'
    color_def  = 'k'
    verbos_def = False
    amplo_def  = None
    amphi_def  = None
    
    parser = OptionParser(description='Optional input parameters.', usage ='usage: %prog [options]' + usage())
    parser.add_option('-g', '--gfname',  dest='gfname', default=gfname_def, action='store', type='string', help='geometry file name, default = %s' % gfname_def)
    parser.add_option('-i', '--ifname',  dest='ifname', default=ifname_def, action='store', type='string', help='image file name, default = %s' % ifname_def)
    parser.add_option('-v', '--verbos',  dest='verbos', default=verbos_def, action='store_true',           help='verbosity, default = %s' % str(verbos_def))
    parser.add_option('-c', '--color',   dest='color',  default=color_def,  action='store', type='string', help='color of rings around peaks, default = %s' % color_def)
    parser.add_option('-L', '--amplo',   dest='amplo',  default=amplo_def,  action='store', type='int',    help='Low intensity limit, default = %s' % str(amplo_def))
    parser.add_option('-H', '--amphi',   dest='amphi',  default=amphi_def,  action='store', type='int',    help='High intensity limit, default = %s' % str(amphi_def))
 
    (opts, args) = parser.parse_args()
    return (opts, args)

#--------------------

if __name__ == '__main__' :

    proc_name = os.path.basename(sys.argv[0])

    #if len(sys.argv)==1 :
    #    print 'Try command: %s -h' % proc_name
    #    sys.exit ('End of %s' % proc_name)
        
    (opts, args) = input_options_parser()

    #if opts.verbos :
    if True :
        print 'Command arguments:', ' '.join(sys.argv)
        print '  opts:\n', opts
        print '  args:\n', args

    store.gfname = opts.gfname
    store.ifname = opts.ifname
    store.verbos = opts.verbos
    store.amplo  = opts.amplo 
    store.amphi  = opts.amphi 
    store.color  = opts.color 

    #if   opts.proc==1 : image_of_sensors   (opts.gfname, opts.afname, opts.ifname, opts.cbits)
    #else : print 'Non-recognized process option; implemented options: -p1, -p2, and -p3'

    #do_plot_one()

    do_plots()

    sys.exit('The End')

#--------------------
