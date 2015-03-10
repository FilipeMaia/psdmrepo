#!/usr/bin/env python
#--------------------

import os
import sys
import numpy as np
    
#--------------------

def mask_cspad_2x1_edges() :
    """Returns CSPAD array of shape (5920, 388) with mask of 2x1 edges
    """
    segs, rows, cols, colsh = 32, 185, 388, 388/2
    zero_col = np.zeros(rows,dtype=np.int)
    zero_row = np.zeros(cols,dtype=np.int)
    mask2x1  = np.ones((rows,cols),dtype=np.int)
    mask2x1[0, :]      = zero_row # mask top    edge
    mask2x1[-1,:]      = zero_row # mask bottom edge
    mask2x1[:, 0]      = zero_col # mask left   edge
    mask2x1[:,-1]      = zero_col # mask right  edge
    mask2x1[:,colsh-1] = zero_col # mask central-left  column
    mask2x1[:,colsh]   = zero_col # mask central-right column
    #print 'mask2x1:\n', mask2x1
    
    return np.vstack([mask2x1 for seg in range(segs)])

#--------------------

if __name__ == '__main__' :

    mask = mask_cspad_2x1_edges()
    print 'mask:\n', mask
    print 'mask.shape:', mask.shape

    fname = 'cspad_arr_mask_2x1_edges'
    print 'Save file %s.txt' % fname
    np.savetxt(fname+'.txt', mask, fmt='%d')
    print 'Save file %s.npy' % fname
    np.save   (fname+'.npy', mask)

    sys.exit('The End')

#--------------------
