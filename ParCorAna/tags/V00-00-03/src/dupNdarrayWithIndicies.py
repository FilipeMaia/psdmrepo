import numpy as np
import psana
from PsanaUtil import psanaNdArrays

'''psana module to track how pixels in ndarrays get set in 
image producer 2D arrays.
'''

def label(ndarray, dim):
    rank = len(ndarray.shape)
    assert dim >=0 and dim < rank
    if rank == 2:
        for row in range(ndarray.shape[0]):
            for col in range(ndarray.shape[1]):
                index = [row, col]
                ndarray[row,col]= 10 + index[dim]
    elif rank == 3:
        for row in range(ndarray.shape[0]):
            for col in range(ndarray.shape[1]):
                for subcol in range(ndarray.shape[2]):
                    index = [row, col, subcol]
                    ndarray[row,col, subcol]= 10 + index[dim]
    elif rank == 4:
        for row in range(ndarray.shape[0]):
            for col in range(ndarray.shape[1]):
                for subcol in range(ndarray.shape[2]):
                    for subsubcol in range(ndarray.shape[3]):
                        index = [row, col, subcol, subsubcol]
                        ndarray[row,col, subcol, subsubcol]= 10 + index[dim]
    else:
        raise Exception("can't handle rank=%d" % rank)

class dupNdarrayWithIndicies(object):
    '''Looks for the first ndarray (of any type) with the given src/key.
    Puts R new ndarrays in the event store, where R is the rank of the 
    ndarray that was found. These ndarrays will have _dim_r added to their
    keys, where r is dimensions, 0,1,2, etc. The type will be ndarray_int32_x where x is the
    dimension.
    The values of these arrays will be the 10 + dimension index, i.e::

      10+row for dim_0
      10+col for dim_1
      10+subcol for dim_2, etc.

    The +10 will distinguish them from border pixels in the image.
    
    One can then identify the mapping from ndarray coordinates to image coordinates.
    For example, if for key calibrated a 3D (32,185,388) array is found, 3 arrays 
    will be added to the event::

      calibrated_dim_0:  values between 10 and 41
      calibrated_dim_1:  values between 10 and 194
      calibrated_dim_2:  values between 10 and 397

    If three image producers are run on these three keys, you can look for anything that is 10 
    or higher, and subtract 10 to recover the dim 0, 1, 2 that was used for a given row,col
    '''
    def beginjob(self,evt,env):
        self.src = self.configSrc('src')
        self.key = self.configStr('key')
        self.done = False

    def event(self,evt,env):
        if self.done: return
        for ndarrayType in psanaNdArrays:
            ndarray = evt.get(ndarrayType, self.src, self.key)
            if ndarray is None: continue
            rank = len(ndarray.shape)
            for dim in range(rank):
                ndarray_dim = np.zeros(ndarray.shape, dtype=np.int32)
                label(ndarray_dim, dim)
                evt.put(ndarray_dim, self.src, self.key + '_dim_%d' % dim)
            self.done = True
            break

            
