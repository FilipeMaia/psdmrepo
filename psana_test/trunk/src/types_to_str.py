import ctypes
import numpy as np
import sys

dll = ctypes.cdll.LoadLibrary('libpsana_test.so')
psana_test_adler32 = dll.psana_test_adler32
psana_test_adler32.restype = ctypes.c_ulong
psana_test_adler32.argtypes = [ctypes.c_void_p, ctypes.c_ulong]

def doIndent(indent,lvl):
    return ((' '*indent)*lvl)

def str_to_str(v,indent=0, lvl=0):
    return doIndent(indent,lvl) + str(v)

# decimal for signed inteters
def int8_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl)+str(v)

def int16_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl)+str(v)

def int32_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl)+str(v)

def enum_to_str(v, indent=0, lvl=0, methodsep=''):
    return int32_to_str(v,indent,lvl)

def int64_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl)+str(v)

# hex for unsigned integers
def uint8_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl) + ('0x%X' % v)

def uint16_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl) + ('0x%X' % v)

def uint32_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl) + ('0x%X' % v)

def uint64_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl) + ('0x%X' % v)

def float_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl) + ("%.4e" % v)

def double_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl) + ("%.4e" % v)

def getQuantileIndicies(n):
    return [0, 
            max(0,min(n-1, int(float(n)/4.0))),
            max(0,min(n-1, int(float(n)/2.0))),
            max(0,min(n-1, int(float(3*n)/4.0))),
            max(0,n-1)]

def getTypeFn(dt):
    if dt == np.uint8:
        return uint8_to_str
    elif dt == np.int8:
        return int8_to_str
    elif dt == np.int16:
        return int16_to_str
    elif dt == np.uint16:
        return uint16_to_str
    elif dt == np.uint32:
        return uint32_to_str
    elif dt == np.int32:
        return int32_to_str
    elif dt == np.uint64:
        return uint64_to_str
    elif dt == np.int64:
        return int64_to_str
    elif dt == np.float32:
        return float_to_str
    elif dt == np.float64:
        return double_to_str
    return None

def ndarray_to_str(a, indent=0, lvl=0):
    dimstr = ' x '.join(map(str,a.shape))
    dim = len(a.shape)
    numElem = 1
    for dim in a.shape:
        numElem *= dim
    outstr = doIndent(indent,lvl)
    outstr +=  "ndarray_%s_%d: dim=[ %s ]" % (a.dtype.name, dim, dimstr)
    if numElem == 0:
        return outstr

    numBytesForCheckSum = numElem * a.dtype.itemsize
    a_flat = a.flatten()
    assert a_flat.flags['C_CONTIGUOUS']
    assert a_flat.shape[0] == numElem
    adler = psana_test_adler32(a_flat.ctypes.get_as_parameter(),
                               ctypes.c_uint64(numBytesForCheckSum))
    outstr += (" adler32=%s" % uint64_to_str(adler))
    a_flat.sort()
    qinds = getQuantileIndicies(a_flat.shape[0])
    type2str = getTypeFn(a_flat.dtype)
    outstr += " min=" + type2str(a_flat[qinds[0] ])
    outstr += " 25th=" + type2str(a_flat[qinds[1] ])
    outstr += " median=" + type2str(a_flat[qinds[2] ])
    outstr += " 75th=" + type2str(a_flat[qinds[3] ])
    outstr += " max=" + type2str(a_flat[qinds[4] ])
    return outstr

