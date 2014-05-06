import ctypes
import numpy as np
import sys
from IPython import embed

dll = ctypes.cdll.LoadLibrary('libpsana_test.so')
psana_test_adler32 = dll.psana_test_adler32
psana_test_adler32.restype = ctypes.c_ulong
psana_test_adler32.argtypes = [ctypes.c_void_p, ctypes.c_ulong]

def doIndent(indent,lvl):
    return ((' '*indent)*lvl)
# decimal for signed inteters
def int8_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl)+str(v)

def int16_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl)+str(v)

def int32_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl)+str(v)

def int64_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl)+str(v)

# hex for unsigned integers
def uint8_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl) + ('0x%2.2x' % v)

def uint16_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl) + ('0x%4.4x' % v)

def uint32_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl) + ('0x%8.8x' % v)

def uint64_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl) + ('0x%16.16x' % v)

def str_to_str(v, indent=0, lvl=0):
    return doIndent(indent,lvl) + str(v)

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

def ndarray_to_str(a, dim, indent=0, lvl=0):
    assert len(a.shape)==dim
    dimstr = ' x '.join(map(str,a.shape))
    n = 1
    for dim in a.shape:
        n *= dim
    n *= a.dtype.itemsize
    a_flat = a.flatten()
    assert a_flat.flags['C_CONTIGUOUS']
    if n > 0:
        adler = psana_test_adler32(a_flat.ctypes.get_as_parameter(),
                                   ctypes.c_uint64(n))
    else:
        adler = 0
    sys.stdout.flush()
    a_flat.sort()
    qinds = getQuantileIndicies(a_flat.shape[0])
    outstr = doIndent(indent,lvl)
    outstr += "dim=[ %s ] adler32=0x%16.16x" % (dimstr, adler)
    type2str = getTypeFn(a_flat.dtype)
    if n > 0:
        outstr += " min=" + type2str(a_flat[qinds[0] ])
        outstr += " 25th=" + type2str(a_flat[qinds[1] ])
        outstr += " median=" + type2str(a_flat[qinds[2] ])
        outstr += " 75th=" + type2str(a_flat[qinds[3] ])
        outstr += " max=" + type2str(a_flat[qinds[4] ])
    return outstr

def ndarray_float32_1_to_str(a, indent=0, lvl=0):
    return "ndarray_float32_1: %s" % ndarray_to_str(a,1,indent,lvl)

def ndarray_float32_2_to_str(a, indent=0, lvl=0):
    return "ndarray_float32_2: %s" % ndarray_to_str(a,2,indent,lvl)

def ndarray_float32_3_to_str(a, indent=0, lvl=0):
    return "ndarray_float32_3: %s" % ndarray_to_str(a,3,indent,lvl)

def ndarray_float32_4_to_str(a, indent=0, lvl=0):
    return "ndarray_float32_4: %s" % ndarray_to_str(a,4,indent,lvl)

def ndarray_float32_5_to_str(a, indent=0, lvl=0):
    return "ndarray_float32_5: %s" % ndarray_to_str(a,5,indent,lvl)

def ndarray_float32_6_to_str(a, indent=0, lvl=0):
    return "ndarray_float32_6: %s" % ndarray_to_str(a,6,indent,lvl)

def ndarray_float64_1_to_str(a, indent=0, lvl=0):
    return "ndarray_float64_1: %s" % ndarray_to_str(a,1,indent,lvl)

def ndarray_float64_2_to_str(a, indent=0, lvl=0):
    return "ndarray_float64_2: %s" % ndarray_to_str(a,2,indent,lvl)

def ndarray_float64_3_to_str(a, indent=0, lvl=0):
    return "ndarray_float64_3: %s" % ndarray_to_str(a,3,indent,lvl)

def ndarray_float64_4_to_str(a, indent=0, lvl=0):
    return "ndarray_float64_4: %s" % ndarray_to_str(a,4,indent,lvl)

def ndarray_float64_5_to_str(a, indent=0, lvl=0):
    return "ndarray_float64_5: %s" % ndarray_to_str(a,5,indent,lvl)

def ndarray_float64_6_to_str(a, indent=0, lvl=0):
    return "ndarray_float64_6: %s" % ndarray_to_str(a,6,indent,lvl)

def ndarray_int16_1_to_str(a, indent=0, lvl=0):
    return "ndarray_int16_1: %s" % ndarray_to_str(a,1,indent,lvl)

def ndarray_int16_2_to_str(a, indent=0, lvl=0):
    return "ndarray_int16_2: %s" % ndarray_to_str(a,2,indent,lvl)

def ndarray_int16_3_to_str(a, indent=0, lvl=0):
    return "ndarray_int16_3: %s" % ndarray_to_str(a,3,indent,lvl)

def ndarray_int16_4_to_str(a, indent=0, lvl=0):
    return "ndarray_int16_4: %s" % ndarray_to_str(a,4,indent,lvl)

def ndarray_int16_5_to_str(a, indent=0, lvl=0):
    return "ndarray_int16_5: %s" % ndarray_to_str(a,5,indent,lvl)

def ndarray_int16_6_to_str(a, indent=0, lvl=0):
    return "ndarray_int16_6: %s" % ndarray_to_str(a,6,indent,lvl)

def ndarray_int32_1_to_str(a, indent=0, lvl=0):
    return "ndarray_int32_1: %s" % ndarray_to_str(a,1,indent,lvl)

def ndarray_int32_2_to_str(a, indent=0, lvl=0):
    return "ndarray_int32_2: %s" % ndarray_to_str(a,2,indent,lvl)

def ndarray_int32_3_to_str(a, indent=0, lvl=0):
    return "ndarray_int32_3: %s" % ndarray_to_str(a,3,indent,lvl)

def ndarray_int32_4_to_str(a, indent=0, lvl=0):
    return "ndarray_int32_4: %s" % ndarray_to_str(a,4,indent,lvl)

def ndarray_int32_5_to_str(a, indent=0, lvl=0):
    return "ndarray_int32_5: %s" % ndarray_to_str(a,5,indent,lvl)

def ndarray_int32_6_to_str(a, indent=0, lvl=0):
    return "ndarray_int32_6: %s" % ndarray_to_str(a,6,indent,lvl)

def ndarray_int64_1_to_str(a, indent=0, lvl=0):
    return "ndarray_int64_1: %s" % ndarray_to_str(a,1,indent,lvl)

def ndarray_int64_2_to_str(a, indent=0, lvl=0):
    return "ndarray_int64_2: %s" % ndarray_to_str(a,2,indent,lvl)

def ndarray_int64_3_to_str(a, indent=0, lvl=0):
    return "ndarray_int64_3: %s" % ndarray_to_str(a,3,indent,lvl)

def ndarray_int64_4_to_str(a, indent=0, lvl=0):
    return "ndarray_int64_4: %s" % ndarray_to_str(a,4,indent,lvl)

def ndarray_int64_5_to_str(a, indent=0, lvl=0):
    return "ndarray_int64_5: %s" % ndarray_to_str(a,5,indent,lvl)

def ndarray_int64_6_to_str(a, indent=0, lvl=0):
    return "ndarray_int64_6: %s" % ndarray_to_str(a,6,indent,lvl)

def ndarray_uint8_1_to_str(a, indent=0, lvl=0):
    return "ndarray_uint8_1: %s" % ndarray_to_str(a,1,indent,lvl)

def ndarray_uint8_2_to_str(a, indent=0, lvl=0):
    return "ndarray_uint8_2: %s" % ndarray_to_str(a,2,indent,lvl)

def ndarray_uint8_3_to_str(a, indent=0, lvl=0):
    return "ndarray_uint8_3: %s" % ndarray_to_str(a,3,indent,lvl)

def ndarray_uint8_4_to_str(a, indent=0, lvl=0):
    return "ndarray_uint8_4: %s" % ndarray_to_str(a,4,indent,lvl)

def ndarray_uint8_5_to_str(a, indent=0, lvl=0):
    return "ndarray_uint8_5: %s" % ndarray_to_str(a,5,indent,lvl)

def ndarray_uint8_6_to_str(a, indent=0, lvl=0):
    return "ndarray_uint8_6: %s" % ndarray_to_str(a,6,indent,lvl)

def ndarray_uint16_1_to_str(a, indent=0, lvl=0):
    return "ndarray_uint16_1: %s" % ndarray_to_str(a,1,indent,lvl)

def ndarray_uint16_2_to_str(a, indent=0, lvl=0):
    return "ndarray_uint16_2: %s" % ndarray_to_str(a,2,indent,lvl)

def ndarray_uint16_3_to_str(a, indent=0, lvl=0):
    return "ndarray_uint16_3: %s" % ndarray_to_str(a,3,indent,lvl)

def ndarray_uint16_4_to_str(a, indent=0, lvl=0):
    return "ndarray_uint16_4: %s" % ndarray_to_str(a,4,indent,lvl)

def ndarray_uint16_5_to_str(a, indent=0, lvl=0):
    return "ndarray_uint16_5: %s" % ndarray_to_str(a,5,indent,lvl)

def ndarray_uint16_6_to_str(a, indent=0, lvl=0):
    return "ndarray_uint16_6: %s" % ndarray_to_str(a,6,indent,lvl)

def ndarray_uint32_1_to_str(a, indent=0, lvl=0):
    return "ndarray_uint32_1: %s" % ndarray_to_str(a,1,indent,lvl)

def ndarray_uint32_2_to_str(a, indent=0, lvl=0):
    return "ndarray_uint32_2: %s" % ndarray_to_str(a,2,indent,lvl)

def ndarray_uint32_3_to_str(a, indent=0, lvl=0):
    return "ndarray_uint32_3: %s" % ndarray_to_str(a,3,indent,lvl)

def ndarray_uint32_4_to_str(a, indent=0, lvl=0):
    return "ndarray_uint32_4: %s" % ndarray_to_str(a,4,indent,lvl)

def ndarray_uint32_5_to_str(a, indent=0, lvl=0):
    return "ndarray_uint32_5: %s" % ndarray_to_str(a,5,indent,lvl)

def ndarray_uint32_6_to_str(a, indent=0, lvl=0):
    return "ndarray_uint32_6: %s" % ndarray_to_str(a,6,indent,lvl)

def ndarray_uint64_1_to_str(a, indent=0, lvl=0):
    return "ndarray_uint64_1: %s" % ndarray_to_str(a,1,indent,lvl)

def ndarray_uint64_2_to_str(a, indent=0, lvl=0):
    return "ndarray_uint64_2: %s" % ndarray_to_str(a,2,indent,lvl)

def ndarray_uint64_3_to_str(a, indent=0, lvl=0):
    return "ndarray_uint64_3: %s" % ndarray_to_str(a,3,indent,lvl)

def ndarray_uint64_4_to_str(a, indent=0, lvl=0):
    return "ndarray_uint64_4: %s" % ndarray_to_str(a,4,indent,lvl)

def ndarray_uint64_5_to_str(a, indent=0, lvl=0):
    return "ndarray_uint64_5: %s" % ndarray_to_str(a,5,indent,lvl)

def ndarray_uint64_6_to_str(a, indent=0, lvl=0):
    return "ndarray_uint64_6: %s" % ndarray_to_str(a,6,indent,lvl)

