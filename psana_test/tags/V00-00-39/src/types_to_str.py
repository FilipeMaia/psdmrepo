import ctypes
import numpy as np
import sys

dll = ctypes.cdll.LoadLibrary('libpsana_test.so')
psana_test_adler32 = dll.psana_test_adler32
psana_test_adler32.restype = ctypes.c_ulong
psana_test_adler32.argtypes = [ctypes.c_void_p, ctypes.c_ulong]

def doIndent(indent,lvl):
    return ((' '*indent)*lvl)

def epicsPvToStr(pv, addPvId=True):
    def toStr(x):
        return str(x)

    def numericToStr(x):
        if isinstance(x,float):
            return '%.4e' % x
        else:
            return str(x)
    # all epics pv have these attributes:
    if addPvId:
        res = ' pvid=%s' % pv.pvId()
    else:
        res = ''
    dbrType = pv.dbrType()
    res += ' dbrtype=%s' % dbrType 
    if pv.isTime():
        res += " isTime=1"
    elif pv.isCtrl():
        res += " isCtrl=1" 
        res += ' pvName=%s' % pv.pvName()
    res += " numElements=%s" % pv.numElements()
    res += ' status=%s' % pv.status()
    res += ' severity=%s' % pv.severity()
    if pv.isCtrl():
        dbr = pv.dbr()
        if dbrType not in [28, 31]:
            # not a CtrlString and not a CtrlEnum so it has a dbr with:
            res += ' units=%s' % dbr.units()
            res += ' upper_disp_limit=%s' % numericToStr(dbr.upper_disp_limit())
            res += ' lower_disp_limit=%s' % numericToStr(dbr.lower_disp_limit())
            res += ' upper_alarm_limit=%s' % numericToStr(dbr.upper_alarm_limit())
            res += ' upper_warning_limit=%s' % numericToStr(dbr.upper_warning_limit())
            res += ' lower_warning_limit=%s' % numericToStr(dbr.lower_warning_limit())
            res += ' lower_alarm_limit=%s' % numericToStr(dbr.lower_alarm_limit())
            res += ' upper_ctrl_limit=%s' % numericToStr(dbr.upper_ctrl_limit())
            res += ' lower_ctrl_limit=%s' % numericToStr(dbr.lower_ctrl_limit())
        elif dbrType == 28:
            # CtrlString
            pass
        elif dbrType == 31:
            # CtrlEnum
            res += ' no_str=%d' % dbr.no_str()
            for ii in range(dbr.no_str()):
                res += ' enum[%d]=%s' % (ii, dbr.strings(ii))
    elif pv.isTime():
        stamp = pv.stamp()
        res += ' stamp.sec=%s stamp.nsec=%s' % (stamp.sec(), stamp.nsec())
    # for string pv's (dbrType = 14 or 28), elements are not value_type, 
    # and data() will take an index argument
    if pv.dbrType() in [14,28]:
        dataStr = [str(pv.data(idx)) for idx in range(pv.numElements())]
    else:
        data = pv.data()
        assert len(data.shape)==1, "unexpected: pv has shape that is not len 1: shape=%r" % data.shape
        fn = getTypeFn(data.dtype)
        assert fn is not None
        if data.shape[0] < 20:
            dataStr = ' '.join(map(fn,data.flatten()))
        else:
            dataStr = ndarray_to_str(data, indent=0, lvl=0)

    res += ' data=%s' % dataStr
    return res

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
    rank = len(a.shape)
    numElem = 1
    for dim in a.shape:
        numElem *= dim
    outstr = doIndent(indent,lvl)
    outstr +=  "ndarray_%s_%d: dim=[ %s ]" % (a.dtype.name, rank, dimstr)
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

