import psana
import sys
from sys import stderr, stdout
import numpy as np

__doc__ = '''
Psana Module for Testing. 

Puts two ndarray during each event.
One is const, the other non-const.
Both are 3 x 4 2D arrays of np.float32, or C++ float
with the values 0.0, 1.0, 2.0 ... 11.0

config:

const_key        keystring for const ndarray
nonconst_key     keystring for nonconst ndarray

if const_key or non_const_key is blank, does not put an ndarry in.
'''

class PsanaModulePutArrayPy(object):
    def __init__(self):
        self.const_key = self.configStr("const_key",'')
        self.nonconst_key = self.configStr("nonconst_key",'')
        
    def event(self,evt,env):
        nonconst_arr = np.ndarray.astype(np.array(range(12)),np.float32)        
        const_arr = np.ndarray.astype(np.array(range(12)),np.float32)
        nonconst_arr.resize((3,4))
        const_arr.resize((3,4))
        const_arr.flags['WRITEABLE']=False
        if len(self.nonconst_key)>0:
            evt.put(nonconst_arr, self.nonconst_key)
        if len(self.const_key)>0:
            evt.put(const_arr, self.const_key)
            

