import psana
import sys
from sys import stderr, stdout

__doc__ = '''
Psana Module for Testing. 

Gets two ndarray during each event.
One is const, the other non-const.
Prints

nonconst_arr null
const_arr null

if it cannot obtain them, otherwise

nonconst_arr: #typeid
const_arr: #typeid

it also attempts to write to the const array.
If it can, it writes to stderr

ERROR: writing to const ndarray did not trigger exception

likewise if writing to the nonconst array does not work, an exception is triggered.
'''

class PsanaModuleGetArrayPy(object):
    def __init__(self):
        self.const_key = self.configStr("const_key",'')
        self.nonconst_key = self.configStr("nonconst_key",'')
        
    def event(self,evt,env):
        nonconst_arr = None
        const_arr = None
        if len(self.nonconst_key)>0:
            nonconst_arr = evt.get(psana.ndarray_float32_2, self.nonconst_key)
        if (nonconst_arr is None):
            stdout.write("nonconst_arr null\n")
        else:
            stdout.write("nonconst_arr: %r\n" % id(nonconst_arr))

        if len(self.const_key)>0:
            const_arr = evt.get(psana.ndarray_float32_2,self.const_key)
        if (const_arr is None):
            stdout.write("const_arr null\n")
        else:
            stdout.write("const_arr: %r\n" % id(const_arr))

        if const_arr is not None:
            exceptionMsg = ''
            try:
                const_arr[0,0]=234.0
            except Exception,e:
                exceptionMsg = str(e)
            if len(exceptionMsg)>0:
                stdout.write("success: trying to write to const ndarray triggered exception: %s\n" % exceptionMsg)
            else:
                stderr.write("ERROR: writing to const ndarray did not trigger exception\n")

        if nonconst_arr is not None:
            try:
                nonconst_arr[0,0]=234.0
                stdout.write("success: writing to nonconst array worked fine\n")
            except Exception,e:
                stderr.write("ERROR: writing to nonconst array generated exception: %s\n" % str(e))
            

