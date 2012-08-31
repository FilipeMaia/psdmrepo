"""
Tool which selects correct python version for PSDM releases.
"""
import os
from os.path import join as pjoin

from SConsTools.trace import *
from SConsTools.scons_functions import *


def generate(env):
    
    if env['SIT_ARCH_OS'] == 'rhel5':

        prefix = "/usr"
        version = "2.4"
        # RH-distributed Python is patched to use lib64 on 64-bit platforms
        libdir = env['LIB_ABI']
    
    elif env['SIT_ARCH_OS'] == 'rhel6':
        
        prefix = pjoin("/reg/g/psdm/sw/external/python/2.7.2", env['SIT_ARCH_BASE_OPT'])
        version = "2.7"
        libdir = "lib"
        
    elif env['SIT_ARCH_OS'] == 'ubu11':
        
        prefix = "/usr"
        version = "2.7"
        libdir = 'lib'
    
    env['PYTHON_PREFIX'] = prefix
    env['PYTHON_VERSION'] = version
    env['PYTHON'] = "python"+version
    env['PYTHON_LIBDIRNAME'] = libdir
    env['PYTHON_INCDIR'] = pjoin(prefix, "include", env['PYTHON'])
    env['PYTHON_LIBDIR'] = pjoin(prefix, libdir)
    env['PYTHON_BINDIR'] = pjoin(prefix, "bin")
    env['PYTHON_BIN'] = pjoin(env['PYTHON_BINDIR'], env['PYTHON'])
    
    env['SCRIPT_SUBS']['PYTHON'] = env['PYTHON_BIN']
    
    trace ( "Initialized psdm_python tool", "psdm_python", 2 )

def exists(env):
    return _qtdir(env) is not None
