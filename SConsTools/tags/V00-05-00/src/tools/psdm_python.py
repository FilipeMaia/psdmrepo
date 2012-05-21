"""
Tool which selects correct python version for PSDM releases.
"""
import os

from SConsTools.trace import *
from SConsTools.scons_functions import *


def generate(env):
    
    if env['SIT_ARCH_OS'] == 'rhel5':
        
        env['PYTHON_VERSION'] = "2.4"
        env['PYTHON'] = "python"+env['PYTHON_VERSION']
        env['PYTHON_INCDIR'] = "/usr/include/"+env['PYTHON']
        env['PYTHON_LIBDIR'] = "/usr/"+env['LIB_ABI']
        env['PYTHON_BIN'] = "/usr/bin/"+env['PYTHON']
    
    elif env['SIT_ARCH_OS'] == 'rhel6':
        
        env['PYTHON_VERSION'] = "2.6"
        env['PYTHON'] = "python"+env['PYTHON_VERSION']
        env['PYTHON_INCDIR'] = "/usr/include/"+env['PYTHON']
        env['PYTHON_LIBDIR'] = "/usr/"+env['LIB_ABI']
        env['PYTHON_BIN'] = "/usr/bin/"+env['PYTHON']
    
    elif env['SIT_ARCH_OS'] == 'ubu11':
        
        env['PYTHON_VERSION'] = "2.7"
        env['PYTHON'] = "python"+env['PYTHON_VERSION']
        env['PYTHON_INCDIR'] = "/usr/include/"+env['PYTHON']
        env['PYTHON_LIBDIR'] = "/usr/"+env['LIB_ABI']
        env['PYTHON_BIN'] = "/usr/bin/"+env['PYTHON']
    
    env['SCRIPT_SUBS']['PYTHON'] = env['PYTHON_BIN']
    
    trace ( "Initialized psdm_python tool", "psdm_python", 2 )

def exists(env):
    return _qtdir(env) is not None
