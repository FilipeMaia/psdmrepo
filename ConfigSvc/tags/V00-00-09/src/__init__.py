"""
This module provides wrapper for :py:mod:`_ConfigSvc` extension module, wrapper is
necessary to set correct ldopen() flags before importing :py:mod:`_ConfigSvc`. 
This module imports all names defined in :py:mod:`_ConfigSvc` and can be used 
in place of :py:mod:`_ConfigSvc` for accessing its contents. For complete reference
see documentation of :py:mod:`_ConfigSvc`.
"""

import sys
if sys.platform == 'linux2':
    # on Linux with g++ one needs RTLD_GLOBAL for dlopen
    # which Python does not set by default
    import DLFCN
    flags = sys.getdlopenflags()
    sys.setdlopenflags( flags | DLFCN.RTLD_GLOBAL )    
    from _ConfigSvc import *
    sys.setdlopenflags( flags )    
    del flags
    del DLFCN
else:
    from _ConfigSvc import *
del sys

