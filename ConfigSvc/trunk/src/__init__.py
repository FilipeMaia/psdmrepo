#
# import everything from _ConfigSvc
#
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

