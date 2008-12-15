#===============================================================================
#
# Main SCons script for LUSI release building
#
# $Id$
#
#===============================================================================

import os
import sys
import pprint
from os.path import join as pjoin

from SCons.Defaults import *
from SCons.Script import *

from trace import *

_gcc_opt = { 'opt' : '-O3',
            'deb' : '-g',
            'prof' : '-pg' }

# ===================================
#   Setup default build environment
# ===================================
def setupCompilers ( env ) :
    
    proc = env['LUSI_ARCH_PROC']
    os = env['LUSI_ARCH_OS']
    comp = env['LUSI_ARCH_COMPILER']
    opt = env['LUSI_ARCH_OPT']
    
    if comp == 'gcc34' :
        env['CC'] = 'gcc-3.4'
        env['CXX'] = 'g++-3.4'
        env['CCFLAGS'] = _gcc_opt.get(opt,'')

    env['PYTHON_VERSION'] = "2.6"

    # various substitutions for the scripts 
    env.SetDefault (SCRIPT_SUBS = {})
    env['SCRIPT_SUBS']['PYTHON'] = "/bin/env python"+env['PYTHON_VERSION']
