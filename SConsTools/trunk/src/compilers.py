#===============================================================================
#
# Main SCons script for SIT release building
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
            'dbg' : '-g',
            'deb' : '-g',     # 'deb' was an unfortunate name for 'dbg'
            'prof' : '-pg' }

_ld_opt = { 'opt' : '',
            'dbg' : '-g',
            'deb' : '-g',     # 'deb' was an unfortunate name for 'dbg'
            'prof' : '-pg -static' }

# ===================================
#   Setup default build environment
# ===================================
def setupCompilers ( env ) :
    
    proc = env['SIT_ARCH_PROC']
    os = env['SIT_ARCH_OS']
    comp = env['SIT_ARCH_COMPILER']
    opt = env['SIT_ARCH_OPT']
    
    if comp == 'gcc34' :
        env['CC'] = 'gcc34'
        env['CXX'] = 'g++34'
        if os == 'slc4' :
            env['CC'] = 'gcc-3.4'
            env['CXX'] = 'g++-3.4'
        env['CCFLAGS'] = _gcc_opt.get(opt,'') + ' -Wall'
        env['LDFLAGS'] = _ld_opt.get(opt,'')

    elif comp == 'gcc41' :
        env['CC'] = 'gcc'
        env['CXX'] = 'g++'
        env['CCFLAGS'] = _gcc_opt.get(opt,'') + ' -Wall'
        env['LDFLAGS'] = _ld_opt.get(opt,'')

    env['PYTHON_VERSION'] = "2.4"
    env['PYTHON'] = "/usr/bin/python"+env['PYTHON_VERSION']

    # various substitutions for the scripts 
    env.SetDefault (SCRIPT_SUBS = {})
    env['SCRIPT_SUBS']['PYTHON'] = env['PYTHON']
