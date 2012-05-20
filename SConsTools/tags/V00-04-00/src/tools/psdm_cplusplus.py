"""
Tool which selects correct C++ compiler version and options for PSDM releases.
"""
import os

from SConsTools.trace import *
from SConsTools.scons_functions import *

_gcc_opt = { 'opt' : '-O3',
            'dbg' : '-g',
            'deb' : '-g',     # 'deb' was an unfortunate name for 'dbg'
            'prof' : '-pg' }

_ld_opt = { 'opt' : '',
            'dbg' : '-g',
            'deb' : '-g',     # 'deb' was an unfortunate name for 'dbg'
            'prof' : '-pg -static' }

def generate(env):
    
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
        env['CXXFLAGS'] = '-Wno-invalid-offsetof'
        env['LDFLAGS'] = _ld_opt.get(opt,'')

    elif comp == 'gcc41' :
        env['CC'] = 'gcc'
        env['CXX'] = 'g++'
        env['CCFLAGS'] = _gcc_opt.get(opt,'') + ' -Wall'
        env['CXXFLAGS'] = '-Wno-invalid-offsetof'
        env['LDFLAGS'] = _ld_opt.get(opt,'')

    elif comp == 'gcc46' :
        env['CC'] = 'gcc-4.6'
        env['CXX'] = 'g++-4.6'
        env['CCFLAGS'] = _gcc_opt.get(opt,'') + ' -Wall'
        env['CXXFLAGS'] = '-Wno-invalid-offsetof'
        env['LDFLAGS'] = _ld_opt.get(opt,'')

    
    trace ( "Initialized psdm_python tool", "psdm_python", 2 )

def exists(env):
    return _qtdir(env) is not None
