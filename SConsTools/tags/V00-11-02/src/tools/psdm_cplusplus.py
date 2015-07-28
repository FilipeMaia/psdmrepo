"""
Tool which selects correct C++ compiler version and options for PSDM releases.
"""
import os

from SConsTools.trace import *
from SConsTools.scons_functions import *

_gcc_opt = { 'opt' : '-O3',
            'dbg' : '-g' }

_ld_opt = { 'opt' : '',
            'dbg' : '-g' }

def generate(env):
    
    os = env['SIT_ARCH_OS']
    comp = env['SIT_ARCH_COMPILER']
    opt = env['SIT_ARCH_OPT']
    
    if comp == 'gcc41' :
        env['CC'] = 'gcc'
        env['CXX'] = 'g++'
        env.Append(CCFLAGS = ' ' + _gcc_opt.get(opt,'') + ' -Wall -Wno-unknown-pragmas')
        env.Append(CXXFLAGS = ' -Wno-invalid-offsetof')
        env.Append(LINKFLAGS = ' ' + _ld_opt.get(opt,''))

    elif comp in ['gcc44', 'gcc45'] :
        env['CC'] = 'gcc'
        env['CXX'] = 'g++'
        env.Append(CCFLAGS = ' ' + _gcc_opt.get(opt,'') + ' -Wall')
        env.Append(CXXFLAGS = ' -Wno-invalid-offsetof')
        env.Append(LINKFLAGS = ' ' + _ld_opt.get(opt,''))

    elif comp == 'gcc46' :
        env['CC'] = 'gcc-4.6'
        env['CXX'] = 'g++-4.6'
        env.Append(CCFLAGS = ' ' + _gcc_opt.get(opt,'') + ' -Wall')
        env.Append(CXXFLAGS = ' -Wno-invalid-offsetof')
        env.Append(LINKFLAGS = ' ' + _ld_opt.get(opt,''))

    elif comp == 'gcc48' :
        env['CC'] = 'gcc'
        env['CXX'] = 'g++'
        env.Append(CCFLAGS = ' ' + _gcc_opt.get(opt,'') + ' -Wall')
        env.Append(CXXFLAGS = ' -Wno-invalid-offsetof')
        env.Append(LINKFLAGS = ' ' + _ld_opt.get(opt,'') + ' -Wl,--copy-dt-needed-entries')

    
    trace ( "Initialized psdm_cplusplus tool", "psdm_cplusplus", 2 )

def exists(env):
    return True
