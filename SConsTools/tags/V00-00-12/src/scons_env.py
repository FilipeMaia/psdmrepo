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

# ===================================
#   Setup default build environment
# ===================================
def buildEnv () :
    
    
    vars = Variables()
    vars.Add('LUSI_ARCH', "Use to change the LUSI_ARCH value during build", os.environ['LUSI_ARCH'] )
    vars.Add('LUSI_REPOS', "Use to change the LUSI_REPOS value during build", os.environ.get('LUSI_REPOS',"") )
    vars.Add('PKG_DEPS_FILE', "name of the package dependency file", '.pkg_tree.pkl' )
    vars.Add('TRACE', "Set to positive value to trace processing", 0)
    
    # make environment, also make it default
    env = DefaultEnvironment( variables = vars )

    # set trace level based on the command line value
    tracev = int(env['TRACE'])
    setTraceLevel ( tracev )

    # get repository list from it
    lusi_repos = env['LUSI_REPOS']
    lusi_repos = [ r for r in lusi_repos.split(':') if r ]
    # all repos including local
    all_lusi_repos = [ '#' ] + lusi_repos

    # arch parts
    lusi_arch = env['LUSI_ARCH']
    lusi_arch_parts = lusi_arch.split('-')
    lusi_arch_base = '-'.join(lusi_arch_parts[0:3])

    # LUSI_ROOT
    lusi_root = os.environ.get( "LUSI_ROOT", "" )

    # build all paths    
    archdir = pjoin("#arch/${LUSI_ARCH}")
    archincdir = pjoin("${ARCHDIR}/geninc")
    bindir = pjoin("${ARCHDIR}/bin")
    libdir = pjoin("${ARCHDIR}/lib")
    pydir = pjoin("${ARCHDIR}/python")
    cpppath = []
    for r in all_lusi_repos :
        cpppath.append ( pjoin(r,"arch/${LUSI_ARCH}/geninc") )
        cpppath.append ( pjoin(r,"include") )
    libpath = [ pjoin(r,"arch/${LUSI_ARCH}/lib") for r in all_lusi_repos ]
    
    # set other variables in environment
    env.Replace( ARCHDIR = archdir,
                 ARCHINCDIR = archincdir,
                 BINDIR = bindir,
                 LIBDIR = libdir,
                 PYDIR = pydir,
                 CPPPATH = cpppath,
                 LIBPATH = libpath,
                 LUSI_ROOT = lusi_root,
                 LUSI_ARCH_PROC = lusi_arch_parts[0],
                 LUSI_ARCH_OS = lusi_arch_parts[1],
                 LUSI_ARCH_COMPILER = lusi_arch_parts[2],
                 LUSI_ARCH_OPT = lusi_arch_parts[3],
                 LUSI_ARCH_BASE = lusi_arch_base,
                 LUSI_REPOS = lusi_repos,
                 PKG_TREE = {},
                 PKG_TREE_BASE = {},
                 PKG_TREE_BINS = {},
                 ALL_TARGETS = {} )
    
    # may want to use "relative" RPATH
    # env.Replace( RPATH = env.Literal("'$$ORIGIN/../lib'") )

    # these lists will be filled by standard rules
    env['ALL_TARGETS']['LIBS'] = []
    env['ALL_TARGETS']['BINS'] = []
    env['ALL_TARGETS']['TESTS'] = []

    # generate help    
    Help(vars.GenerateHelpText(env))
    
    #trace ( "Build env = "+pprint.pformat(env.Dictionary()), "buildEnv", 7 )
    
    #for r in lusi_repos :
    #    trace ( "Add repository "+r, "<top>", 2 )
    #    Repository( r )

    return env
