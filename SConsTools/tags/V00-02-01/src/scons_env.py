#===============================================================================
#
# Main SCons script for SIT release building
#
# $Id$
#
#===============================================================================

import os
import sys
from pprint import *
from os.path import join as pjoin

from SCons.Defaults import *
from SCons.Script import *

from trace import *

# ===================================
#   Setup default build environment
# ===================================
def buildEnv () :
    
    
    vars = Variables()
    vars.Add('SIT_ARCH', "Use to change the SIT_ARCH value during build", os.environ['SIT_ARCH'] )
    vars.Add('SIT_REPOS', "Use to change the SIT_REPOS value during build", os.environ.get('SIT_REPOS',"") )
    vars.Add('PKG_DEPS_FILE', "name of the package dependency file", '.pkg_tree.pkl' )
    vars.Add('TRACE', "Set to positive value to trace processing", 0)
    
    # make environment, also make it default
    env = DefaultEnvironment(ENV = os.environ, variables = vars)

    # set trace level based on the command line value
    setTraceLevel(int(env['TRACE']))

    # get repository list from it
    sit_repos = [ r for r in env['SIT_REPOS'].split(':') if r ]
    
    # all repos including local
    all_sit_repos = [ '#' ] + sit_repos

    # SIT_ROOT
    sit_root = os.environ["SIT_ROOT"]

    # arch parts
    sit_arch = env['SIT_ARCH']
    sit_arch_parts = sit_arch.split('-')
    sit_arch_base = '-'.join(sit_arch_parts[0:3])

    # extend environment with tools
    tools = ['pyext', 'cython', 'symlink', 'pycompile', 'unittest', 'script_install']
    toolpath = [ pjoin(r, "arch", sit_arch, "python/SConsTools/tools") for r in all_sit_repos ]
    trace ( "toolpath = " + pformat(toolpath), "buildEnv", 3 )
    for tool in tools:
        tool = env.Tool(tool, toolpath = toolpath)
    
    # build all paths    
    archdir = pjoin("#arch/", sit_arch)
    archincdir = "${ARCHDIR}/geninc"
    bindir = "${ARCHDIR}/bin"
    libdir = "${ARCHDIR}/lib"
    pydir = "${ARCHDIR}/python"
    cpppath = []
    for r in all_sit_repos :
        cpppath.append(pjoin(r, "arch", sit_arch, "geninc"))
        cpppath.append(pjoin(r, "include"))
    libpath = [ pjoin(r, "arch", sit_arch, "lib") for r in all_sit_repos ]

    cythonflags = ["--cplus", '-I', pjoin("arch", sit_arch, "geninc"), '-I', 'include']
    for r in sit_repos :
        cythonflags += ["-I", pjoin(r, "arch", sit_arch, "geninc")]
        cythonflags += ["-I", pjoin(r, "include")]
    cythonflags = ' '.join(cythonflags)

    # set other variables in environment
    env.Replace( ARCHDIR = archdir,
                 ARCHINCDIR = archincdir,
                 BINDIR = bindir,
                 LIBDIR = libdir,
                 PYDIR = pydir,
                 CPPPATH = cpppath,
                 LIBPATH = libpath,
                 SIT_ROOT = sit_root,
                 SIT_ARCH_PROC = sit_arch_parts[0],
                 SIT_ARCH_OS = sit_arch_parts[1],
                 SIT_ARCH_COMPILER = sit_arch_parts[2],
                 SIT_ARCH_OPT = sit_arch_parts[3],
                 SIT_ARCH_BASE = sit_arch_base,
                 SIT_REPOS = sit_repos,
                 PKG_TREE = {},
                 PKG_TREE_BASE = {},
                 PKG_TREE_BINDEPS = {},
                 PKG_TREE_LIB = {},
                 PKG_TREE_BINS = {},
                 ALL_TARGETS = {},
                 CXXFILESUFFIX = ".cpp",
                 CYTHONFLAGS = cythonflags,
                 CYTHONCFILESUFFIX = ".cpp"
                 )

    # may want to use "relative" RPATH
    # env.Replace( RPATH = env.Literal("'$$ORIGIN/../lib'") )

    # these lists will be filled by standard rules
    env['ALL_TARGETS']['LIBS'] = []
    env['ALL_TARGETS']['BINS'] = []
    env['ALL_TARGETS']['TESTS'] = []

    # generate help
    Help(vars.GenerateHelpText(env))
    
    trace ( "Build env = " + pformat(env.Dictionary()), "buildEnv", 7 )
    
    #for r in sit_repos :
    #    trace ( "Add repository "+r, "<top>", 2 )
    #    Repository( r )

    return env
