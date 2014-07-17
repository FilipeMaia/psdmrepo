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


def _getNumCpus():
    # determin a number of CPUs in a system
    try:
        return os.sysconf('SC_NPROCESSORS_ONLN')
    except:
        # guess minimum is one but modern systems
        # have at least couple of cores
        return 2

# ===================================
#   Setup default build environment
# ===================================
def buildEnv () :

    # use half of all CPUs
    SetOption('num_jobs', _getNumCpus()/2 or 1)

    # SIT_ROOT
    sit_root = os.environ["SIT_ROOT"]
    
    # SIT_RELEASE
    sit_release = os.environ['SIT_RELEASE']

    # default DESTDIR
    destdir = pjoin(sit_root, "sw/releases", sit_release)

    # SIT_EXTERNAL_SW
    sit_external_sw = pjoin(sit_root, "sw/external")

    vars = Variables()
    vars.AddVariables(
        ('CPPFLAGS', "User-specified C preprocessor options", ""),
        ('CCFLAGS', "General options that are passed to the C and C++ compilers", ""),
        ('CFLAGS', "General options that are passed to the C compiler (C only; not C++)", ""),
        ('CXXFLAGS', "General options that are passed to the C++ compiler", ""),
        ('LINKFLAGS', "General user options passed to the linker", ""),
        ('SIT_ARCH', "Use to change the SIT_ARCH value during build", os.environ['SIT_ARCH']),
        ('SIT_RELEASE', "Use to change the SIT_RELEASE value during build", sit_release),
        ('SIT_REPOS', "Use to change the SIT_REPOS value during build", os.environ.get('SIT_REPOS', "")),
        PathVariable('SIT_EXTERNAL_SW', "Use to change the SIT_EXTERNAL_SW value during build", sit_external_sw, PathVariable.PathIsDir),
        PathVariable('PKG_DEPS_FILE', "Name of the package dependency file", '.pkg_tree.pkl', PathVariable.PathAccept),
        PathVariable('PKG_LIST_FILE', "Name of the package list file", '/dev/stdout', PathVariable.PathAccept),
        PathVariable('DESTDIR', "destination directory for install target", destdir, PathVariable.PathAccept),
        ('TRACE', "Set to positive value to trace processing", 0)
    )

    # make environment, also make it default
    env = DefaultEnvironment(ENV=os.environ, variables=vars)
    vars.GenerateHelpText(env)

    # set trace level based on the command line value
    setTraceLevel(int(env['TRACE']))

    # get repository list from it
    sit_repos = [ r for r in env['SIT_REPOS'].split(':') if r ]

    # all repos including local
    all_sit_repos = [ '#' ] + sit_repos

    # arch parts
    sit_arch = env['SIT_ARCH']
    sit_arch_parts = sit_arch.split('-')
    sit_arch_base = '-'.join(sit_arch_parts[0:3])

    # LIB_ABI will translate either to lib or lib64 depending on which architecture we are
    lib_abis = {'x86_64-rhel5': "lib64", 
                'x86_64-rhel6': "lib64", 
                'x86_64-rhel7': "lib64",
                'x86_64-suse11': "lib64", 
                'x86_64-suse12': "lib64", 
                'x86_64-ubu12': 'lib/x86_64-linux-gnu'}
    lib_abi = lib_abis.get(sit_arch_parts[0]+'-'+sit_arch_parts[1], "lib")

    # build all paths    
    archdir = pjoin("#arch/", sit_arch)
    archincdir = "${ARCHDIR}/geninc"
    bindir = "${ARCHDIR}/bin"
    libdir = "${ARCHDIR}/lib"
    pydir = "${ARCHDIR}/python"
    phpdir = "${ARCHDIR}/php"
    cpppath = ['.']   # this translates to package directory, not to top dir
    for r in all_sit_repos :
        cpppath.append(pjoin(r, "arch", sit_arch, "geninc"))
        cpppath.append(pjoin(r, "include"))
    libpath = [ pjoin(r, "arch", sit_arch, "lib") for r in all_sit_repos ]

    # set other variables in environment
    env.Replace(ARCHDIR=archdir,
                ARCHINCDIR=archincdir,
                BINDIR=bindir,
                LIBDIR=libdir,
                PYDIR=pydir,
                PHPDIR=phpdir,
                CPPPATH=cpppath,
                LIBPATH=libpath,
                LIB_ABI=lib_abi,
                SIT_ROOT=sit_root,
                SIT_ARCH_PROC=sit_arch_parts[0],
                SIT_ARCH_OS=sit_arch_parts[1],
                SIT_ARCH_COMPILER=sit_arch_parts[2],
                SIT_ARCH_OPT=sit_arch_parts[3],
                SIT_ARCH_BASE=sit_arch_base,
                SIT_ARCH_BASE_OPT=sit_arch_base+"-opt",
                SIT_ARCH_BASE_DBG=sit_arch_base+"-dbg",
                SIT_RELEASE=sit_release,
                SIT_REPOS=sit_repos,
                PKG_TREE={},
                PKG_TREE_BASE={},
                PKG_TREE_BINDEPS={},
                PKG_TREE_LIB={},
                PKG_TREE_BINS={},
                ALL_TARGETS={},
                CXXFILESUFFIX=".cpp",
                EXT_PACKAGE_INFO = {},
                SCRIPT_SUBS = {},
                DOC_TARGETS = {}
                )

    # location of the tools
    toolpath = [ pjoin(r, "arch", sit_arch, "python/SConsTools/tools") for r in all_sit_repos ]
    env.Replace(TOOLPATH=toolpath)

    # extend environment with tools
    tools = ['psdm_cplusplus', 'psdm_python', 'pyext', 'cython', 'symlink', 
             'pycompile', 'pylint', 'unittest', 'script_install', 'pkg_list', 
             'release_install', 'special_scanners']
    trace ("toolpath = " + pformat(toolpath), "buildEnv", 3)
    for tool in tools:
        tool = env.Tool(tool, toolpath=toolpath)

    # override some CYTHON vars
    cythonflags = ["--cplus", '-I', '.', '-I', pjoin("arch", sit_arch, "geninc"), '-I', 'include']
    for r in sit_repos :
        cythonflags += ["-I", pjoin(r, "arch", sit_arch, "geninc")]
        cythonflags += ["-I", pjoin(r, "include")]
    cythonflags = ' '.join(cythonflags)
    env.Replace(CYTHONFLAGS=cythonflags, CYTHONCFILESUFFIX=".cpp")

    # use alternative location for sconsign file
    env.SConsignFile(pjoin("build", sit_arch, ".sconsign"))

    # may want to use "relative" RPATH
    # env.Replace( RPATH = env.Literal("'$$ORIGIN/../lib'") )

    # these lists will be filled by standard rules
    env['ALL_TARGETS']['INCLUDES'] = []
    env['ALL_TARGETS']['LIBS'] = []
    env['ALL_TARGETS']['BINS'] = []
    env['ALL_TARGETS']['TESTS'] = []
    env['ALL_TARGETS']['PYLINT'] = []

    # generate help
    Help(vars.GenerateHelpText(env))

    trace ("Build env = " + pformat(env.Dictionary()), "buildEnv", 7)

    #for r in sit_repos :
    #    trace ( "Add repository "+r, "<top>", 2 )
    #    Repository( r )

    return env
