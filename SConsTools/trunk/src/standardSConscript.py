#===============================================================================
#
# SConscript fuction for standard LUSI package
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

from SConsTools.trace import *
from SConsTools.dependencies import *


_cplusplus_ext = [ 'cc', 'cpp', 'cxx', 'C', 'c' ]

def _normbinsrc ( f ):
    if not os.path.split(f)[0] :
        return os.path.join("app",f)
    else :
        return f

#
# This is the content of the standard SConscript
#
def standardSConscript( **kw ) :

    """ Understands following keywords, all optional:
        BINS - dictionary of executables and their corresponding source files
        LIBS - list of additional libraries that have to be linked with applications
    """

    pkg = os.path.basename(os.getcwd())
    trace ( "Standard SConscript in `"+pkg+"'", "SConscript", 1 )
    
    env = DefaultEnvironment()
    bindir = env['BINDIR']
    libdir = env['LIBDIR']
    pydir = env['PYDIR']

    # Program options
    binkw = {}
    if 'LIBS' in kw :
        libs = kw['LIBS']
        if isinstance(libs,(str,unicode)) : libs = libs.split()
        binkw['LIBS'] = libs
    if 'LIBDIRS' in kw : binkw['LIBDIRS'] = kw['LIBDIRS']

    #
    # Process src/ directory, make library from all compilable files
    #
    libsrcs = Flatten ( [ Glob("src/*."+ext, source=True ) for ext in _cplusplus_ext ] )
    if libsrcs :        
        trace ( "libsrcs = "+pformat([str(s) for s in libsrcs]), "SConscript", 2 )

        # c++ files area compiled into library  
        if libsrcs :
             
            lib = env.SharedLibrary ( pkg, source=libsrcs )
            env.Install ( libdir, source=lib )
            deps = findAllDependencies( lib[0] )
            trace ( "deps = " + pformat(deps), "SConscript", 4 )
            
            setPkgDeps ( env, pkg, deps )
            setPkgLibs ( env, pkg, [ pkg ] )
            
            binkw.setdefault('LIBS',[]).insert ( 0, pkg )

    #
    # Process src/ directory, link python sources
    #
    pysrcs = Glob("src/*.py", source=True, strings=True )
    if pysrcs :
        trace ( "pysrcs = "+pformat(pysrcs), "SConscript", 2 )

        # python files area installed into python/Package
        for src in pysrcs :
            
            # make symlink for every .py file and compile it into .pyc
            basename = os.path.basename(src)
            pydst = pjoin(pydir,pkg,basename)
            env.Symlink ( pydst, source=src )
            env.PyCompile ( pydst+"c", source=pydst )
            
            # make __init__.py and compile it
            ini = pjoin(pydir,pkg,"__init__.py")
            env.Command ( ini, "", [ Touch("$TARGET") ] )
            env.PyCompile ( ini+"c", source=ini )

    #
    # Process app/ directory, install all scripts
    #
    app_files = Glob("app/*", source=True, strings=True )
    if app_files :
        
        scripts = [ f for f in app_files if not os.path.splitext(f)[1] and os.path.isfile(f) ]
        trace ( "scripts = "+pformat(scripts), "SConscript", 2 )

        # Scripts are copied to bin/ directory
        for s in scripts : 
            env.ScriptInstall ( os.path.join(bindir,os.path.basename(s)), s )

    #
    # Process app/ directory, build all from C++ sources
    #
    bins = kw.get('BINS',{})
    if bins :
        for k in bins.iterkeys() :
            src = bins[k]
            if isinstance(src,(str,unicode)) : src = src.split()
            src = [ _normbinsrc(s) for s in src ]
            bins[k] = src
    else :
        cpps = Flatten ( [ Glob("app/*."+ext, source=True, strings=True ) for ext in _cplusplus_ext ] )
        for f in cpps :
            bin = os.path.splitext(os.path.basename(f))[0]
            bins[bin] = [ f ]
    if bins :

        trace ( "bins = "+pformat(bins), "SConscript", 2 )
        
        for bin, srcs in bins.iteritems() :
            
            b = env.Program( bin, source=srcs, **binkw )
            env.Install ( bindir, source=b )
            
            deps = findAllDependencies( b[0] )
            trace ( bin+" deps = " + pformat(deps), "SConscript", 4 )
            
            setBinDeps ( env, b[0], deps )
            
