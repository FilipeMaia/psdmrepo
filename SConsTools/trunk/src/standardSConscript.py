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

# normalize source file path
def _normbinsrc ( dir, f ):
    if not os.path.split(f)[0] :
        return os.path.join(dir,f)
    else :
        return f

# get list of strings from kw, single string is split
def _getkwlist ( kw, name ):
    if name in kw :
        res = kw[name]
        if isinstance(res,(str,unicode)) : res = res.split()
        return res
    return []

# get package name
def _getpkg ( kw ) :
    pkg = kw.get('package', None )
    if not pkg : pkg = os.path.basename(os.getcwd())
    return pkg
    

#
# This is the content of the standard SConscript
#
def standardSConscript( **kw ) :

    """ Understands following keywords, all optional:
        BINS - dictionary of executables and their corresponding source files
        LIBS - list of additional libraries that have to be linked with applications
    """

    pkg = _getpkg ( kw )
    trace ( "Standard SConscript in `"+pkg+"'", "SConscript", 1 )

    ukw = kw.copy()
    ukw['package'] = pkg
    
    standardLib( **ukw )
    standardPyLib( **ukw )
    standardScripts( **ukw )
    standardBins ( **ukw )
    standardTests ( **ukw )

#
# Process src/ directory, make library from all compilable files
#
def standardLib( **kw ) :
    
    libsrcs = Flatten ( [ Glob("src/*."+ext, source=True ) for ext in _cplusplus_ext ] )
    if libsrcs :
        
        trace ( "libsrcs = "+pformat([str(s) for s in libsrcs]), "SConscript", 2 )

        pkg = _getpkg( kw )
        
        env = DefaultEnvironment()
        libdir = env['LIBDIR']

        lib = env.SharedLibrary ( pkg, source=libsrcs )
        env.Install ( libdir, source=lib )
        
        # get the list of dependencies for this package
        deps = findAllDependencies( lib[0] )
        setPkgDeps ( env, pkg, deps )
        trace ( "deps = " + pformat(deps), "SConscript", 4 )

        # get the list of libraries need for this package
        libs = [pkg] + _getkwlist ( kw, 'LIBS' )
        setPkgLibs ( env, pkg, libs )
        
#
# Process src/ directory, link python sources
#
def standardPyLib( **kw ) :
    
    pysrcs = Glob("src/*.py", source=True, strings=True )
    if pysrcs :
        
        pkg = _getpkg( kw )
        
        env = DefaultEnvironment()
        pydir = env['PYDIR']

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
def standardScripts( **kw ) :
    
    app_files = Glob("app/*", source=True, strings=True )
    if app_files :

        env = DefaultEnvironment()
        bindir = env['BINDIR']

        scripts = [ f for f in app_files if not os.path.splitext(f)[1] and os.path.isfile(f) ]
        trace ( "scripts = "+pformat(scripts), "SConscript", 2 )

        # Scripts are copied to bin/ directory
        for s in scripts : 
            env.ScriptInstall ( os.path.join(bindir,os.path.basename(s)), s )

#
# Process app/ directory, build all executables from C++ sources
#
def standardBins( **kw ) :
    _standardBins ( 'app', 'BINS', True, **kw )

#
# Process test/ directory, build all executables from C++ sources
#
def standardTests( **kw ) :
    _standardBins ( 'test', 'TESTS', False, **kw )

#
# Build binaries, possibly install them
#
def _standardBins( appdir, binenv, install, **kw ) :

    # make list of binaries and their dependencies if it has not been passed to us
    bins = kw.get(binenv,{})
    if bins :
        for k in bins.iterkeys() :
            src = bins[k]
            if isinstance(src,(str,unicode)) : src = src.split()
            src = [ _normbinsrc(appdir,s) for s in src ]
            bins[k] = src
    else :
        cpps = Flatten ( [ Glob(appdir+"/*."+ext, source=True, strings=True ) for ext in _cplusplus_ext ] )
        for f in cpps :
            bin = os.path.splitext(os.path.basename(f))[0]
            bins[bin] = [ f ]
            
    # make rules for the binaries
    if bins :

        trace ( "bins = "+pformat(bins), "SConscript", 2 )

        env = DefaultEnvironment()
        bindir = env['BINDIR']
        
        # Program options
        binkw = {}
        binkw['LIBS'] = _getkwlist ( kw, 'LIBS' )
        #binkw['LIBS'].insert ( 0, _getpkg( kw ) )
        binkw['LIBDIRS'] = _getkwlist ( kw, 'LIBDIRS' )
    
        for bin, srcs in bins.iteritems() :
            
            b = env.Program( bin, source=srcs, **binkw )
            if install : env.Install ( bindir, source=b )
            
            deps = findAllDependencies( b[0] )
            trace ( bin+" deps = " + pformat(deps), "SConscript", 4 )
            
            setBinDeps ( env, b[0], deps )
