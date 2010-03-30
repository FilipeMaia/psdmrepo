#===============================================================================
#
# SConscript fuction for standard external package
#
# $Id$
#
#===============================================================================

import os
import sys
from os.path import join as pjoin
from fnmatch import fnmatch

from SCons.Defaults import *
from SCons.Script import *

from SConsTools.trace import *
from SConsTools.dependencies import *

#
# This is an interface package for the external package. We wan to make
# symlinks to the include files, libs and binaries
#

#
# Find a correct prefix directory.
#
def _prefix ( prefix, env ):
    
    # if prefix ends with LUSI_ARCH, discard it
    head, tail = os.path.split( prefix )
    if not tail : head, tail = os.path.split( head )        
    if tail == env['LUSI_ARCH'] : prefix = head

    # First try $LUSI_ARCH
    pfx = pjoin( prefix, env['LUSI_ARCH'] )
    if os.path.isdir( pfx ) : return pfx

    # for 'prof' try to substitute with 'dbg'
    if env['LUSI_ARCH_OPT'] == 'prof' :
        pfx = pjoin( prefix, env['LUSI_ARCH_BASE']+'-dbg' )
        if os.path.isdir( pfx ) : return pfx

    # Then try $LUSI_ARCH_BASE
    pfx = pjoin( prefix, env['LUSI_ARCH_BASE'] )
    if os.path.isdir( pfx ) : return pfx

    # otherwise try 'opt'
    pfx = pjoin( prefix, env['LUSI_ARCH_BASE']+'-opt' )
    if os.path.isdir( pfx ) : return pfx

    # nothing works, just return what we have
    return prefix


# build package name from prefix and directory
def _absdir ( prefix, dir ):
    if dir is None: 
        return None
    if prefix and not os.path.isabs( dir ) :
        dir = pjoin( prefix, dir )
    if not os.path.isdir( dir ) :
        dir = None
    return dir

def _glob ( dir, patterns ):
    
    if patterns is None :
        return os.listdir(dir)
    
    # patterns could be space-separated string of patterns
    if isinstance(patterns,(str,unicode)) : 
        patterns = patterns.split()
    if not patterns : return []

    result = []
    for l in os.listdir(dir) :
        for p in patterns :
            if fnmatch ( l, p ) : result.append(l)
            
    return result


#
# Define all builders for the external package
#
def standardExternalPackage ( package, **kw ) :
    """ Understands following keywords (all are optional):
        PREFIX   - top directory of the external package
        INCDIR   - include directory, absolute or relative to PREFIX
        INCLUDES - include files to copy (space-separated list of patterns)
        PYDIR    - Python src directory, absolute or relative to PREFIX
        LINKPY   - Python files to link (patterns), or all files if not present
        PYDIRSEP - if present and evaluates to True installs python code to a 
                   separate directory arch/$LUSI_ARCH/python/<package>
        LIBDIR   - libraries directory, absolute or relative to PREFIX
        LINKLIBS - library names to link, or all libs if not present
        BINDIR   - binaries directory, absolute or relative to PREFIX
        LINKBINS - binary names to link, or all libs if not present
        PKGLIBS  - names of libraries that have to be linked for this package
        DEPS     - names of other packages that we depend upon
    """
    
    pkg = os.path.basename(os.getcwd())
    trace ( "Standard SConscript for external package `"+package+"'", "SConscript", 1 )
    
    env = DefaultEnvironment()
    
    prefix = _prefix ( kw.get('PREFIX'), env )
    trace ( "prefix: %s" % prefix, "standardExternalPackage", 3 )
    
    # link include directory
    inc_dir = _absdir ( prefix, kw.get('INCDIR') )
    if inc_dir is not None :

        trace ( "include_dir: %s" % inc_dir, "standardExternalPackage", 5 )
        
        # make 'geninc' directory if not there yet
        archinc = Dir(env.subst("$ARCHINCDIR"))
        archinc = str(archinc)
        if not os.path.isdir( archinc ) : os.makedirs( archinc )

        includes = kw.get('INCLUDES')
        if not includes :
            
            # link the whole include directory
            target = pjoin(archinc,package)
            if not os.path.lexists(target) : os.symlink ( inc_dir, target )
            
        else:

            # make target package directory if needed
            targetdir = pjoin( archinc, package )
            if not os.path.isdir( targetdir ) : os.makedirs( targetdir )
            
            # copy individual files
            includes = _glob ( inc_dir, includes )
            for inc in includes :
                loc = pjoin( inc_dir, inc )
                target = pjoin( targetdir, inc )
                targ = env.Symlink( target, loc )
                trace ( "linkinc: %s -> %s" % (str(targ[0]),loc), "standardExternalPackage", 5 )

    
    # link python directory
    py_dir = _absdir ( prefix, kw.get('PYDIR') )
    if py_dir is not None :
        trace ( "py_dir: %s" % py_dir, "standardExternalPackage", 5 )
        if kw.get('PYDIRSEP',False) :
            # make a link to the whole dir
            targ = env.Symlink ( Dir(pjoin(env.subst("$PYDIR"),package)), Dir(py_dir) )
            env['ALL_TARGETS']['LIBS'].extend ( targ )
        else :
            # make links for every file in the directory
            files = kw.get('LINKPY')
            files = _glob ( py_dir, files )
            for f in files :
                loc = pjoin(py_dir,f)
                if not os.path.isdir(loc) :
                    targ = env.Symlink ( pjoin(env.subst("$PYDIR"),f), loc )
                else :
                    targ = env.Symlink ( Dir(pjoin(env.subst("$PYDIR"),f)), Dir(loc) )
                trace ( "linkpy: %s -> %s" % (str(targ[0]),loc), "standardExternalPackage", 5 )
                env['ALL_TARGETS']['LIBS'].extend( targ )
            
    
    # link all libraries
    lib_dir = _absdir ( prefix, kw.get('LIBDIR') )
    if lib_dir is not None :
        trace ( "lib_dir: %s" % lib_dir, "standardExternalPackage", 5 )
        
        # make a list of libs to link
        libraries = kw.get('LINKLIBS')
        trace ( "libraries: %s" % libraries, "standardExternalPackage", 5 )
        libraries = _glob ( lib_dir, libraries )
            
        trace ( "libraries: %s" % libraries, "standardExternalPackage", 5 )
        for f in libraries :
            loc = pjoin(lib_dir,f)
            if os.path.isfile(loc) :
                #targ = env.Install( "$LIBDIR", loc )
                targ = env.Symlink ( pjoin(env.subst("$LIBDIR"),f), loc )
                trace ( "linklib: %s -> %s" % (str(targ[0]),loc), "standardExternalPackage", 5 )
                env['ALL_TARGETS']['LIBS'].extend ( targ )

    # link all executables
    bin_dir = _absdir ( prefix, kw.get('BINDIR') )
    if bin_dir is not None :
        trace ( "bin_dir: %s" % bin_dir, "standardExternalPackage", 5 )
        
        # make list of binaries to link
        binaries = kw.get('LINKBINS')
        binaries = _glob ( bin_dir, binaries )
        
        for f in binaries :
            loc = pjoin(bin_dir,f)
            if os.path.isfile(loc) :
                targ = env.Symlink ( pjoin(env.subst("$BINDIR"),f), loc )
                env['ALL_TARGETS']['BINS'].extend ( targ )

    # add my libs to a package tree
    addPkgLibs ( env, package, kw.get('PKGLIBS',[]) )
    
    # add packages that I depend on
    setPkgDeps ( env, package, kw.get('DEPS',[]) )
