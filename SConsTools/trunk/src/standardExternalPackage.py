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

# build package name from prefix and directory
def _absdir ( prefix, dir ):
    if not dir : 
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
        PYDIR    - Python src directory, absolute or relative to PREFIX
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
    
    prefix = kw.get('PREFIX',None)
    trace ( "prefix: %s" % prefix, "standardExternalPackage", 3 )
    
    # link include directory
    inc_dir = _absdir ( prefix, kw.get('INCDIR',None) )
    if inc_dir :

        trace ( "include_dir: %s" % inc_dir, "standardExternalPackage", 5 )
        
        # make 'geninc' directory if not there yet
        archinc = Dir(env.subst("$ARCHINCDIR"))
        archinc = str(archinc)
        if not os.path.isdir( archinc ) : os.makedirs( archinc )

        target = pjoin(archinc,package)
        if not os.path.lexists(target) : os.symlink ( inc_dir, target )
        
    
    # link python directory
    py_dir = _absdir ( prefix, kw.get('PYDIR',None) )
    if py_dir :
        trace ( "py_dir: %s" % py_dir, "standardExternalPackage", 5 )
        if kw.get('PYDIRSEP',False) :
            # make a link to the whole dir
            targ = env.Symlink ( Dir(pjoin(env.subst("$PYDIR"),package)), Dir(py_dir) )
            env['ALL_TARGETS']['LIBS'].extend ( targ )
        else :
            # make links for every file in the directory
            files = os.listdir(py_dir)
            for f in files :
                loc = pjoin(py_dir,f)
                if not os.path.isdir(loc) :
                    targ = env.Symlink ( pjoin(env.subst("$PYDIR"),f), loc )
                    env['ALL_TARGETS']['LIBS'].extend( targ )
            
    
    # link all libraries
    lib_dir = _absdir ( prefix, kw.get('LIBDIR',None) )
    if lib_dir :
        trace ( "lib_dir: %s" % lib_dir, "standardExternalPackage", 5 )
        
        # make a list of libs to link
        libraries = kw.get('LINKLIBS',None)
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
    bin_dir = _absdir ( prefix, kw.get('BINDIR',None) )
    if bin_dir :
        trace ( "bin_dir: %s" % bin_dir, "standardExternalPackage", 5 )
        
        # make list of binaries to link
        binaries = kw.get('LINKBINS',None)
        binaries = _glob ( bin_dir, binaries )
        
        for f in binaries :
            loc = pjoin(bin_dir,f)
            if os.path.isfile(loc) :
                targ = env.Symlink ( pjoin(env.subst("$BINDIR"),f), loc )
                env['ALL_TARGETS']['BINS'].extend ( targ )

    # add my libs to a package tree
    setPkgLibs ( env, package, kw.get('PKGLIBS',[]) )
    
    # add packages that I depend on
    setPkgDeps ( env, package, kw.get('DEPS',[]) )
