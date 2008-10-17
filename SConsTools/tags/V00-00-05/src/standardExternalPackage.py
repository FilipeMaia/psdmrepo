#===============================================================================
#
# SConscript fuction for standard external package
#
# $Id: standardSConscript.py 12 2008-10-09 00:27:34Z salnikov $
#
#===============================================================================

import os
import sys
from os.path import join as pjoin

from SCons.Defaults import *
from SCons.Script import *

from SConsTools.trace import *
from SConsTools.dependencies import *

#
# This is an interface package for the external package. We wan to make
# symlinks to the include files, libs and binaries
#

# build absolute package name from prefix and directory
def _absdir ( prefix, dir ):
    if not dir : 
        return None
    if prefix and not os.path.isabs( dir ) :
        dir = pjoin( prefix, dir )
    if not os.path.isdir( dir ) :
        dir = None
    return dir

#
# Define all builders for the external package
#
def standardExternalPackage ( package, **kw ) :
    """ Understands following keywords (all are optional):
        PREFIX   - top directory of the external package
        INDIR    - include directory, absolute or relative to PREFIX 
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
    trace ( "SConscript in `"+pkg+"'", "SConscript", 1 )
    
    env = DefaultEnvironment()
    
    prefix = kw.get('PREFIX',None)
    trace ( "prefix: %s" % prefix, "standardExternalPackage", 3 )
    
    arch = env['LUSI_ARCH']
    
    # link include directory
    inc_dir = _absdir ( prefix, kw.get('INCDIR',None) )
    if inc_dir :
        trace ( "include_dir: %s" % inc_dir, "standardExternalPackage", 5 )
        env.Symlink ( Dir(pjoin(env.subst("$ARCHINCDIR"),package)), Dir(inc_dir) )
    
    # link python directory
    py_dir = _absdir ( prefix, kw.get('PYDIR',None) )
    if py_dir :
        trace ( "py_dir: %s" % py_dir, "standardExternalPackage", 5 )
        if kw.get('PYDIRSEP',False) :
            # make a link to the whole dir
            env.Symlink ( Dir(pjoin(env.subst("$PYDIR"),package)), Dir(py_dir) )
        else :
            # make links for every file in the directory
            files = os.listdir(py_dir)
            for f in files :
                loc = pjoin(py_dir,f)
                if not os.path.isdir(loc) :
                    env.Symlink ( pjoin(env.subst("$PYDIR"),f), loc )
            
    
    # link all libraries
    lib_dir = _absdir ( prefix, kw.get('LIBDIR',None) )
    if lib_dir :
        trace ( "lib_dir: %s" % lib_dir, "standardExternalPackage", 5 )
        libraries = kw.get('LINKLIBS',None)
        if not libraries : libraries = os.listdir(lib_dir)
        for f in libraries :
            loc = pjoin(lib_dir,f)
            if os.path.isfile(loc) :
                env.Symlink ( pjoin(env.subst("$LIBDIR"),f), loc )

    # link all executables
    bin_dir = _absdir ( prefix, kw.get('BINDIR',None) )
    if bin_dir :
        trace ( "bin_dir: %s" % bin_dir, "standardExternalPackage", 5 )
        binaries = kw.get('LINKBINS',None)
        if not binaries : binaries = os.listdir(bin_dir)
        for f in binaries :
            loc = pjoin(bin_dir,f)
            if os.path.isfile(loc) :
                env.Symlink ( pjoin(env.subst("$BINDIR"),f), loc )

    # add my libs to a package tree
    pkg_libs = kw.get('PKGLIBS',None)
    if pkg_libs :
        setPkgLibs ( env, package, pkg_libs )
    
    # add packages that I depend on
    deps = kw.get('DEPS',None)
    if deps :
        setPkgDeps ( env, package, deps )
