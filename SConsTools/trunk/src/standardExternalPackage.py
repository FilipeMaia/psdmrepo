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
def standardExternalPackage ( package, prefix=None, 
                              inc_dir=None, lib_dir=None, bin_dir=None, py_dir=None,
                              libraries=None, binaries=None ) :
    
    pkg = os.path.basename(os.getcwd())
    trace ( "SConscript in `"+pkg+"'", "SConscript", 1 )
    
    env = DefaultEnvironment()
    
    trace ( "prefix: %s" % prefix, "standardExternalPackage", 3 )
    
    arch = env['LUSI_ARCH']
    
    # link include directory
    inc_dir = _absdir ( prefix, inc_dir )
    if inc_dir :
        trace ( "include_dir: %s" % inc_dir, "standardExternalPackage", 5 )
        env.Symlink ( Dir(pjoin(env.subst("$ARCHINCDIR"),package)), Dir(inc_dir) )
    
    # link python directory
    py_dir = _absdir ( prefix, py_dir )
    if py_dir :
        trace ( "py_dir: %s" % py_dir, "standardExternalPackage", 5 )
        env.Symlink ( Dir(pjoin(env.subst("$PYDIR"),package)), Dir(py_dir) )
    
    # link all libraries
    lib_dir = _absdir ( prefix, lib_dir )
    if lib_dir :
        trace ( "lib_dir: %s" % lib_dir, "standardExternalPackage", 5 )
        if not libraries : libraries = os.listdir(lib_dir)
        for f in libraries :
            loc = pjoin(lib_dir,f)
            if os.path.isfile(loc) :
                env.Symlink ( pjoin(env.subst("$LIBDIR"),f), loc )

    # link all executables
    bin_dir = _absdir ( prefix, bin_dir )
    if bin_dir :
        trace ( "bin_dir: %s" % bin_dir, "standardExternalPackage", 5 )
        if not binaries : binaries = os.listdir(bin_dir)
        for f in binaries :
            loc = pjoin(bin_dir,f)
            if os.path.isfile(loc) :
                env.Symlink ( pjoin(env.subst("$BINDIR"),f), loc )
