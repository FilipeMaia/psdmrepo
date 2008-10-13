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

#
# This is the content of the standard SConscript
#
def standardSConscript() :

    pkg = os.path.basename(os.getcwd())
    trace ( "SConscript in `"+pkg+"'", "SConscript", 1 )

    env = DefaultEnvironment()
    
    bindir = env['BINDIR']
    libdir = env['LIBDIR']
    pydir = env['PYDIR']
    
    #
    # Process src/ directory, make library from all compilable files
    #
    libsrcs = Glob("src/*.cpp", source=True )
    if libsrcs :        
        trace ( "libsrcs = "+pformat([str(s) for s in libsrcs]), "SConscript", 2 )

        # c++ files area compiled into library  
        if libsrcs : 
            lib = env.SharedLibrary ( pkg, source=libsrcs )
            env.Install ( libdir, source=lib )

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
    app_files = Glob("app/*.cpp", source=True, strings=True )
    if app_files :

        trace ( "app_files = "+pformat(app_files), "SConscript", 2 )
        
        # for now build separate application for every C++ source file
        for s in app_files :
            
            obj = env.Object( source = s )
            bin = env.Program( source=obj )
            env.Install ( bindir, source=bin )
            
            trace ( "children = " + pformat([ str(c) for c in obj[0].children()]), "SConscript", 4 )
