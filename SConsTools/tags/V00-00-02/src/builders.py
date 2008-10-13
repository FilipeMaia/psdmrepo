#!/bin/env scons
#===============================================================================
#
# Main SCons script for LUSI release building
#
# $Id$
#
#===============================================================================

import os
import sys
import re

from SCons.Script import *

from trace import *
from scons_functions import *


def _makeSymlink ( target, source, env ) :
    """Both target and source should be a single file"""
    if len(target) != 1 :
        fail ( "unexpected number of targets for symlink: "+str(target) )
    if len(source) != 1 :
        fail ( "unexpected number of sources for symlink: "+str(source) )

    target = str(target[0])
    source = str(source[0].abspath)
    trace ( "Executing symlink `%s' -> `%s'" % ( target, source ), "makeSymlink", 3 )

    # if target already exists then remove it
    if os.path.islink( target ) : os.remove( target )
    os.symlink ( source, target )

def _pyCompile ( target, source, env ) :
    """Both target and source should be a single file"""
    if len(target) != 1 :
        fail ( "unexpected number of targets for pyCompile: "+str(target) )
    if len(source) != 1 :
        fail ( "unexpected number of sources for pyCompile: "+str(source) )

    target = str(target[0])
    source = str(source[0])
    trace ( "Executing pycompile `%s'" % ( source ), "pyCompile", 3 )

    try :
        import py_compile
        py_compile.compile ( source, target, doraise = True )
    except py_compile.PyCompileError, e :
        print str(e)
        return -1

def _scriptInstall ( target, source, env ) :
    """Both target and source should be a single file"""
    if len(target) != 1 :
        fail ( "unexpected number of targets for scriptInstall: "+str(target) )
    if len(source) != 1 :
        fail ( "unexpected number of sources for scriptInstall: "+str(source) )
    source = str(source[0])
    target = str(target[0])

    subs = env.get('SCRIPT_SUBS',{})
        
    # read into memory
    f = open( source )
    data = f.read()
    f.close()

    # split file into appearances of "@...@"
    data = re.split ( '(@[^@]*@)', data )
    for i in range(len(data)) :
        w = data[i]
        if len(w)>2 and w.startswith('@') :
             data[i] = subs.get(w[1:-1],w)
    data = "".join(data)

    # store it
    f = open( target, 'w' )
    f.write ( "".join(data) )
    f.close()
    os.chmod ( target, 0755 )

def setupBuilders ( env ):
    
    env.Append( BUILDERS = { 'Symlink' : Builder ( action = _makeSymlink ) } )
    env.Append( BUILDERS = { 'PyCompile' : Builder ( action = _pyCompile ) } )
    env.Append( BUILDERS = { 'ScriptInstall' : Builder ( action = _scriptInstall ) } )
