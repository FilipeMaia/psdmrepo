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

def setupBuilders ( env ):
    
    env.Append( BUILDERS = { 'Symlink' : Builder ( action = _makeSymlink ) } )
    env.Append( BUILDERS = { 'PyCompile' : Builder ( action = _pyCompile ) } )
    