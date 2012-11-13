"""SCons.Tool.pycompile

Tool-specific initialization for PyCompile builder.

AUTHORS:
 - Andy Salnikov

"""

import os
import subprocess

import SCons
from SCons.Builder import Builder
from SCons.Action import Action

from SConsTools.trace import *
from SConsTools.scons_functions import *

def _fmtList ( lst ):
    return '[' + ','.join(map(str, lst)) + ']'

class _pyCompile :
    
    def __call__( self, target, source, env ) :
        """Both target and source should be a single file"""
        if len(target) != 1 :
            fail ( "unexpected number of targets for pyCompile: "+str(target) )
        if len(source) != 1 :
            fail ( "unexpected number of sources for pyCompile: "+str(source) )
    
        target = str(target[0])
        source = str(source[0])
        trace ( "Executing pycompile `%s'" % ( source ), "pyCompile", 3 )
    
        # we need to compile it using "standard" python which may be
        # different from python running SCons
        cmd = [env['PYTHON_BIN'], '-c', 'import py_compile; py_compile.compile("%s", "%s", doraise=True)' % (source, target)]
        rc = subprocess.call(cmd)
        if rc != 0:
            return -1
        
    def strfunction ( self, target, source, env ):
        try :
            return "Compiling Python code: \"" + str(source[0]) + "\""
        except :
            return 'PyCompile('+_fmtlist(target)+', '+_fmtlist(source)+')'

def create_builder(env):
    try:
        builder = env['BUILDERS']['PyCompile']
    except KeyError:
        builder = SCons.Builder.Builder(action = _pyCompile())
        env['BUILDERS']['PyCompile'] = builder

    return builder

def generate(env):
    """Add Builders and construction variables for making symlinks."""

    # Create the PythonExtension builder
    create_builder(env)

    trace ( "Initialized PyCompile tool", "PyCompile", 2 )

def exists(env):
    return True
