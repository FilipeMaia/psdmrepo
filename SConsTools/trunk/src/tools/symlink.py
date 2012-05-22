"""SCons.Tool.symlink

Tool-specific initialization for symlink builder.

AUTHORS:
 - Andy Salnikov

"""

import os

import SCons
from SCons.Builder import Builder
from SCons.Action import Action

from SConsTools.trace import *
from SConsTools.scons_functions import *

def _fmtList ( lst ):
    return '[' + ','.join(map(str, lst)) + ']'

class _makeSymlink :

    def __call__ ( self, target, source, env ) :
        """Both target and source should be a single file"""
        if len(target) != 1 :
            fail ( "unexpected number of targets for symlink: "+str(target) )
        if len(source) != 1 :
            fail ( "unexpected number of sources for symlink: "+str(source) )
    
        target = str(target[0])
        source = str(source[0].abspath)
        trace ( "Executing symlink `%s' -> `%s'" % ( target, source ), "makeSymlink", 3 )
    
        # may need to make a directory for target
        targetdir = os.path.dirname ( target )
        if not os.path.isdir( targetdir ) : os.makedirs( targetdir )
    
        # if target already exists then remove it
        if os.path.islink( target ) : os.remove( target )
        
        # create symlink now
        os.symlink ( source, target )

    def strfunction ( self, target, source, env ):
        try :
            return "Creating symlink: `" + str(target[0]) + "' -> `" + str(source[0]) + "'"
        except :
            return 'MakeSymlink('+_fmtlist(target)+', '+_fmtlist(source)+')'

def create_builder(env):
    try:
        builder = env['BUILDERS']['Symlink']
    except KeyError:
        builder = SCons.Builder.Builder(action = _makeSymlink())
        env['BUILDERS']['Symlink'] = builder

    return builder

def generate(env):
    """Add Builders and construction variables for making symlinks."""

    # Create the PythonExtension builder
    create_builder(env)

    trace ( "Initialized symlink tool", "symlink", 2 )

def exists(env):
    return True
