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
    
    def __init__(self, relative=False):
        self._rel = relative

    def __call__ ( self, target, source, env ) :
        """Both target and source should be a single file"""
        if len(target) != 1 :
            fail ( "unexpected number of targets for symlink: "+str(target) )
        if len(source) != 1 :
            fail ( "unexpected number of sources for symlink: "+str(source) )
    
        if self._rel:
            # rel_path behaves differently when target is a directory
            if target[0].__class__ is SCons.Node.FS.Dir:
                source = target[0].get_dir().rel_path(source[0])
            else:
                source = target[0].rel_path(source[0])
        else:
            source = str(source[0].abspath)
        target = str(target[0])
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

def create_builders(env):
    try:
        builder = env['BUILDERS']['Symlink']
    except KeyError:
        builder = SCons.Builder.Builder(action = _makeSymlink())
        env['BUILDERS']['Symlink'] = builder
    try:
        builder = env['BUILDERS']['SymlinkRel']
    except KeyError:
        builder = SCons.Builder.Builder(action = _makeSymlink(True))
        env['BUILDERS']['SymlinkRel'] = builder

def generate(env):
    """Add Builders and construction variables for making symlinks."""

    # Create the PythonExtension builder
    create_builders(env)

    trace ( "Initialized symlink tool", "symlink", 2 )

def exists(env):
    return True
