"""SCons.Tool.script_install

Tool-specific initialization for ScriptInstall builder.

AUTHORS:
 - Andy Salnikov

"""

import os
import re

import SCons
from SCons.Builder import Builder
from SCons.Action import Action

from SConsTools.trace import *
from SConsTools.scons_functions import *

def _fmtList ( lst ):
    return '[' + ','.join(map(str, lst)) + ']'

class _scriptInstall :
    
    def __call__ ( self, target, source, env ) :
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

    def strfunction ( self, target, source, env ):
        try :
            return "Installing script: \"" + str(target[0]) + "\""
        except :
            return 'ScriptInstall('+_fmtlist(target)+', '+_fmtlist(source)+')'

def create_builder(env):
    try:
        builder = env['BUILDERS']['ScriptInstall']
    except KeyError:
        builder = SCons.Builder.Builder(action = _scriptInstall())
        env['BUILDERS']['ScriptInstall'] = builder

    return builder

def generate(env):
    """Add Builders and construction variables for making symlinks."""

    # Create the PythonExtension builder
    create_builder(env)

    trace ( "Initialized ScriptInstall tool", "ScriptInstall", 2 )

def exists(env):
    return True
