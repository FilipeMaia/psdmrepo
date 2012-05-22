"""SCons.Tool.release_install

Tool-specific initialization for release_install builder.

AUTHORS:
 - Andy Salnikov

"""

import os

import SCons
from SCons.Builder import Builder
from SCons.Action import Action

from SConsTools.trace import *
from SConsTools.scons_functions import *


def _fmtList(lst):
    return '[' + ','.join(map(str, lst)) + ']'

class _makeReleaseInstall:

    def __call__(self, target, source, env) :
        """Target should be a single file, no source is needed"""
        if len(target) != 1 : fail("unexpected number of targets for ReleaseInstall: " + str(target))
        if len(source) != 0 : fail("unexpected number of sources for ReleaseInstall: " + str(source))

        destdir = str(target[0])
        trace("Executing ReleaseInstall: destdir=%s" % (destdir,), "makeReleaseInstall", 3)

        # play safe, destination directory must not exist
        if os.path.exists(destdir): fail("ReleaseInstall: destination directory already exists: " + destdir)

        # make dest directory
        os.makedirs(destdir)

        # copy all files/directories except for build-time only files
        for file in os.listdir('.'):
            
            if file not in ['build', '.sconsign.dblite']:
                if os.path.isdir(file):
                    dst = os.path.join(destdir, file)
                    trace("ReleaseInstall: copy dir `%s' to `%s'" % (file, dst), "makeReleaseInstall", 3)
                    shutil.copytree(file, dst, True)
                else:
                    dst = os.path.join(destdir, file)
                    trace("ReleaseInstall: copy file `%s' to `%s'" % (file, dst), "makeReleaseInstall", 3)
                    shutil.copy2(file, dst)

    def strfunction(self, target, source, env):
        try :
            return "Install release in " + str(target[0])
        except :
            return 'ReleaseInstall(' + _fmtlist(target) + ')'

def create_builder(env):
    try:
        builder = env['BUILDERS']['ReleaseInstall']
    except KeyError:
        builder = SCons.Builder.Builder(action=_makeReleaseInstall())
        env['BUILDERS']['ReleaseInstall'] = builder

    return builder

def generate(env):
    """Add special Builder for installing release to a new location."""

    # Create the PythonExtension builder
    create_builder(env)

    trace("Initialized release_install tool", "release_install", 2)

def exists(env):
    return True
