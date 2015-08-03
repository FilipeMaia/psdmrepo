"""SCons.Tool.pkg_list

Tool-specific initialization for pkg_list .

AUTHORS:
 - Andy Salnikov

"""

import os
import sys

import SCons
from SCons.Builder import Builder
from SCons.Action import Action

from SConsTools.trace import *
from SConsTools.scons_functions import *

def _fmtList(lst):
    return '[' + ','.join(map(str, lst)) + ']'

class _makePkgList:

    def __call__(self, target, source, env) :
        """Target should be a single file, no source is needed"""
        if len(target) != 1 : fail("unexpected number of targets for PkgList: " + str(target))
        if len(source) != 0 : fail("unexpected number of sources for PkgList: " + str(source))

        target = env['PKG_LIST_FILE']
        trace("Executing PkgList `%s'" % (target,), "makePkgList", 3)

        # may need to make a directory for target
        targetdir = os.path.normpath(os.path.dirname(target))
        if not os.path.isdir(targetdir): 
            os.makedirs(targetdir)

        out = open(target, "w")

        # build requirements list
        for pkg, pkginfos in env['EXT_PACKAGE_INFO'].iteritems():
            trace("package %s, pkginfos %s" % (pkg, pkginfos), "makePkgList", 4)
            for pkginfo in pkginfos:
                pkginfo = [env.subst(info) for info in pkginfo]
                print >> out, "%s$%s" % (pkg, '%'.join(pkginfo))

        # add also scons, need to guess version name and python version (which is python running scons)
        pkg = "scons-%s-python%d.%d" % (SCons.__version__, sys.version_info[0], sys.version_info[1])
        print >> out, "scons$scons%%%s%%python%d.%d" % (SCons.__version__, sys.version_info[0], sys.version_info[1])

        out.close()

    def strfunction(self, target, source, env):
        try :
            return "Creating package list file: `" + env['PKG_LIST_FILE'] + "'"
        except :
            return 'PkgList(target=' + _fmtlist(target) + ')'

def create_builder(env):
    try:
        builder = env['BUILDERS']['PackageList']
    except KeyError:
        builder = SCons.Builder.Builder(action=_makePkgList())
        env['BUILDERS']['PackageList'] = builder

    return builder

def generate(env):
    """Add Builders and construction variables for making package list file."""

    # Create the PythonExtension builder
    create_builder(env)

    trace("Initialized pkg_list tool", "pkg_list", 2)

def exists(env):
    return True
