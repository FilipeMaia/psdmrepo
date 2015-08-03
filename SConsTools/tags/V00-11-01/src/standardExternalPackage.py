#===============================================================================
#
# SConscript function for standard external package
#
# $Id$
#
#===============================================================================

import os
import sys
import types
from os.path import join as pjoin
from fnmatch import fnmatch

from SCons.Defaults import *
from SCons.Script import *

from SConsTools.trace import *
from SConsTools.dependencies import *

from scons_functions import fail, warning

#
# This is an interface package for the external package. We wan to make
# symlinks to the include files, libs and binaries
#

#
# Find a correct prefix directory, returns tuple (prefix, arch)
#
def _prefix(prefix, env):

    # if prefix ends with SIT_ARCH, discard it for now, find real arch
    prefix = env.subst(prefix)
    head, tail = os.path.split(prefix)
    if not tail : head, tail = os.path.split(head)
    if tail == env['SIT_ARCH']: prefix = head

    # First try $SIT_ARCH
    arch = env['SIT_ARCH']
    if os.path.isdir(pjoin(prefix, arch)) and not os.path.islink(pjoin(prefix, arch)):
        return (prefix, arch)

    # for 'prof' try to substitute with 'dbg'
    if env['SIT_ARCH_OPT'] == 'prof' :
        arch = env['SIT_ARCH_BASE_DBG']
        if os.path.isdir(pjoin(prefix, arch)) and not os.path.islink(pjoin(prefix, arch)):
            return (prefix, arch)

    # otherwise try 'opt'
    arch = env['SIT_ARCH_BASE_OPT']
    if os.path.isdir(pjoin(prefix, arch)) and not os.path.islink(pjoin(prefix, arch)):
        return (prefix, arch)
    
    # Then try $SIT_ARCH_BASE
    arch = env['SIT_ARCH_BASE']
    if os.path.isdir(pjoin(prefix, arch)) and not os.path.islink(pjoin(prefix, arch)):
        return (prefix, arch)

    # nothing works, just return what we have
    return (prefix, "")


def _glob(dir, patterns):

    if patterns is None :
        return os.listdir(dir)

    # patterns could be space-separated string of patterns
    if isinstance(patterns, (str, unicode)) :
        patterns = patterns.split()
    if not patterns : return []

    result = []
    for l in os.listdir(dir) :
        for p in patterns :
            if fnmatch(l, p) : result.append(l)

    return result

def _get_dir(package, dirvar, kw, env, prefix):
    """
    Locate directory name specified as the variable in keyword arguments,
    if directory is specified then append prefix if needed and check that directory exists
    """
    dir = kw.get(dirvar)
    if dir is not None:
        dir = env.subst(dir)
        if prefix and not os.path.isabs(dir) :
            dir = pjoin(prefix, dir)
        if not os.path.isdir(dir):
            msg = "Building external package %s: missing %s directory: %s" % (package, dirvar, dir)
            if kw.get('OPTIONAL'):
                dir = None
                warning(msg)
            else:
                fail(msg)
    return dir

#
# Define all builders for the external package
#
def standardExternalPackage(package, **kw) :
    """ Understands following keywords (all are optional):
        PREFIX   - top directory of the external package
        INCDIR   - include directory, absolute or relative to PREFIX
        INCLUDES - include files to copy (space-separated list of patterns)
        PYDIR    - Python src directory, absolute or relative to PREFIX
        LINKPY   - Python files to link (patterns), or all files if not present
        PYDIRSEP - if present and evaluates to True installs python code to a 
                   separate directory arch/$SIT_ARCH/python/<package>
        LIBDIR   - libraries directory, absolute or relative to PREFIX
        COPYLIBS - library names to copy
        LINKLIBS - library names to link, or all libs if LINKLIBS and COPYLIBS are empty
        BINDIR   - binaries directory, absolute or relative to PREFIX
        LINKBINS - binary names to link, or all binaries if not present
        PKGLIBS  - names of libraries that have to be linked for this package
        DEPS     - names of other packages that we depend upon
        PKGINFO  - package information, such as RPM package name
        OPTIONAL - If true then missing directories do not cause build errors
        DOCGEN   - if this is is a string or list of strings then it should be name(s) of document 
                   generators, otherwise it is a dict with generator name as key and a list of 
                   file/directory names as values (may also be a string).
    """

    pkg = os.path.basename(os.getcwd())
    trace("Standard SConscript for external package `" + package + "'", "SConscript", 1)

    env = DefaultEnvironment()

    prefix, arch = _prefix(kw.get('PREFIX'), env)
    trace("prefix, arch: %s, %s" % (prefix, arch), "standardExternalPackage", 3)
    if arch: prefix = os.path.join(prefix, arch)
    trace("prefix: %s" % prefix, "standardExternalPackage", 3)

    # link include directory
    inc_dir = _get_dir(package, 'INCDIR', kw, env, prefix)
    if inc_dir is not None :

        trace("include_dir: %s" % inc_dir, "standardExternalPackage", 5)

        # make 'geninc' directory if not there yet
        archinc = Dir(env.subst("$ARCHINCDIR"))
        archinc = str(archinc)
        if not os.path.isdir(archinc) : os.makedirs(archinc)

        includes = kw.get('INCLUDES')
        if not includes :

            # link the whole include directory
            target = pjoin(archinc, package)
            if not os.path.lexists(target) :
                targ = os.symlink(inc_dir, target)
                env['ALL_TARGETS']['INCLUDES'].append(targ)

        else:

            # make target package directory if needed
            targetdir = pjoin(archinc, package)
            if not os.path.isdir(targetdir) : os.makedirs(targetdir)

            # copy individual files
            includes = _glob(inc_dir, includes)
            for inc in includes :
                loc = pjoin(inc_dir, inc)
                target = pjoin(targetdir, inc)
                targ = env.Symlink(target, loc)
                env['ALL_TARGETS']['INCLUDES'].extend(targ)
                trace("linkinc: %s -> %s" % (str(targ[0]), loc), "standardExternalPackage", 5)


    # link python directory
    py_dir = _get_dir(package, 'PYDIR', kw, env, prefix)
    if py_dir is not None :
        
        # make 'python' directory if not there yet
        archpy = Dir(env.subst("$PYDIR"))
        archpy = str(archpy)
        if not os.path.isdir(archpy) : os.makedirs(archpy)

        trace("py_dir: %s" % py_dir, "standardExternalPackage", 5)
        if kw.get('PYDIRSEP', False) :
            # make a link to the whole dir
            targ = env.Symlink(Dir(pjoin(env.subst("$PYDIR"), package)), Dir(py_dir))
            env['ALL_TARGETS']['LIBS'].extend(targ)
        else :
            # make links for every file in the directory
            files = kw.get('LINKPY')
            files = _glob(py_dir, files)
            for f in files :
                loc = pjoin(py_dir, f)
                if not os.path.isdir(loc) :
                    targ = env.Symlink(pjoin(env.subst("$PYDIR"), f), loc)
                else :
                    targ = env.Symlink(Dir(pjoin(env.subst("$PYDIR"), f)), Dir(loc))
                trace("linkpy: %s -> %s" % (str(targ[0]), loc), "standardExternalPackage", 5)
                env['ALL_TARGETS']['LIBS'].extend(targ)


    # link all libraries
    lib_dir = _get_dir(package, 'LIBDIR', kw, env, prefix)
    if lib_dir is not None :
        trace("lib_dir: %s" % lib_dir, "standardExternalPackage", 5)

        # list of libraries to copy
        copylibs = kw.get('COPYLIBS')
        trace("copylibs: %s" % copylibs, "standardExternalPackage", 5)
        if copylibs:
            copylibs = _glob(lib_dir, copylibs)
            trace("copylibs: %s" % copylibs, "standardExternalPackage", 5)
            for f in copylibs :
                loc = pjoin(lib_dir, f)
                if os.path.isfile(loc) :
                    #targ = env.Install( "$LIBDIR", loc )
                    targ = env.Install(env.subst("$LIBDIR"), loc)
                    trace("copylib: %s -> %s" % (loc, str(targ[0])), "standardExternalPackage", 5)
                    env['ALL_TARGETS']['LIBS'].extend(targ)

        # make a list of libs to link
        libraries = kw.get('LINKLIBS')
        trace("libraries: %s" % libraries, "standardExternalPackage", 5)
        if not libraries and copylibs:
            # if COPYLIBS is there but not LINKLIBS do not lin anything
            libraries = []
        else:
            # even if LINKLIBS is empty link all libraries
            libraries = _glob(lib_dir, libraries)

        trace("libraries: %s" % libraries, "standardExternalPackage", 5)
        for f in libraries :
            loc = pjoin(lib_dir, f)
            if os.path.isfile(loc) :
                #targ = env.Install( "$LIBDIR", loc )
                targ = env.Symlink(pjoin(env.subst("$LIBDIR"), f), loc)
                trace("linklib: %s -> %s" % (str(targ[0]), loc), "standardExternalPackage", 5)
                env['ALL_TARGETS']['LIBS'].extend(targ)

    # link all executables
    bin_dir = _get_dir(package, 'BINDIR', kw, env, prefix)
    if bin_dir is not None :
        trace("bin_dir: %s" % bin_dir, "standardExternalPackage", 5)

        # make list of binaries to link
        binaries = kw.get('LINKBINS')
        binaries = _glob(bin_dir, binaries)

        for f in binaries :
            loc = pjoin(bin_dir, f)
            if os.path.isfile(loc) :
                targ = env.Symlink(pjoin(env.subst("$BINDIR"), f), loc)
                env['ALL_TARGETS']['BINS'].extend(targ)

    # add my libs to a package tree
    addPkgLibs(package, kw.get('PKGLIBS', []))

    # add packages that I depend on
    setPkgDeps(package, kw.get('DEPS', []))

    # get package information, if kw PKGINFO is present then use it (wrap
    # into tuple if it is a string)
    pkginfo = None
    if 'PKGINFO' in kw:
        pkginfo = kw.get('PKGINFO')
        if isinstance(pkginfo, types.StringTypes):
            # if contains literal $SIT_ARCH.found replace it with real arch
            pkginfo = pkginfo.replace('$SIT_ARCH.found', arch)
            pkginfo = tuple(pkginfo,)
        elif pkginfo is not None:
            # if contains literal $SIT_ARCH.found replace it with real arch
            pkginfo = tuple([info.replace('$SIT_ARCH.found', arch) for info in pkginfo])
    elif prefix:
        # if PREFIX starts with SIT_EXTERNAL_SW then strip it and split remaining path
        # then add the result to the list in the environment
        sprefix = prefix.split(os.path.sep)
        sextsw = env['SIT_EXTERNAL_SW'].split(os.path.sep)
        if sprefix[:len(sextsw)] == sextsw:
            pkginfo = tuple(sprefix[len(sextsw):])
    if pkginfo:
        env['EXT_PACKAGE_INFO'].setdefault(package, []).append(pkginfo)
        trace("pkginfo: %s" % (pkginfo,), "standardExternalPackage", 4)
        trace("EXT_PACKAGE_INFO: %s" % env['EXT_PACKAGE_INFO'], "standardExternalPackage", 4)


    #
    # update 'DOC_TARGETS' in default environment if DOCGEN is specified
    #
    docgen = kw.get('DOCGEN')
    if docgen:
        if isinstance(docgen, types.StringTypes):
            # string may contain a list of names
            docgen = docgen.split()
        if isinstance(docgen, types.ListType):
            # make dict out of list, value is package name
            docgen = dict([(k, pkg) for k in docgen])
        for gen, dir in docgen.items():
            if dir:
                env = DefaultEnvironment()
                if isinstance(dir, types.StringTypes):
                    # string may contain a list of names
                    dir = dir.split()
                for d in dir:
                    d = env.subst(d)
                    env['DOC_TARGETS'].setdefault(gen, []).append(d)
