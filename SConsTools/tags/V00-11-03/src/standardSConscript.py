#===============================================================================
#
# SConscript fuction for standard SIT package
#
# $Id$
#
#===============================================================================

import os
import sys
import types
from pprint import *
from os.path import join as pjoin

from SCons.Defaults import *
from SCons.Script import *

from SConsTools.trace import *
from SConsTools.dependencies import *
from SConsTools.scons_functions import *


_cplusplus_ext = [ 'cc', 'cpp', 'cxx', 'C', 'c' ]
_cython_ext = [ 'pyx' ]

# normalize source file path
def _normbinsrc ( dir, f ):
    if not os.path.split(f)[0] :
        return os.path.join(dir,f)
    else :
        return f

# get list of strings from kw, single string is split
def _getkwlist ( kw, name ):
    if name in kw :
        res = kw[name]
        if isinstance(res,(str,unicode)) : res = res.split()
        return res
    return []

# get package name
def _getpkg ( kw ) :
    pkg = kw.get('package', None )
    if not pkg : pkg = os.path.basename(os.getcwd())
    return pkg
    

#
# This is the content of the standard SConscript
#
def standardSConscript( **kw ) :

    """ Understands following keywords, all optional:
        LIBS - list of additional libraries needed by this package
        LIBPATH - list of directories for additional libraries
        BINS - dictionary of executables and their corresponding source files
        TESTS - dictionary of test applications and their corresponding source files
        SCRIPTS - list of scripts in app/ directory
        UTESTS - names of the unit tests to run, if not given then all tests are unit tests
        PYEXTMOD - name of the Python extension module, package name used by default
        CCFLAGS - additional flags passed to C/C++ compilers
        NEED_QT - set to True to enable Qt support
        PHPDIR - either string or dictionary, will link directory to arch/../php/ area
        DOCGEN   - if this is is a string or list of strings then it should be name(s) of document 
                   generators, otherwise it is a dict with generator name as key and a list of 
                   file/directory names as values (may also be a string).
    """

    pkg = _getpkg ( kw )
    trace ( "Standard SConscript in `"+pkg+"'", "SConscript", 1 )

    ukw = kw.copy()
    ukw['package'] = pkg
    if 'env' in kw: del ukw['env']
    
    env = DefaultEnvironment().Clone()
    if kw.get('NEED_QT', False):
        # extend environment with QT stuff
        env.Tool('qt4', toolpath = env['TOOLPATH'])
        ukw.setdefault('LIBS', []).extend(env['QT4_LIBS'])
        ukw.setdefault('LIBPATH', []).append(env['QT4_LIBDIR'])

        standardMoc( env, **ukw )
        
    standardLib( env, **ukw )
    pymod = standardPyLib( env, **ukw )
    pyext = standardPyExt( env, **ukw )
    standardPhpLib( env, **ukw )
    standardScripts( env, **ukw )
    standardBins ( env, **ukw )
    standardTests ( env, **ukw )

    pymods = filter(None, [pymod, pyext])
    defdoc = {'doxy-all': pkg, 'pydoc-all': pymods}
    docgen = kw.get('DOCGEN', defdoc)
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

#
# Process include/ directory, run moc on any include file which contains Q_OBJECT
#
def standardMoc( env, **kw ) :
    
    pkg = _getpkg( kw )
    
    # get headers    
    headers = Glob("include/*.h", source=True)
    headers = [str(h) for h in headers]

    headers = [h for h in headers if 'Q_OBJECT' in file(h).read()]
    trace ( "moc headers = "+str(map(str,headers)), "SConscript", 2 )

    for h in headers:
        base = os.path.splitext(os.path.basename(h))[0]
        env.Moc(pjoin("src",base+"_moc.cpp"), h)


#
# Process src/ directory, make library from all compilable files
#
def standardLib( env, **kw ) :
    
    libsrcs = Flatten ( [ Glob("src/*."+ext, source=True, strings=True ) for ext in _cplusplus_ext ] )
    libsrcs.sort()
    if libsrcs :
        
        trace ( "libsrcs = "+str(map(str,libsrcs)), "SConscript", 2 )

        pkg = _getpkg( kw )
        
        libdir = env['LIBDIR']
        extalibs = _getkwlist ( kw, 'LIBS' )
        extalibpath = _getkwlist ( kw, 'LIBPATH' )


        binkw = {}
        binkw['LIBS'] = extalibs
        binkw['LIBPATH'] = extalibpath + env['LIBPATH']
        if 'CCFLAGS' in kw:
            binkw['CCFLAGS'] = env['CCFLAGS'] + ' ' + kw['CCFLAGS']
        lib = env.SharedLibrary ( pkg, source=libsrcs, **binkw )
        ilib = env.Install ( libdir, source=lib )
        DefaultEnvironment()['ALL_TARGETS']['LIBS'].extend ( ilib )
        
        # get the list of libraries need for this package
        libs = [pkg] + extalibs
        addPkgLib ( pkg, lib[0] )
        addPkgLibs ( pkg, libs, extalibpath )
        
        return lib
        
#
# Process src/ directory, link python sources
#
def standardPyLib( env, **kw ) :

    pysrcs = Glob("src/*.py", source=True, strings=True )
    if pysrcs :
        
        pkg = _getpkg( kw )
        
        pydir = env['PYDIR']

        trace ( "pysrcs = "+str(map(str,pysrcs)), "SConscript", 2 )

        # python files area installed into python/Package
        doinit = True
        for src in pysrcs :
            
            # make symlink for every .py file and compile it into .pyc
            basename = os.path.basename(src)
            if basename == "__init__.py" : doinit = False
            pydst = pjoin(pydir,pkg,basename)
            env.SymlinkRel ( pydst, source=src )
            pyc = env.PyCompile ( pydst+"c", source=pydst )
            DefaultEnvironment()['ALL_TARGETS']['LIBS'].extend ( pyc )

            # target is fake and is never created
            lint = env.Pylint(pydst+".pylint~", source=pydst)
            DefaultEnvironment()['ALL_TARGETS']['PYLINT'].extend(lint)

        if doinit :
            # make __init__.py and compile it
            ini = pjoin(pydir,pkg,"__init__.py")
            env.Command ( ini, "", [ Touch("$TARGET") ] )
            pyc = env.PyCompile ( ini+"c", source=ini )
            DefaultEnvironment()['ALL_TARGETS']['LIBS'].extend ( pyc )

        return pkg

#
# Process pyext/ directory, build python extension module
#
def standardPyExt( env, **kw ) :

    pkg = _getpkg( kw )

    # check for Cython files first    
    cysrcs = Flatten([MyGlob("pyext/*."+ext, source=True, strings=True, recursive=True) for ext in _cython_ext])
    trace ( "cysrcs = "+str(map(str, cysrcs)), "SConscript", 2 )
    extsrcs = [env.Cython(src) for src in cysrcs]
    trace ( "pyextsrc = "+str(map(str, extsrcs)), "SConscript", 2 )

    # this glob will find *.c files produced by Cython so I don't add above files
    extsrcs = Flatten([MyGlob("pyext/*."+ext, source=True, strings=True, recursive=True) for ext in _cplusplus_ext])
    if extsrcs :
        
        trace ( "pyextsrc = "+str(map(str, extsrcs)), "SConscript", 2 )
        
        pydir = env['PYDIR']

        extmodname = kw.get('PYEXTMOD', pkg)

        objects = [env.PythonObject(src) for src in extsrcs]
        # if package builds standard library then add it to the link
        libs = DefaultEnvironment()['PKG_TREE_LIB'].get(pkg, [])
        if libs: libs = [pkg]
        trace ( "pyext libs = "+str(map(str, libs)), "SConscript", 2 )
        extmod = env.PythonExtension ( extmodname, source=objects, LIBS=libs)
        iextmod = env.Install ( pydir, source=extmod )
        DefaultEnvironment()['ALL_TARGETS']['LIBS'].extend ( iextmod )
        
        # get the list of libraries need for this package
        addPkgLib ( pkg, extmod[0] )
        
        return extmodname

#
# Process content of PHPDIR argument, create directories in $ARCHDIR/php.
# If PHPDIR is a dictionary then it will create symlinks
#    $ARCHDIR/php/<key> -> Package/<value> 
# for each key:value pair in the dictionary. If PHPDIR is a string it will
# create symlink
#    $ARCHDIR/php/Package -> Package/PHPDIR
#
def standardPhpLib( env, **kw ) :

    phpdir = kw.get('PHPDIR')
    if not phpdir: return
    
    pkg = _getpkg( kw )
    
    # if PHPDIR is not a dict then make it a dict with a key equal to package name
    if not isinstance(phpdir, dict): phpdir = { pkg: phpdir }

    for link, dir in phpdir.items():
        dstdir = env.subst(pjoin(env['PHPDIR'], link))
        trace ( "php link = "+dstdir, "SConscript", 2 )
        src = env.subst(pjoin('#'+pkg, dir))
        trace ( "php src = "+str(src), "SConscript", 2 )
        dst = env.SymlinkRel(Dir(dstdir), source=Dir(src))
        DefaultEnvironment()['ALL_TARGETS']['LIBS'].extend(dst)

#
# Process app/ directory, install all scripts
#
def standardScripts( env, **kw ) :
    
    targets = _standardScripts ( env, 'app', 'SCRIPTS', env['BINDIR'], **kw )
    DefaultEnvironment()['ALL_TARGETS']['BINS'].extend( targets )

#
# Process app/ directory, build all executables from C++ sources
#
def standardBins( env, **kw ) :
    
    targets = _standardBins ( env, 'app', 'BINS', True, **kw )
    DefaultEnvironment()['ALL_TARGETS']['BINS'].extend( targets )

#
# Process test/ directory, build all executables from C++ sources
#
def standardTests( env, **kw ) :
    
    #trace ( "Build env = "+pformat(env.Dictionary()), "<top>", 7 )

    # binaries in the test/ directory
    targets0 = _standardBins ( env, 'test', 'TESTS', False, **kw )
    DefaultEnvironment()['ALL_TARGETS']['TESTS'].extend( targets0 )

    # also scripts in the test/ directory
    targets1 = _standardScripts ( env, 'test', 'TEST_SCRIPTS', "", **kw )
    DefaultEnvironment()['ALL_TARGETS']['TESTS'].extend( targets1 )
    
    targets = targets0 + targets1

    # make a list of unit tests
    utests = kw.get('UTESTS', None)
    if utests is None :
        utests = targets
    else :
        # filter matching targets
        utests = [ t for t in targets if os.path.basename(str(t)) in utests ]

    # make new unit test target
    trace ( "utests = "+str(map(str,utests)), "SConscript", 2 )
    for u in utests :
        t = env.UnitTest ( str(u)+'.utest', u )
        DefaultEnvironment()['ALL_TARGETS']['TESTS'].extend( t )
        # dependencies for unit tests are difficult to figure out,
        # especially for Python scripts, so make them run every time
        env.AlwaysBuild(t)

#
# Build binaries, possibly install them
#
def _standardBins( env, appdir, binenv, install, **kw ) :

    # make list of binaries and their dependencies if it has not been passed to us
    bins = kw.get(binenv,{})
    if bins :
        for k in bins.iterkeys() :
            src = bins[k]
            if isinstance(src,(str,unicode)) : src = src.split()
            src = [ _normbinsrc(appdir,s) for s in src ]
            bins[k] = src
    else :
        cpps = Flatten ( [ Glob(appdir+"/*."+ext, source=True, strings=True ) for ext in _cplusplus_ext ] )
        for f in cpps :
            bin = os.path.splitext(os.path.basename(f))[0]
            bins[bin] = [ f ]
            
    # make rules for the binaries
    targets = []
    if bins :

        trace ( "bins = "+str(map(str,bins)), "SConscript", 2 )

        bindir = env['BINDIR']
        
        # Program options
        binkw = {}
        binkw['LIBS'] = _getkwlist ( kw, 'LIBS' )
        #binkw['LIBS'].insert ( 0, _getpkg( kw ) )
        env.Prepend(LIBPATH = _getkwlist ( kw, 'LIBPATH' ))
        if 'CCFLAGS' in kw:
            binkw['CCFLAGS'] = env['CCFLAGS'] + ' ' + kw['CCFLAGS']
    
        for bin, srcs in bins.iteritems() :
            
            b = env.Program( bin, source=srcs, **binkw )
            setPkgBins ( kw['package'], b[0] )
            if install : 
                b = env.Install ( bindir, source=b )
                
            targets.extend( b )
            
    return targets

#
# Process app/ directory, install all scripts
#
def _standardScripts( env, appdir, binenv, installdir, **kw ) :

    scripts = kw.get(binenv,None)
    if scripts is None :
        # grab any file without extension in app/ directory
        scripts = Glob(appdir+"/*", source=True, strings=True )
        scripts = [ ( f, str(Entry(f)) ) for f in scripts ]
        scripts = [ s[0] for s in scripts if not os.path.splitext(s[1])[1] and os.path.isfile(s[1]) ]
    else :
        scripts = [ _normbinsrc(appdir,s) for s in scripts ]

    trace ( "scripts = "+str(map(str,scripts)), "SConscript", 2 )

    # Scripts are installed in 'installdir' directory
    targets = []
    for s in scripts : 
        dst = pjoin(installdir,os.path.basename(s))
        trace ( "install script = "+dst, "SConscript", 2 )
        script = env.ScriptInstall(dst, s)
        targets.extend ( script )

    return targets
