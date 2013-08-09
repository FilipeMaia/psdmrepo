#===============================================================================
#
# Main SCons script for SIT release building
#
# $Id$
#
#===============================================================================

"""
This module is for managing the dependencies between the packages
in the SIT releases.

It keeps the the dependency graph in the construction environment.
The main structure is represented as a dictionary with one entry 
per package. The key is the package name and the value is another 
dictionary with these keys:

  'DEPS' -> list of the packages that this package depends on
  'LIBS' -> list of library names that this package provides
  'LIBDIRS' -> list of directories where the libraries live
  
There are few dictionaries kept in the construction environment:

  'PKG_TREE_BASE' - the dependency tree for base release(s)
  'PKG_TREE'- dependency tree for current (local) release
  'PKG_TREE_LIB' -> libraries built by the package
  'PKG_TREE_BINS' -> binaries built by the package
  
PKG_TREE_BASE is read from the file(s) in the corresponding base release(s).
PKG_TREE is built by the SConsTools and then saved in the file.

One more dictionary with the environment key 'PKG_TREE_BINDEPS' keeps 
the dependencies of every executable. It is a dictionary with the key
being the Node object of the built executable and the value as a list
of the package names that executable needs to link to.
 
"""

import os
import sys
import re
import cPickle
from pprint import *

from SCons.Script import *

from trace import *
from scons_functions import *

#
# Guess package name from the path of the (include) file
#
_boostPackages = {
        'date_time' : 'boost_date_time',
        'date_time.hpp' : 'boost_date_time',
        'filesystem' : 'boost_filesystem', 
        'filesystem.hpp' : 'boost_filesystem', 
        'iostreams' : 'boost_iostreams',
        'regex' : 'boost_regex',
        'cregex.hpp' : 'boost_regex',
        'regex.hpp' : 'boost_regex',
        'regex.h' : 'boost_regex',
        'thread' : 'boost_thread',
        'thread.hpp' : 'boost_thread',
        'test' : 'boost_unit_test_framework', 
        'python' : 'boost_python', 
        'python.hpp' : 'boost_python', 
        }
def _guessBoostPackage ( p ) :
    return _boostPackages.get ( p, 'boost' )

def _guessPackage ( path ):
    
    f = path.split(os.sep)
    f.reverse() # for easier counting and reverse searching
    
    trace ( 'path: %s' % f, '_guessPackage', 9 )
    #
    # First try to see if it comes from boost, in which case it
    # will be in the form .../arch/$SIT_ARCH/geninc/boost/.....
    #
    if len(f) > 4 :
        try :
            i = f.index('geninc')
            if i > 1 and i+2 < len(f) and f[i-1] == 'boost' and f[i+2] == 'arch' :
                p = _guessBoostPackage ( f[i-2] )
                if p : 
                    trace ( 'Child comes from boost/%s' % p, '_guessPackage', 8 )
                    return p
        except :
            # probably not boost, do other tests
            pass
    
    try :
        x = f.index('geninc') 
        if x > 2 and f[x+2] == 'arch' and f[x-1] == 'pdsdata':
            # .../arch/$SIT_ARCH/geninc/pdsdata/package/File
            if f[x-2] == 'xtc' : 
                pkg = 'pdsdata'
            else:
                pkg = 'pdsdata_' + f[x-2]
            trace ( 'Child comes from %s' % pkg, '_guessPackage', 8 )
            return pkg
        if f[x+2] == 'arch' :
            # .../arch/$SIT_ARCH/geninc/Package/file
            trace ( 'Child comes from %s' % f[x-1], '_guessPackage', 8 )
            return f[x-1]
    except :
        pass
        
    if len(f) > 2 and f[2] == 'include' :
        
        # .../include/Package/file
        trace ( 'Child comes from %s' % f[1], '_guessPackage', 8 )
        return f[1]

#
# Returns the list of all packages that given node depends upon.
# Only direct dependencies are evaluated. Analyzes all SCons children
# and looks for the include files. The directory name where include 
# file is located gives the name of the package.
#
def findAllDependencies( node ):

    res = set()
    for child in node.children() :
        # take all children which are include files, i.e. they live in
        # .../arch/${SIT_ARCH}/genarch/Package/ or include/Package/ directory
        f = str(child)
        trace ( 'Checking child %s' % f, 'findAllDependencies', 8 )
        p = _guessPackage ( f )
        if p : 
            res.add ( p )
        res.update ( findAllDependencies(child) )
        
    return res

#
# Define package libraries - everything that has to be linked to application
#
def addPkgLibs ( pkg, libs, libdirs = [] ):

    env = DefaultEnvironment()
    if libs :
        pkg_info = env['PKG_TREE'].setdefault( pkg, {} )
        if isinstance(libs,(str,unicode)) : libs = libs.split()
        pkg_info.setdefault('LIBS', []).extend(libs)
    if libdirs :
        pkg_info = env['PKG_TREE'].setdefault( pkg, {} )
        if isinstance(libdirs,(str,unicode)) : libdirs = libdirs.split()
        pkg_info.setdefault('LIBDIRS', []).extend(libdirs)

#
# Define package library
#
def addPkgLib ( pkg, lib ):

    env = DefaultEnvironment()
    pkg_info = env['PKG_TREE_LIB'].setdefault(pkg, []).append(lib)

#
# Define package library
#
def setPkgBins ( pkg, bin ):

    env = DefaultEnvironment()
    pkg_info = env['PKG_TREE_BINS'].setdefault( pkg, [] ).append(bin)

#
# Define package dependencies - the list of other package names that should
# be linked when the package library is linked to the application
#
def setPkgDeps ( pkg, deps ):
    
    env = DefaultEnvironment()
    if deps : 
        pkg_info = env['PKG_TREE'].setdefault( pkg, {} )
        if isinstance(deps,(str,unicode)) : deps = deps.split()
        # do not include self-dependencies
        pkg_info['DEPS'] = [ d for d in deps if d != pkg ]

#
# Store package dependency data in a file
#
def storePkgDeps ( fileName ):
    
    env = DefaultEnvironment()
    trace ( 'Storing release dependencies in file %s' % fileName, 'storePkgDeps', 2 )
    f = open ( fileName, 'wb' )
    cPickle.dump( env['PKG_TREE'], f )
    f.close()

#
# Restore package dependency data from a file
#
def loadPkgDeps ( fileName  ):
    
    env = DefaultEnvironment()
    trace ( 'Loading release dependencies from file %s' % fileName, 'loadPkgDeps', 2 )
    f = open ( fileName, 'rb' )
    env['PKG_TREE_BASE'].update( cPickle.load( f ) )
    f.close()

#
# generator method for DFS scan of dependency tree
#
class _CycleError ( Exception ) :
    def __init__ (self, pkg1, pkg2):
        Exception.__init__ ( self, "Dependency cycle detected between packages "+pkg1+" and "+pkg2 )

_WHITE = 0
_GRAY = 1
_BLACK = 2
def _toposort ( pkg_tree, pkg, colors ):
    
    colors[pkg] = _GRAY
    
    adj = pkg_tree.get(pkg,{}).get('DEPS',[])
    for a in adj :
        acol = colors.get(a,_WHITE)
        if acol == _GRAY :
            # means cycle
            raise _CycleError ( pkg, a )
        elif acol == _WHITE :
            for c in _toposort ( pkg_tree, a, colors ) :
                yield c
    yield pkg
    colors[pkg] = _BLACK
            
#
# analyze complete dependency tree and adjust dependencies and libs
#
def adjustPkgDeps():

    env = DefaultEnvironment()

    trace ( 'Resolving release dependencies', 'adjustPkgDeps', 2 )

    # complete package tree which includes base and local releases
    pkg_tree = env['PKG_TREE_BASE'].copy()
    pkg_tree.update( env['PKG_TREE'] )

    # evaluate package dependencies for libraries
    for pkg, libs in env['PKG_TREE_LIB'].iteritems() :
        for lib in libs:
            
            trace ( "checking dependencies for library "+str(lib), "adjustPkgDeps", 4 )
            deps = findAllDependencies ( lib )
            # self-dependencies are not needed here
            deps.discard(pkg)
            
            # dirty hack
            if 'boost_python' in deps: lib.env.Append(CPPPATH=env['PYTHON_INCDIR'])
            
            # another dirty hack, RdbMySQL package includes mysql heades but
            # does not need mysql client library 
            if pkg == "RdbMySQL": deps.discard("mysql")
            
            trace ( "package "+pkg+" deps = " + str(map(str,deps)), "adjustPkgDeps", 4 )
            setPkgDeps ( pkg, deps )
            
            # add all libraries from the packages
            for d in deps :
                libs = pkg_tree.get(d,{}).get( 'LIBS', [] )
                lib.env['LIBS'].extend ( libs )
            trace ( str(lib)+" libs = " + str(map(str,lib.env['LIBS'])), "adjustPkgDeps", 4 )

            #
            # This is a hack to tell scons to rescan the library, otherwise
            # it can decide in some cases that it has been scanned already
            # and won't scan it again, and the libraries that we have just
            # added will not be in the dependency list which can result in 
            # the unnecessary rebuilding of the binary
            # 
            if lib.env['LIBS'] : 
                lib.implicit = None
            
    # iterate over all binaries
    for pkg, bins in env['PKG_TREE_BINS'].iteritems() :
        for bin in bins :

            trace ( "checking dependencies for binary "+str(bin), "adjustPkgDeps", 4 )
            bindeps = findAllDependencies ( bin )
 
            # build ordered list of all dependencies
            alldeps = []
            for d in bindeps :
                for c in _toposort( pkg_tree, d, {} ) :
                    alldeps.append ( c )
            alldeps.reverse()
            
            # now get all their libraries and add to the binary
            trace ( str(bin)+" deps = " + str(map(str,alldeps)), "adjustPkgDeps", 4 )
            for d in alldeps :
                libs = pkg_tree.get(d,{}).get( 'LIBS', [] )
                libpath = pkg_tree.get(d,{}).get( 'LIBDIRS', [] )
                bin.env['LIBS'].extend ( libs )
                bin.env['LIBPATH'].extend ( libpath )
            trace ( str(bin)+" libs = " + str(map(str,bin.env['LIBS'])), "adjustPkgDeps", 4 )
            
            #
            # This is a hack to tell scons to rescan the binary, otherwise
            # it can decide in some cases that it has been scanned already
            # and won't scan it again, and the libraries that we have just
            # added will not be in the dependency list which can result in 
            # the unnecessary rebuilding of the binary
            # 
            if bin.env['LIBS'] : 
                bin.implicit = None


class PrintDependencies(object):
    
    def __init__(self, trees, reverse=False):
        """Constructor takes the list of trees"""
        self.tree = {}
        for tree in trees:
            self.tree.update(tree)
        self.reverse = reverse
    
    def __call__(self, *args, **kw):

        deptree = {}
        if self.reverse:
            for pkg in self.tree.keys():
                for dep in self.tree[pkg].get('DEPS', []):
                    deptree.setdefault(dep, []).append(pkg)
        else:
            for pkg in self.tree.keys():
                deps = self.tree[pkg].get('DEPS', [])
                deptree[pkg] = deps
                
        for pkg in sorted(deptree.keys()):
            deps = sorted(deptree[pkg])
            print pkg, "->", ' '.join(deps)
