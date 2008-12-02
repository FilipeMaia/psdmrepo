#===============================================================================
#
# Main SCons script for LUSI release building
#
# $Id: builders.py 34 2008-10-13 23:58:56Z salnikov $
#
#===============================================================================

"""
This module is for managing the dependencies between the packages
in the LUSI releases.

It keeps the the dependency graph in the construction environment.
The main structure is represented as a dictionary with one entry 
per package. The key is the package name and the value is another 
dictionary with these keys:

  'DEPS' -> list of the packages that this package depends on
  'LIBS' -> list of library names that this package provides
  'LIBDIRS' -> list of directories where the libraries live
  
There are two dictionaries kept in the construction environment:

  'PKG_TREE_BASE' - the dependency tree for base release(s)
  'PKG_TREE'- dependency tree for current (local) release
  
PKG_TREE_BASE is read from the file(s) in the corresponding base release(s).
PKG_TREE is built by the SConsTools and then saved in the file.

One more dictionary with the environment key 'PKG_TREE_BINS' keeps 
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
        }
def _guessBoostPackage ( p ) :
    return _boostPackages.get ( p, 'boost' )

def _guessPackage ( path ):
    
    f = path.split(os.sep)
    f.reverse() # for easier counting and reverse searching
    
    #
    # First try to see if it comes from boost, in which case it
    # will be in the form .../arch/$LUSI_ARCH/geninc/boost/.....
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
        if f[x+2] == 'arch' :
            # .../arch/$LUSI_ARCH/geninc/Package/file
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
        # .../arch/${LUSI_ARCH}/genarch/Package/ or include/Package/ directory
        f = str(child)
        trace ( 'Checking child %s' % f, 'findAllDependencies', 8 )
        p = _guessPackage ( f )
        if p : 
            res.add ( p )
        else :
            res.update ( findAllDependencies(child) )
        
    return res

#
# Define package libraries - everything that has to be linked to application
#
def setPkgLibs ( env, pkg, libs, libdirs = [] ):

    pkg_info = env['PKG_TREE'].setdefault( pkg, {} )
    if libs :
        if isinstance(libs,(str,unicode)) : libs = libs.split()
        pkg_info['LIBS'] = libs
    if libdirs :
        if isinstance(libdirs,(str,unicode)) : libdirs = libdirs.split()
        pkg_info['LIBDIRS'] = libdirs

#
# Define package dependencies - the list of other package names that should
# be linked when the package library is linked to the application
#
def setPkgDeps ( env, pkg, deps ):
    
    pkg_info = env['PKG_TREE'].setdefault( pkg, {} )
    if deps : 
        if isinstance(deps,(str,unicode)) : deps = deps.split()
        # do not include self-dependencies
        pkg_info['DEPS'] = [ d for d in deps if d != pkg ]

#
# Define binary dependencies
#
def setBinDeps ( env, bin, deps ):
    
    pkg_info = env['PKG_TREE_BINS'][bin] = deps

#
# Store package dependency data in a file
#
def storePkgDeps ( env, fileName ):
    
    trace ( 'Storing release dependencies in file %s' % fileName, 'storePkgDeps', 2 )
    f = open ( fileName, 'wb' )
    cPickle.dump( env['PKG_TREE'], f )
    f.close()

#
# Restore package dependency data from a file
#
def loadPkgDeps ( env, fileName  ):
    
    trace ( 'Loading release dependencies from file %s' % fileName, 'loadPkgDeps', 2 )
    f = open ( fileName, 'rb' )
    env['PKG_TREE_BASE'].update( cPickle.load( f ) )
    f.close()

#
# generator method for DFS scan of dependency tree
#
class _CycleError ( Exception ) :
    def __init__ (self, pkg1, pkg2):
        Exception.__init__ ( self, "Dependecy cycle detected between packages "+pkg1+" and "+pkg2 )

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
def adjustPkgDeps ( env ) :

    trace ( 'Resolving release dependencies', 'adjustPkgDeps', 2 )
    
    # complete package tree which includes base and local releases
    pkg_tree = env['PKG_TREE_BASE'].copy()
    pkg_tree.update( env['PKG_TREE'] )

    for bin, bindeps in env['PKG_TREE_BINS'].iteritems() :
 
        # build ordered list of all dependencies
        alldeps = []
        for d in bindeps :
            for c in _toposort( pkg_tree, d, {} ) :
                alldeps.append ( c )
        alldeps.reverse()
        
        # now get all their libraries and add to the binary
        for d in alldeps :
            libs = pkg_tree.get(d,{}).get( 'LIBS', [] )
            bin.env['LIBS'].extend ( libs ) 
