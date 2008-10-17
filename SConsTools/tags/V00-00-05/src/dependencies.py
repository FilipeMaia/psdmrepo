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
        if os.path.isfile(f) :
            f = f.split(os.sep)
            if len(f) > 4 and f[-3] == 'geninc' and f[-5] == 'arch' :
                trace ( 'Child comes from %s' % f[-2], 'findAllDependencies', 8 )
                res.add ( f[-2] )
            elif len(f) > 2 and f[-3] == 'include' :
                trace ( 'Child comes from %s' % f[-2], 'findAllDependencies', 8 )
                res.add ( f[-2] )
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
        pkg_info['DEPS'] = deps

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
def _genAllDeps ( pkg_tree, deps, visited ) :
    
    for d in deps :
        if d in pkg_tree :
            for c in _genAllDeps ( pkg_tree, pkg_tree[d].get('DEPS',[]), visited ) :
                yield c
            if d not in visited :
                visited.add ( d )
                yield d
            
            
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
        visited = set()
        alldeps = [ x for x in _genAllDeps(pkg_tree,bindeps,visited) ]
        alldeps.reverse()
        
        trace ( 'bin: %s alldeps: %s' % ( pformat(bin.env.Dictionary()), alldeps ), 'adjustPkgDeps', 4 )

        # now get all their libraries and add to the binary
        for d in alldeps :
            libs = pkg_tree[d].get( 'LIBS', [] )
            bin.env['LIBS'].extend ( libs ) 
