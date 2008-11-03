#===============================================================================
#
# Main SCons script for LUSI release building
#
# $Id$
#
#===============================================================================

import os
import sys
import re

from SCons.Script import *

from trace import *
from scons_functions import *

def _fmtList ( lst ):
    return '[' + ','.join(map(str,target)) + ']'

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
            return "Creating symlink: \"" + str(target[0]) + "\""
        except :
            return 'MakeSymlink('+_fmtlist(target)+', '+_fmtlist(target)+')'
            

class _pyCompile :
    
    def __call__( self, target, source, env ) :
        """Both target and source should be a single file"""
        if len(target) != 1 :
            fail ( "unexpected number of targets for pyCompile: "+str(target) )
        if len(source) != 1 :
            fail ( "unexpected number of sources for pyCompile: "+str(source) )
    
        target = str(target[0])
        source = str(source[0])
        trace ( "Executing pycompile `%s'" % ( source ), "pyCompile", 3 )
    
        try :
            import py_compile
            py_compile.compile ( source, target, doraise = True )
        except py_compile.PyCompileError, e :
            print str(e)
            return -1
        
    def strfunction ( self, target, source, env ):
        try :
            return "Compiling Python code: \"" + str(source[0]) + "\""
        except :
            return 'PyCompile('+_fmtlist(target)+', '+_fmtlist(target)+')'
            

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
            return 'ScriptInstall('+_fmtlist(target)+', '+_fmtlist(target)+')'
            

class _unitTest :
    
    def __call__ ( self, target, source, env ) :
        """Both target and source should be a single file"""
        if len(target) != 1 :
            fail ( "unexpected number of targets for unitTest: "+str(target) )
        if len(source) != 1 :
            fail ( "unexpected number of sources for unitTest: "+str(source) )
    
        out = str(target[0])
        bin = str(source[0])
    
        try :
    
            cmd = bin+ ' > ' + out + ' 2>&1' 
            trace ( "Executing unitTest `%s'" % ( bin ), "unitTest", 3 )
            ret = os.system ( cmd )
    
            if ret != 0 :
                l = '*** Unit test failed, check log file '+out+' ***'
                s = '*'*len(l)
                print s
                print l
                print s
            else :
                print "UnitTest successful: "+bin
            
        except :
            
            print 'Failure running unit test '+out
            
    def strfunction ( self, target, source, env ):
        try :
            return "Running UnitTest: \"" + str(source[0]) + "\""
        except :
            return 'UnitTest('+_fmtlist(target)+', '+_fmtlist(target)+')'
            

def setupBuilders ( env ):
    
    builders = {
        'Symlink' : Builder ( action = _makeSymlink() ),
        'PyCompile' : Builder ( action = _pyCompile() ),
        'ScriptInstall' : Builder ( action = _scriptInstall() ),
        'UnitTest' : Builder ( action = _unitTest() ),
        }

    env.Append( BUILDERS = builders )
