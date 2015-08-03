"""SCons.Tool.unittest

Tool-specific initialization for UnitTest builder.

AUTHORS:
 - Andy Salnikov

"""

import os
import time

import SCons
from SCons.Builder import Builder
from SCons.Action import Action

from SConsTools.trace import *
from SConsTools.scons_functions import *

def _fmtList ( lst ):
    return '[' + ','.join(map(str, lst)) + ']'

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
            time.sleep(1)
            ret = os.system ( cmd )
    
            if ret != 0 :
                try:
                    logfilecontents = file(out).read()
                except Exception, e:
                    logfilecontents = "-- could not read logfile. Exception received:\n"
                    logfilecontents += str(e)
                    logfilecontents += "\n----------"
                l = '*** Unit test failed, contens of log file: '+out+' ***\n'
                s = '*' * len(l)
                l += logfilecontents 
                print s
                print l
                print s
                return ret
            else :
                print "UnitTest successful: "+bin
            
        except :
            
            print 'Failure running unit test '+out
            
    def strfunction ( self, target, source, env ):
        try :
            return "Running UnitTest: \"" + str(source[0]) + "\""
        except :
            return 'UnitTest('+_fmtlist(target)+', '+_fmtlist(source)+')'

def create_builder(env):
    try:
        builder = env['BUILDERS']['UnitTest']
    except KeyError:
        builder = SCons.Builder.Builder(action = _unitTest())
        env['BUILDERS']['UnitTest'] = builder

    return builder

def generate(env):
    """Add Builders and construction variables for making symlinks."""

    # Create the PythonExtension builder
    create_builder(env)

    trace ( "Initialized UnitTest tool", "UnitTest", 2 )

def exists(env):
    return True
