"""
Tool which will setup a bunch of special scanners.
"""

import os

from SConsTools.trace import *
import SCons.Scanner
import SCons.Tool


class CScanner(SCons.Scanner.ClassicCPP):
    
    def scan(self, node, path=()):

        if node.includes is None:
            # do not need dependencies originating from boost headers
            f = str(node).split(os.sep)
            trace ( 'node: %s' % str(f), 'findAllDependencies', 4 )
            if len(f) > 4 and f[0] == 'arch' and f[2] == 'geninc':
                node.includes = []
    
        return super(SCons.Scanner.ClassicCPP, self).scan(node, path)

def generate(env):

    cscanner = CScanner("CScanner", 
                        "$CPPSUFFIXES",
                        "CPPPATH", 
                        '^[ \t]*#[ \t]*(?:include|import)[ \t]*(<|")([^>"]+)(>|")')
    for suffix in SCons.Tool.CSuffixes:
        SCons.Tool.SourceFileScanner.add_scanner(suffix, cscanner)
    
    trace ( "Initialized special_scanners tool", "special_scanners", 2 )

def exists(env):
    return True
